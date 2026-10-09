"""Causal residual-conditioned latent revision; factual calibration only."""
import math

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data import Segments
from ..models import distance, positions, prediction_loss
from .model import ChunkPolicy


class RevisionSegments(Segments):
    def sample_spec(self, index):
        # Preserve the previous sampler's task/episode/start and goal draws.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        rows = self.tasks[self.names[int(rng.integers(len(self.names)))]]
        row = rows[int(rng.integers(len(rows)))]
        limit = row['steps'] - self.horizon
        if row.get('first_success_action') is not None:
            limit = min(limit, row['first_success_action'] // self.c['data']['action_repeat'])
        start = int(rng.integers(limit + 1))
        grng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 1701]))
        delta = int(grng.choice([5, 10, 20, 40, 60]))
        return row, start, delta

    def __getitem__(self, index):
        row, start, delta = self.sample_spec(index)
        history = self.c['latent_revision']['history_steps']
        left = max(0, start-history)
        with h5py.File(self.root/row['path'], 'r') as f:
            # Dense reads of the short local block avoid repeated HDF5 decoding.
            local = (f[self.state_key][left:start+6].astype(np.float32)-self.mean)/self.std
            goal = (f[self.state_key][start+delta].astype(np.float32)-self.mean)/self.std
            actions = f['actions'][left:start+5].astype(np.float32)
        offset = start-left
        states = np.stack([local[offset], local[offset+1], local[offset+5], goal])
        hs = np.zeros((history, *states.shape[1:]), np.float32)
        hn, ha = np.zeros_like(hs), np.zeros((history, 8), np.float32)
        mask = np.zeros(history, bool)
        if offset:
            hs[-offset:], hn[-offset:] = local[:offset], local[1:offset+1]
            ha[-offset:] = actions[:offset]; mask[-offset:] = True
        return {k:torch.from_numpy(v.copy()) for k,v in dict(z=states,
            z4=local[offset+4], a=actions[offset:offset+5], history_start=hs,
            history_next=hn, history_action=ha, history_mask=mask).items()}


class RecurrentWorkspace(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_norm, self.cross_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4*width),
                                nn.GELU(), nn.Linear(4*width, width))

    def forward(self, state, context, padding):
        q = self.self_norm(state)
        state = state + self.self_attention(q,q,q,need_weights=False)[0]
        state = state + self.cross_attention(self.cross_norm(state),context,context,
                                             key_padding_mask=padding,need_weights=False)[0]
        return state + self.ff(state)


class RevisionPolicy(ChunkPolicy):
    def __init__(self, c):
        super().__init__(c)
        cfg = c['latent_revision']; width = c['controller_grounded']['width']
        self.history_steps, self.reasoning_steps = cfg['history_steps'], cfg['reasoning_steps']
        self.error_scale, self.correction_bound = cfg['error_scale'], cfg['correction_bound']
        self.calibration_weight = cfg['calibration_weight']
        self.tokens = math.prod(c['encoder']['token_grid'])
        self.error_input = nn.Linear(c['encoder']['dim'], width)
        self.movement_input = nn.Linear(c['encoder']['dim'], width)
        self.progress_input = nn.Linear(3, width)
        self.history_type = nn.Parameter(torch.randn(width)*.02)
        self.workspace_queries = nn.Parameter(torch.randn(cfg['workspace_slots'],width)*.02)
        self.core = RecurrentWorkspace(width)
        self.workspace_norm = nn.LayerNorm(width)
        self.policy_context = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.policy_context.weight)
        def readout(dim):
            layer = nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.Linear(width,1))
            nn.init.zeros_(layer[-1].weight); nn.init.zeros_(layer[-1].bias)
            return layer
        # Prefix predictions cannot depend on the unexecuted action suffix.
        self.prefix_error = readout(width+8+self.tokens)
        self.terminal_error = readout(width+40+2*self.tokens)

    def empty_history(self, start):
        b,p,d = start.shape; h=self.history_steps
        blank=start.new_zeros(b,h,p,d)
        return dict(start=blank, next=blank.clone(), predicted=blank.clone(),
                    mask=torch.zeros(b,h,device=start.device,dtype=torch.bool))

    def prepare(self, start, target, goal, history, depth=None):
        b,p,_=start.shape
        x=torch.cat([self.input(v)+self.types[i] for i,v in enumerate((start,target,goal))],1)
        context=self.trunk(x); base=context.mean(1)
        hs,hn,hp=history['start'],history['next'],history['predicted']
        mask=history['mask']
        # Invalid history is zeroed BEFORE projections: arbitrary padding never
        # becomes evidence. Each valid end time is <= the current observation.
        valid=mask[:,:,None,None]
        residual=torch.where(valid,hn-hp,0.)
        motion=torch.where(valid,hn-hs,0.)
        q=target[:,None]
        actual_progress=distance(hs,q)-distance(hn,q)
        predicted_progress=distance(hs,q)-distance(hp,q)
        summary=torch.stack((actual_progress,predicted_progress,distance(hp,hn)), -1)
        summary=torch.where(mask[:,:,None],summary,0.)
        memory=self.error_input(residual*10)+self.movement_input(motion*10)
        memory=memory+self.progress_input(summary*100)[:,:,None]+self.history_type
        memory=memory+positions(p,memory.shape[-1],memory.device,memory.dtype)[None,None]
        memory=memory+positions(self.history_steps,memory.shape[-1],memory.device,memory.dtype)[None,:,None]
        keys=torch.cat((context,memory.flatten(1,2)),1)
        padding=torch.cat((torch.zeros(b,3*p,device=start.device,dtype=torch.bool),
                           (~mask)[:,:,None].expand(-1,-1,p).flatten(1)),1)
        state=self.workspace_queries[None].expand(b,-1,-1)+base[:,None]
        initial=self.workspace_norm(state.mean(1))
        depth=self.reasoning_steps if depth is None else depth
        for _ in range(depth):
            state=self.core(state,keys,padding)
        thought=self.workspace_norm(state.mean(1))
        return dict(base=base,thought=thought,initial=initial,depth=depth)

    def distribution_from_context(self, context):
        h=context['base']+self.policy_context(context['thought'])
        raw=self.output(h).reshape(-1,self.mixtures,81)
        return raw[:,:,0],raw[:,:,1:41].tanh(),.05+.45*raw[:,:,41:].sigmoid()

    def corrections(self, thought, actions, predicted, target):
        token_cost=1-F.cosine_similarity(predicted.float(),target[:,None].float(),dim=-1,eps=1e-8)
        prefix=token_cost[:,0]
        terminal=.5*(token_cost[:,-2]+token_cost[:,-1])
        a=self.prefix_error(torch.cat((thought,actions[:,0],prefix),-1)).squeeze(-1)
        b=self.terminal_error(torch.cat((thought,actions.flatten(1),prefix,terminal),-1)).squeeze(-1)
        return self.correction_bound*torch.stack((a,b),-1).tanh()

    @torch.no_grad()
    def factual_history(self, batch, world):
        hs=batch['history_start']; b,h,p,d=hs.shape
        # One-step independent predictions on RECORDED historical actions.
        pred=world.rollout(hs.flatten(0,1),batch['history_action'].reshape(b*h,1,8))[:,0]
        return dict(start=hs,next=batch['history_next'],predicted=pred.reshape(b,h,p,d),
                    mask=batch['history_mask'])

    def forward(self, batch, world=None, weight=0.):
        z,actions=batch['z'],batch['a']
        history=self.factual_history(batch,world)
        depth=int(torch.randint(1,self.reasoning_steps+1,()).item()) if self.training else self.reasoning_steps
        # Calibration targets must not always be the demonstration endpoint:
        # that would teach a spurious "all candidate terminal costs are zero".
        # Relabel targets, NEVER actions/outcomes, with supported training states.
        choice=torch.randint(0,3,(len(z),),device=z.device)
        q=torch.where((choice==1)[:,None,None],z[:,3],z[:,2])
        q=torch.where((choice==2)[:,None,None],z[:,2].roll(1,0),q)
        # Independent actor/calibration target views share one physical batch
        # of transformer work. This changes neither labels nor loss weighting.
        both=self.prepare(torch.cat((z[:,0],z[:,0])),torch.cat((z[:,2],q)),
            torch.cat((z[:,3],z[:,3])),{k:torch.cat((v,v)) for k,v in history.items()},depth)
        context={k:v[:len(z)] for k,v in both.items() if k!='depth'}
        calibration_context={k:v[len(z):] for k,v in both.items() if k!='depth'}
        logits,means,scales=self.distribution_from_context(context)
        logp=-.5*(((actions.flatten(1)[:,None]-means)/scales).square()+2*scales.log()+math.log(2*math.pi)).sum(-1)
        posterior=F.log_softmax(logits,-1)+logp
        nll=-torch.logsumexp(posterior,-1).mean()/40
        loss=nll
        with torch.no_grad():
            predicted=world.rollout(z[:,0],actions)
            raw_prefix=distance(predicted[:,0],q)
            raw_terminal=.5*(distance(predicted[:,-2],q)+distance(predicted[:,-1],q))
            observed_prefix=distance(z[:,1],q)
            observed_terminal=.5*(distance(batch['z4'],q)+distance(z[:,2],q))
            labels=torch.stack((observed_prefix-raw_prefix,observed_terminal-raw_terminal),-1)
        corrections=self.corrections(calibration_context['thought'],actions,predicted,q)
        initial=self.corrections(calibration_context['initial'],actions,predicted,q)
        error=F.smooth_l1_loss(corrections/self.error_scale,labels/self.error_scale,reduction='none').mean(-1)
        initial_error=F.smooth_l1_loss(initial/self.error_scale,labels/self.error_scale,reduction='none').mean(-1)
        # Supervise the shallow reference too; worsening it cannot lower this
        # loss. The paired term's reference is detached and uses identical data.
        paired=F.relu(error-initial_error.detach()).mean()
        calibration=error.mean()+.25*initial_error.mean()+.25*paired
        loss=loss+self.calibration_weight*calibration
        metrics=dict(action_nll=nll.detach(),calibration_loss=error.mean().detach(),
            shallow_calibration_loss=initial_error.mean().detach(),paired_regression=paired.detach(),
            recorded_raw_prefix_mae=labels[:,0].abs().mean().detach(),
            recorded_corrected_prefix_mae=(labels[:,0]-corrections[:,0]).abs().mean().detach(),
            recorded_raw_terminal_mae=labels[:,1].abs().mean().detach(),
            recorded_corrected_terminal_mae=(labels[:,1]-corrections[:,1]).abs().mean().detach(),
            reasoning_depth=loss.new_tensor(float(depth)),history_fraction=history['mask'].float().mean().detach())
        if weight:
            mode=posterior.detach().argmax(-1)
            proposed=means[torch.arange(len(z),device=z.device),mode].reshape(-1,5,8)
            generated=world.rollout(z[:,0],proposed)
            prefix=prediction_loss(generated[:,0],z[:,1]); endpoint=prediction_loss(generated[:,-1],z[:,2])
            loss=loss+weight*.5*(prefix+endpoint)
            metrics.update(prefix_loss=prefix.detach(),endpoint_loss=endpoint.detach())
        return loss,metrics

    @torch.no_grad()
    def propose_context(self, context, count):
        logits,means,scales=self.distribution_from_context(context)
        modes=torch.multinomial(logits.softmax(-1),count,replacement=True)
        ids=modes[:,:,None].expand(-1,-1,40)
        actions=means.gather(1,ids)+scales.gather(1,ids)*torch.randn(len(logits),count,40,device=means.device)
        actions[:,0]=means[torch.arange(len(logits),device=means.device),logits.argmax(-1)]
        return actions.reshape(len(logits),count,5,8).clamp(-1,1)
