"""Goal-progress readout and causal calibration for bounded flow repair.

Temporal supervision is a weak expert-time proxy, not task-success ground truth.
Generated actions receive no inherited physical-outcome labels. Task identities,
clock time and simulator state are never policy inputs.
"""
import math
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .retrieval_repair import RetrievalRepairSegments, RetrievalRepairPolicy, RetrievalRepairController
from .progress_ttt import ridge_update


class ProgressObjectiveSegments(RetrievalRepairSegments):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row, _, _ = self.sample_spec(index)
        end = min(row['steps'], math.ceil(row['first_success_action']/self.c['data']['action_repeat']))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 7201]))
        earlier = int(rng.integers(max(1, end-5)))
        later = int(rng.integers(min(earlier+5, end), end+1))
        with h5py.File(self.root/row['path'], 'r') as f:
            states = np.stack([f[self.state_key][i].astype(np.float32) for i in (earlier,later,end)])
        item['progress_states'] = torch.from_numpy(((states-self.mean)/self.std).copy())
        scale = self.c['progress_objective']['time_scale_blocks']
        item['progress_labels'] = torch.tensor([min((end-earlier)/scale,1), min((end-later)/scale,1),0.],dtype=torch.float32)
        return item


class GoalProgress(nn.Module):
    """Small spatial readout: preserve patch locations and condition on the goal."""
    def __init__(self, dim, tokens, width):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.project = nn.Linear(dim, width)
        self.head = nn.Sequential(nn.Linear(tokens*width*3,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,1))

    def forward(self, state, goal):
        state, goal = torch.broadcast_tensors(state,goal)
        x=self.project(F.normalize(state.float(),dim=-1)*self.scale)
        g=self.project(F.normalize(goal.float(),dim=-1)*self.scale)
        return self.head(torch.cat((x,g,(x-g).abs()),-1).flatten(-2)).squeeze(-1).sigmoid()


class ProgressObjectivePolicy(RetrievalRepairPolicy):
    def __init__(self, c):
        super().__init__(c)
        self.progress_cfg = c['progress_objective']
        self.goal_progress = GoalProgress(c['encoder']['dim'],self.tokens,self.progress_cfg['width'])
        self.progress_prior = nn.Parameter(torch.zeros(self.rank))
        self.progress_condition = nn.Linear(self.rank,c['controller_grounded']['width'],bias=False)
        nn.init.normal_(self.progress_condition.weight,std=.02)

    def progress_keys(self,start,action,predicted,goal):
        return F.normalize(self.transition_keys(start,action,predicted,goal).mean(-2),dim=-1)

    def fit_progress(self, history, goal):
        mask=history['mask']; valid=mask[...,None,None]
        hs,hp,hn=[torch.where(valid,history[k],0.) for k in ('start','predicted','next')]
        ha=torch.where(mask[...,None],history['action'],0.)
        q=goal[:,None].expand_as(hs)
        keys=self.progress_keys(hs,ha,hp,q)
        # Stop gradients through labels: calibration must not flatten the learned
        # progress function to make its own residual trivially zero.
        with torch.no_grad():
            labels=(self.goal_progress(hn,q)-self.goal_progress(hp,q))/self.progress_cfg['error_scale']
        return ridge_update(keys,labels,mask,self.progress_prior,self.ridge,1)

    def prepare(self,start,target,goal,history,depth=None):
        ctx=super().prepare(start,target,goal,history,depth)
        weights=self.fit_progress(history,goal);delta=weights-self.progress_prior
        state=ctx['state']+self.progress_condition(delta)[:,None]
        thought=self.workspace_norm(state.mean(1))
        return {**ctx,'state':state,'thought':thought,'score_thought':thought,
                'progress_goal':goal,'progress_weights':weights,'progress_delta':delta}

    def progress_correction(self,context,actions,prefix,prior=False):
        b,n=actions.shape[:2]
        start=context['start'][:,None].expand_as(prefix)
        goal=context['progress_goal'][:,None].expand_as(prefix)
        keys=self.progress_keys(start,actions[:,:,0],prefix,goal)
        weights=self.progress_prior.expand(b,-1) if prior else context['progress_weights']
        value=self.progress_cfg['error_scale']*(keys*weights[:,None]).sum(-1)
        bound=self.progress_cfg['correction_bound']
        return bound*(value/bound).tanh()

    def score_candidates(self,context,actions,predicted,target):
        scores=super().score_candidates(context,actions,predicted,target)
        # Same global goal for every local target; incomparable local-goal
        # potentials must not reward a short but irrelevant reach trajectory.
        states=predicted[:,:,[0,3,4]]
        remaining=self.goal_progress(states,context['progress_goal'][:,None,None])
        raw=.5*remaining[:,:,0]+.25*(remaining[:,:,1]+remaining[:,:,2])
        correction=self.progress_correction(context,actions,predicted[:,:,0])
        weight=self.progress_cfg['score_weight']
        return {**scores,'raw':scores['raw']+weight*raw,
                'cost':scores['cost']+weight*(raw+.5*correction.clamp_min(0)),
                'goal_remaining':remaining,'goal_progress_error':correction}

    def auxiliary_loss(self,batch,history,predicted):
        cfg=self.progress_cfg;z=batch['z'];states=batch['progress_states']
        values=self.goal_progress(states,states[:,-1:])
        labels=batch['progress_labels']
        scale=cfg['time_scale_blocks']
        # Also cover the exact hindsight queries used by policy training.
        hindsight=self.goal_progress(z[:,[0,2,3]],z[:,3,None])
        offsets=batch['sample_goal_offset'].float()
        hindsight_labels=torch.stack((offsets,(offsets-5).clamp_min(0),offsets*0),-1).div(scale).clamp(0,1)
        regression=.5*(F.smooth_l1_loss(values,labels,beta=.1)+F.smooth_l1_loss(hindsight,hindsight_labels,beta=.1))
        margin=(labels[:,0]-labels[:,1]).clamp(0,.25)
        order=F.relu(margin-(values[:,0]-values[:,1])).mean()
        # Meta-fit on past observed outcomes; evaluate the NEXT recorded action.
        goal=z[:,3];weights=self.fit_progress(history,goal)
        ctx={'start':z[:,0],'progress_goal':goal,'progress_weights':weights}
        correction=self.progress_correction(ctx,batch['a'][:,None],predicted[:,None,0])[:,0]
        with torch.no_grad():
            error=self.goal_progress(z[:,1],goal)-self.goal_progress(predicted[:,0],goal)
        calibration=F.smooth_l1_loss(correction/cfg['error_scale'],error/cfg['error_scale'])
        prior=self.progress_correction(ctx,batch['a'][:,None],predicted[:,None,0],prior=True)[:,0]
        loss=cfg['loss_weight']*(regression+order)+cfg['calibration_weight']*calibration
        return loss,dict(progress_regression_loss=regression.detach(),progress_order_loss=order.detach(),
            progress_temporal_pair_accuracy=(values[:,0]>values[:,1]).float().mean().detach(),
            progress_temporal_mae=(values-labels).abs().mean().detach(),
            progress_calibration_loss=calibration.detach(),progress_raw_prefix_mae=error.abs().mean(),
            progress_corrected_prefix_mae=(error-correction).abs().mean().detach(),
            progress_prior_prefix_mae=(error-prior).abs().mean().detach())


class ProgressObjectiveController(RetrievalRepairController):
    def target_route_costs(self,context,start,targets,goal,route_cost):
        values=self.model.goal_progress(torch.cat((start,targets)),goal)
        advance=values[:1]-values[1:]
        cfg=self.model.progress_cfg
        # Soft compatibility for the five-block local horizon, never a task-ID
        # filter. Both backward stages and implausibly distant stages pay cost.
        penalty=F.relu(-advance)+F.relu(advance-5/cfg['time_scale_blocks'])
        context['target_progress_advance']=advance
        context['stage_penalty']=cfg['score_weight']*penalty
        return route_cost+context['stage_penalty']

    def plan_diagnostics(self,context,start,targets,actions,prefixes,chosen):
        out=super().plan_diagnostics(context,start,targets,actions,prefixes,chosen)
        goal=context['progress_goal']
        before=self.model.goal_progress(start.expand_as(goal),goal)
        after=self.model.goal_progress(prefixes,goal)
        correction=self.model.progress_correction(context,actions[:,None],prefixes[:,None])[:,0]
        out.update(goal_progress_predicted=float((before-after)[chosen]),
            goal_progress_corrected=float((before-after-correction.clamp_min(0))[chosen]),
            goal_progress_error_correction=float(correction[chosen]),
            goal_progress_fast_delta_norm=float(context['progress_delta'][chosen].norm()),
            candidate_target_stage_advances=context['target_progress_advance'].cpu().tolist(),
            candidate_stage_penalties=context['stage_penalty'].cpu().tolist(),
            candidate_goal_prefix_progress=(before-after).cpu().tolist(),
            candidate_goal_progress_corrections=correction.cpu().tolist(),
            progress_fit_scope='past_executed_prefixes_only_fixed_during_search')
        self.pending_progress=(start.clone(),prefixes[chosen:chosen+1].clone(),goal[chosen:chosen+1].clone(),correction[chosen].clone())
        return out

    def begin_episode(self):
        super().begin_episode();self.pending_progress=None

    @torch.no_grad()
    def observe(self,actual):
        if self.pending_progress is not None:
            start,predicted,goal,correction=self.pending_progress
            before=self.model.goal_progress(start,goal)
            actual_progress=before-self.model.goal_progress(actual,goal)
            forecast=before-self.model.goal_progress(predicted,goal)
            self.trace[-1].update(goal_progress_observed=float(actual_progress),
                goal_progress_raw_error=float((forecast-actual_progress).abs()),
                goal_progress_corrected_error=float((forecast-correction-actual_progress).abs()))
            self.pending_progress=None
        super().observe(actual)
