"""Meta-trained nonlinear fast weights inside LeFlow latent-path planning.

All support labels are observed outcomes of executed macro chunks. The outer
query is excluded from support. Fast tensors are ephemeral, independently fit
per example/episode; source model parameters are never mutated online.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..baselines.leflow import LeFlow
from ..models import positions


def read_memory(weights, keys):
    w0,w1,w2 = weights
    return (F.silu(keys @ w0)*(keys @ w2)) @ w1


def memory_gradients(weights, keys, values, mask):
    """Exact gradient of .5*mean_valid_rows(sum_channels((M(k)-v)^2))."""
    w0,w1,w2 = weights
    gate,linear = keys@w0,keys@w2
    act = F.silu(gate); hidden = act*linear
    error = ((hidden@w1)-values)*mask[...,None]/mask.sum(1).clamp_min(1)[:,None,None]
    dh = error@w1.transpose(-1,-2)
    sig = gate.sigmoid()
    dg = dh*linear*sig*(1+gate*(1-sig))
    return (keys.transpose(-1,-2)@dg, hidden.transpose(-1,-2)@error,
            keys.transpose(-1,-2)@(dh*act))


class FastMemory(nn.Module):
    def __init__(self,c):
        super().__init__(); cfg=c['leflow_ttt']; d=c['encoder']['dim']
        self.dim=cfg['memory_dim']; hidden=cfg['memory_hidden']; self.scale=cfg['residual_scale']
        self.bound=cfg['residual_bound']; self.elastic=cfg['elastic']; self.max_grad=cfg['inner_gradient_cap']
        self.key=nn.Sequential(nn.Linear(2*d+40+16,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,self.dim))
        self.value=nn.Linear(d,self.dim,bias=False)
        self.output=nn.Linear(self.dim,d,bias=False)
        self.query=nn.Sequential(nn.Linear(1024,128),nn.SiLU(),nn.Linear(128,self.dim))
        self.w0=nn.Parameter(torch.randn(self.dim,hidden)/math.sqrt(self.dim))
        self.w1=nn.Parameter(torch.randn(hidden,self.dim)*.1/math.sqrt(hidden))
        self.w2=nn.Parameter(torch.randn(self.dim,hidden)/math.sqrt(self.dim))
        # Bounded learned step size, initially .05, maximum .25.
        self.lr_logit=nn.Parameter(torch.tensor(math.log(.05/(.25-.05))))

    def prior(self,b):
        return tuple(w[None].expand(b,-1,-1) for w in (self.w0,self.w1,self.w2))

    def keys(self,start,actions,predicted):
        p=start.shape[-2]
        spatial=positions(p,16,start.device,start.dtype).expand(*start.shape[:-2],p,16)
        a=actions.flatten(-2)[...,None,:].expand(*start.shape[:-1],40)
        features=torch.cat((F.normalize(start,dim=-1),F.normalize(predicted,dim=-1),a,spatial),-1)
        # RMS normalization retains O(1) coordinates for a small inner learner.
        k=self.key(features)
        return F.normalize(k,dim=-1)*math.sqrt(self.dim)

    def fit(self,history,steps):
        valid=history['mask'][:,:,None,None]
        start,pred,nxt=[torch.where(valid,history[k],0.) for k in ('start','predicted','next')]
        actions=torch.where(history['mask'][:,:,None,None],history['actions'],0.)
        keys=self.keys(start,actions,pred).flatten(1,2)
        values=self.value((nxt-pred)/self.scale).flatten(1,2)
        mask=history['mask'][:,:,None].expand(-1,-1,start.shape[-2]).flatten(1)
        weights=self.prior(len(start)); anchors=weights
        importance=tuple(torch.zeros_like(w) for w in weights)
        lr=.25*self.lr_logit.sigmoid()
        before=read_memory(weights,keys)
        for _ in range(steps):
            grads=memory_gradients(weights,keys,values,mask)
            norm=sum(g.square().sum((1,2)) for g in grads).clamp_min(1e-16).sqrt()
            factor=(self.max_grad/norm).clamp_max(1)[:,None,None]
            updates=tuple(-lr*g*factor for g in grads)
            importance=tuple(.9*f+.1*u.detach().square() for f,u in zip(importance,updates))
            new=[]
            for w,u,anchor,imp in zip(weights,updates,anchors,importance):
                precision=(imp/(imp.mean((1,2),keepdim=True)+1e-8)).clamp_max(10)
                strength=self.elastic*precision
                new.append((w+u+strength*anchor)/(1+strength))
            weights=tuple(new)
        after=read_memory(weights,keys)
        denom=(mask.sum()*self.dim).clamp_min(1)
        diagnostics=dict(support_before=((before-values).square()*mask[...,None]).sum()/denom,
                         support_after=((after-values).square()*mask[...,None]).sum()/denom,
                         fast_delta=sum((w-a).square().sum((1,2)) for w,a in zip(weights,anchors)).sqrt().mean())
        return weights,diagnostics

    def correction(self,weights,start,actions,predicted):
        correction=self.scale*self.output(read_memory(weights,self.keys(start,actions,predicted)))
        return self.bound*(correction/self.bound).tanh()

    def context(self,weights,start,goal):
        q=self.query(torch.cat((start,goal),-1))
        q=F.normalize(q,dim=-1)*math.sqrt(self.dim)
        return read_memory(weights,q[:,None])[:,0]


class AdaptiveLeFlow(LeFlow):
    def __init__(self,c):
        # Native modules are initialized first with the same seed/order as LeFlow.
        super().__init__(c)
        self.cfg=c['leflow_ttt']; self.memory=FastMemory(c)
        self.flow_context=nn.Linear(self.memory.dim,512,bias=False)
        self.inverse_context=nn.Linear(self.memory.dim,512,bias=False)
        nn.init.zeros_(self.flow_context.weight); nn.init.zeros_(self.inverse_context.weight)
        self.inner_steps=self.cfg['inner_steps']

    def set_stage(self,adapter):
        super().set_stage(adapter)
        for module in (self.memory,self.flow_context,self.inverse_context):module.requires_grad_(not adapter)

    def velocity(self,x,t,z0,zg,context):
        flow=self.flow
        cond=flow.start_proj(z0)+flow.goal_proj(zg)+flow.time_embed(t.float())+self.flow_context(context)
        tokens=flow.token_proj(x)+flow.pos_embedding[:,:x.shape[1]]+cond[:,None]
        return flow.out(flow.net(tokens))

    def decode(self,paths,context):
        start,nxt=paths[:,:-1],paths[:,1:]
        x=torch.cat((start,nxt,nxt-start),-1)
        hidden=self.inverse.net[0](x)+self.inverse_context(context)[:,None]
        return self.inverse.net[1:](hidden)

    @torch.no_grad()
    def history(self,batch,world):
        b,s=batch['history_mask'].shape
        start=batch['history_start']; a=batch['history_actions']
        # Frozen support dynamics; imagined outcomes are never target labels.
        pred=world.rollout(start.flatten(0,1),a.flatten(0,1))[:,-1].reshape_as(start)
        return dict(start=start,predicted=pred,next=batch['history_next'],actions=a,mask=batch['history_mask'])

    def forward(self,batch,world=None):
        if self.adapter_stage:return self.adapter(batch)
        z,actions=batch['z'],batch['a']; b=len(z)
        with torch.no_grad():
            compact=self.adapter.encode(z.flatten(0,1)).reshape(b,z.shape[1],512)
            history=self.history(batch,world)
            recorded_prediction=world.rollout(z[:,0],actions)[:,-1]
        weights,diag=self.memory.fit(history,self.inner_steps)
        context=self.memory.context(weights,compact[:,0],compact[:,-1])
        target=compact[:,1:-1]; noise=torch.randn_like(target); t=torch.rand(b,device=z.device)
        noisy=(1-t[:,None,None])*noise+t[:,None,None]*target
        flow_loss=F.mse_loss(self.velocity(noisy,t,compact[:,0],compact[:,-1],context),target-noise)
        normalized=self.decode(compact,context)
        inverse_loss=F.mse_loss(normalized,(actions-self.action_mean)/self.action_std)
        pred_actions=normalized*self.action_std+self.action_mean
        predicted=world(z[:,:-1].flatten(0,1),pred_actions.flatten(0,1))
        predicted_compact=self.adapter.encode(predicted).reshape(b,self.horizon,512)
        consistency=F.mse_loss(predicted_compact,compact[:,1:])
        correction=self.memory.correction(weights,z[:,0],actions,recorded_prediction)
        prediction_loss=F.mse_loss((recorded_prediction+correction)/self.memory.scale,z[:,-1]/self.memory.scale)
        # Explicit reconstruction anchors the learned value space to real error.
        residual=(z[:,-1]-recorded_prediction)/self.memory.scale
        reconstruction=F.mse_loss(self.memory.output(self.memory.value(residual)),residual)
        loss=flow_loss+inverse_loss+.1*consistency+self.cfg['outcome_weight']*prediction_loss+self.cfg['reconstruction_weight']*reconstruction
        return loss,dict(flow_loss=flow_loss.detach(),inverse_loss=inverse_loss.detach(),
            consistency_loss=consistency.detach(),outcome_loss=prediction_loss.detach(),
            residual_reconstruction_loss=reconstruction.detach(),
            recorded_raw_outcome_mse=(recorded_prediction-z[:,-1]).square().mean().detach(),
            recorded_adapted_outcome_mse=(recorded_prediction+correction-z[:,-1]).square().mean().detach(),
            support_loss_before=diag['support_before'].detach(),support_loss_after=diag['support_after'].detach(),
            fast_weight_delta=diag['fast_delta'].detach(),inner_steps=loss.new_tensor(float(self.inner_steps)),
            history_fraction=history['mask'].float().mean().detach(),
            action_outside_bounds=(pred_actions.abs()>1).float().mean().detach())

    @torch.no_grad()
    def sample_adapted(self,start,goal,weights,count,steps,budget=None):
        z0=self.adapter.encode(start); zg=self.adapter.encode(goal)
        context=self.memory.context(weights,z0,zg).expand(count,-1)
        z0,zg=z0.expand(count,-1),zg.expand(count,-1)
        x=torch.randn(count,self.horizon-1,512,device=start.device,dtype=start.dtype)
        for i in range(steps):
            if budget is not None:budget.check()
            t=x.new_full((count,),i/steps)
            x=x+self.velocity(x,t,z0,zg,context)/steps
        paths=torch.cat((z0[:,None],x,zg[:,None]),1)
        return paths,self.decode(paths,context)*self.action_std+self.action_mean
