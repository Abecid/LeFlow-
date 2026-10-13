"""Dimension-general ports of the registered controller families.

SAGE and Feedback-WM remain the previous paper-inspired ports, not author
reproductions. Existing MetaWorld code/checkpoints are not modified.
Input z:[B,H+1,P,D], goal:[B,P,D], actions:[B,H,A]. H=5, P=1 for LeWM.
A=action_block*primitive_dim. Policy coordinates use the actuator unit box;
the world adapter alone converts them to pretrained-world action units.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from flow_jepa.models import Trunk, distance, positions, prediction_loss
from flow_jepa.execution.revision import RecurrentWorkspace
from flow_jepa.execution.progress_ttt import ridge_update
from flow_jepa.baselines.diffusion_port import DenoiseUNet, cosine_alpha_bar
from flow_jepa.baselines.vendor.leflow import LatentPathFlow, InverseDynamics, flow_matching_loss


def patch_distance(z,q):
    return 1-F.cosine_similarity(z.float(),q.float(),dim=-1,eps=1.e-8)


def fm_loss(head, condition, teacher):
    eps=torch.randn_like(teacher)
    t=torch.rand(len(teacher),device=teacher.device)
    x=(1-t[:,None,None])*eps+t[:,None,None]*teacher
    return (head(x,t,condition)-(teacher-eps)).square().mean()


class ActionFlow(nn.Module):
    def __init__(self,s):
        super().__init__();self.s=s
        self.action=nn.Linear(s.flat,s.width)
        self.time=nn.Sequential(nn.Linear(1,s.width),nn.SiLU(),nn.Linear(s.width,s.width))
        self.layers=nn.ModuleList([nn.Sequential(nn.LayerNorm(s.width),
            nn.Linear(s.width,2*s.width),nn.SiLU(),nn.Linear(2*s.width,s.width)) for _ in range(3)])
        self.out=nn.Sequential(nn.LayerNorm(s.width),nn.Linear(s.width,s.flat))
        nn.init.normal_(self.out[-1].weight,std=.001);nn.init.zeros_(self.out[-1].bias)

    def forward(self,x,t,c):
        h=self.action(x.flatten(-2))+self.time(t[...,None])+c
        for layer in self.layers:h=h+layer(h)
        return self.out(h).reshape_as(x)

    @torch.no_grad()
    def sample(self,c,n,steps,budget=None,noise=None):
        s=self.s;b=len(c)
        x=torch.randn(b,n,s.horizon,s.ad,device=c.device) if noise is None else noise.clone()
        if x.shape!=(b,n,s.horizon,s.ad):raise ValueError('Wrong action noise shape')
        for i in range(steps):
            if budget is not None:budget.check()
            x=x+self(x,x.new_full((b,n),i/steps),c[:,None])/steps
        return x


class Context(nn.Module):
    def __init__(self,s,history=False):
        super().__init__();self.s=s;self.use_history=history
        self.visual=nn.Linear(s.dim,s.width);self.types=nn.Parameter(torch.randn(3,s.width)*.02)
        self.trunk=Trunk(s.width,3,4)
        if history:
            self.error=nn.Linear(s.dim,s.width);self.motion=nn.Linear(s.dim,s.width)
            self.summary=nn.Linear(3,s.width);self.history_type=nn.Parameter(torch.randn(s.width)*.02)
            self.queries=nn.Parameter(torch.randn(4,s.width)*.02)
            self.core=RecurrentWorkspace(s.width);self.norm=nn.LayerNorm(s.width)
            self.projection=nn.Linear(s.width,s.width,bias=False)

    def forward(self,z,q,g,hist=None):
        tokens=self.trunk(torch.cat([self.visual(v)+self.types[i]
                                    for i,v in enumerate((z,q,g))],dim=1))
        base=tokens.mean(1)
        if not self.use_history:return dict(base=base,condition=base)
        mask=hist['mask'];valid=mask[:,:,None,None]
        hs,hn,hp=[torch.where(valid,hist[k],0.) for k in ('start','next','predicted')]
        qq=q[:,None]
        summary=torch.stack((distance(hs,qq)-distance(hn,qq),
                             distance(hs,qq)-distance(hp,qq),distance(hp,hn)),-1)
        summary=torch.where(mask[:,:,None],summary,0.)
        memory=self.error((hn-hp)*10)+self.motion((hn-hs)*10)
        memory=memory+self.summary(summary*100)[:,:,None]+self.history_type
        memory=memory+positions(self.s.patches,self.s.width,z.device,z.dtype)[None,None]
        memory=memory+positions(self.s.history,self.s.width,z.device,z.dtype)[None,:,None]
        keys=torch.cat((tokens,memory.flatten(1,2)),1)
        padding=torch.cat((torch.zeros(len(z),3*self.s.patches,device=z.device,dtype=torch.bool),
                    (~mask)[:,:,None].expand(-1,-1,self.s.patches).flatten(1)),1)
        state=self.queries[None]+base[:,None]
        state=self.core(state,keys,padding)
        thought=self.norm(state.mean(1))
        return dict(base=base,state=state,thought=thought,score_thought=thought,keys=keys,
                    padding=padding,condition=base+self.projection(thought),start=z)

    def revise(self,ctx,evidence):
        keys=torch.cat((ctx['keys'],evidence),1)
        pad=F.pad(ctx['padding'],(0,evidence.shape[1]),value=False)
        state=self.core(ctx['state'],keys,pad);thought=self.norm(state.mean(1))
        return {**ctx,'state':state,'thought':thought,
                'condition':ctx['base']+self.projection(thought)}


class ActionPolicy(nn.Module):
    def __init__(self,s):
        super().__init__();self.s=s
        self.ctx=Context(s);self.flow=ActionFlow(s)

    def prepare(self,z,q,g,hist=None):
        if self.s.method=='vanilla_action_flow':q=g
        return self.ctx(z,q,g)

    def loss(self,b,world,bank=None,step=1):
        ctx=self.prepare(b['z'][:,0],b['z'][:,-1],b['goal'])
        loss=fm_loss(self.flow,ctx['condition'],b['a'])
        return loss,dict(flow_loss=loss.detach())

    def propose(self,ctx,n,budget=None):
        return self.flow.sample(ctx['condition'],n,self.s.flow_steps,budget).clamp(-1,1)


def factual_history(b,world):
    hs=b['history_start'];bs,hh,p,d=hs.shape
    with torch.no_grad():
        pred=world(hs.flatten(0,1),b['history_action'].flatten(0,1)).reshape(bs,hh,p,d)
    return dict(start=hs,next=b['history_next'],predicted=pred,
                action=b['history_action'],mask=b['history_mask'])


class ErrorHeads(nn.Module):
    def __init__(self,s):
        super().__init__();self.s=s
        def mlp(n):
            net=nn.Sequential(nn.Linear(n,s.width),nn.GELU(),nn.Linear(s.width,1))
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias);return net
        self.prefix=mlp(s.width+s.ad+s.patches)
        self.terminal=mlp(s.width+s.flat+2*s.patches)

    def forward(self,thought,actions,maps):
        a=self.prefix(torch.cat((thought,actions[:,:,0],maps[:,:,0]),-1)).squeeze(-1)
        terminal=.5*(maps[:,:,-2]+maps[:,:,-1])
        t=self.terminal(torch.cat((thought,actions.flatten(2),maps[:,:,0],terminal),-1)).squeeze(-1)
        return .1*torch.stack((a,t),-1).tanh()


def local_scores(maps,corrections=None):
    raw=.5*maps[:,:,0].mean(-1)+.25*(maps[:,:,-2].mean(-1)+maps[:,:,-1].mean(-1))
    cost=raw if corrections is None else raw+.5*corrections.clamp_min(0).sum(-1)
    return dict(cost=cost,raw=raw,maps=maps,correction=(torch.zeros(*raw.shape,2,device=raw.device)
                   if corrections is None else corrections))


class GMMPolicy(nn.Module):
    def __init__(self,s,history=True,mixtures=4):
        super().__init__();self.s=s;self.mixtures=mixtures;self.history=history
        self.ctx=Context(s,history);self.head=nn.Linear(s.width,mixtures*(1+2*s.flat))
        if history:
            self.errors=ErrorHeads(s);nn.init.zeros_(self.ctx.projection.weight)

    def prepare(self,z,q,g,hist=None):
        ctx=self.ctx(z,q,g,hist)
        if self.history:
            for _ in range(3):
                state=self.ctx.core(ctx['state'],ctx['keys'],ctx['padding'])
                thought=self.ctx.norm(state.mean(1));ctx={**ctx,'state':state,'thought':thought,
                    'score_thought':thought,'condition':ctx['base']+self.ctx.projection(thought)}
        return ctx

    def distribution(self,ctx):
        f=self.s.flat;r=self.head(ctx['condition']).reshape(-1,self.mixtures,1+2*f)
        return r[:,:,0],r[:,:,1:1+f].tanh(),.05+.45*r[:,:,1+f:].sigmoid()

    def nll(self,ctx,a):
        logits,means,scales=self.distribution(ctx)
        lp=-.5*(((a.flatten(1)[:,None]-means)/scales).square()+2*scales.log()+math.log(2*math.pi)).sum(-1)
        return -torch.logsumexp(F.log_softmax(logits,-1)+lp,-1).mean()/self.s.flat

    @torch.no_grad()
    def propose(self,ctx,n,budget=None):
        if budget is not None:budget.check()
        logits,means,scales=self.distribution(ctx)
        modes=torch.multinomial(logits.softmax(-1),n,replacement=True)
        ix=modes[:,:,None].expand(-1,-1,self.s.flat)
        a=means.gather(1,ix)+scales.gather(1,ix)*torch.randn(len(logits),n,self.s.flat,device=logits.device)
        a[:,0]=means[torch.arange(len(logits),device=logits.device),logits.argmax(-1)]
        return a.reshape(len(logits),n,self.s.horizon,self.s.ad).clamp(-1,1)

    def score(self,ctx,a,pred,q):
        maps=patch_distance(pred,q[:,None,None])
        corr=self.errors(ctx['score_thought'][:,None].expand(-1,a.shape[1],-1),a,maps)
        return local_scores(maps,corr)

    def calibration(self,b,world,hist):
        z=b['z'];choice=torch.randint(0,3,(len(z),),device=z.device)
        q=torch.where((choice==1)[:,None,None],b['goal'],z[:,-1])
        q=torch.where((choice==2)[:,None,None],z[:,-1].roll(1,0),q)
        ctx=self.prepare(z[:,0],q,b['goal'],hist)
        with torch.no_grad():
            pred=world.rollout(z[:,0],b['a'])
            labels=torch.stack((distance(z[:,1],q)-distance(pred[:,0],q),
                .5*(distance(z[:,-2],q)+distance(z[:,-1],q)-distance(pred[:,-2],q)-distance(pred[:,-1],q))),-1)
        scores=self.score(ctx,b['a'][:,None],pred[:,None],q)
        return F.smooth_l1_loss(scores['correction'][:,0]/.02,labels/.02)

    def loss(self,b,world,bank=None,step=1):
        hist=factual_history(b,world)
        ctx=self.prepare(b['z'][:,0],b['z'][:,-1],b['goal'],hist)
        nll=self.nll(ctx,b['a']);cal=self.calibration(b,world,hist)
        return nll+.1*cal,dict(action_nll=nll.detach(),calibration=cal.detach())


class RepairPolicy(GMMPolicy):
    def __init__(self,s):
        super().__init__(s,history=True);del self.head
        nn.init.eye_(self.ctx.projection.weight)
        with torch.no_grad():self.ctx.projection.weight.mul_(.1)
        self.flow=ActionFlow(s);self.repair=ActionFlow(s)
        self.anchor=nn.Sequential(nn.Linear(s.flat,s.width),nn.LayerNorm(s.width))
        self.evidence=nn.Sequential(nn.Linear(s.flat+s.horizon*s.patches+3,s.width),
            nn.LayerNorm(s.width),nn.GELU(),nn.Linear(s.width,s.width))
        self.fast_keys=nn.Sequential(nn.Linear(3*s.dim+s.ad+16,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,31))
        self.fast_prior=nn.Parameter(torch.zeros(32));self.fast_condition=nn.Linear(32,s.width,bias=False)
        del self.errors.prefix

    def keys(self,z,a,pred,q):
        p=z.shape[-2];pos=positions(p,16,z.device,z.dtype).expand(*z.shape[:-2],p,16)
        a=a[...,None,:].expand(*z.shape[:-1],self.s.ad)
        x=torch.cat((F.normalize(z,dim=-1),F.normalize(pred,dim=-1),F.normalize(q,dim=-1),a,pos),-1)
        k=self.fast_keys(x)
        return F.normalize(torch.cat((k,torch.ones_like(k[...,:1])),-1),dim=-1)

    def prepare(self,z,q,g,hist=None):
        ctx=self.ctx(z,q,g,hist);mask=hist['mask'];valid=mask[:,:,None,None]
        hs,hp,hn=[torch.where(valid,hist[k],0.) for k in ('start','predicted','next')]
        ha=torch.where(mask[:,:,None],hist['action'],0.);target=q[:,None].expand_as(hs)
        k=self.keys(hs,ha,hp,target);labels=(patch_distance(hn,target)-patch_distance(hp,target))/.02
        w=ridge_update(k.flatten(1,2),labels.flatten(1),
            mask[:,:,None].expand_as(labels).flatten(1),self.fast_prior,.125,self.s.patches)
        state=ctx['state']+self.fast_condition(w-self.fast_prior)[:,None]
        thought=self.ctx.norm(state.mean(1))
        return {**ctx,'state':state,'thought':thought,'score_thought':thought,'fast_weights':w,
                'condition':ctx['base']+self.ctx.projection(thought)}

    def score(self,ctx,a,pred,q):
        b,n=a.shape[:2];maps=patch_distance(pred,q[:,None,None])
        start=ctx['start'][:,None].expand(b,n,*q.shape[1:]);target=q[:,None].expand_as(start)
        k=self.keys(start,a[:,:,0],pred[:,:,0],target)
        raw=.02*(k*ctx['fast_weights'][:,None,None]).sum(-1)
        pm=.1*(raw/.1).tanh();terminal=.5*(maps[:,:,-2]+maps[:,:,-1])
        inp=torch.cat((ctx['score_thought'][:,None].expand(-1,n,-1),a.flatten(2),maps[:,:,0],terminal),-1)
        terr=.1*self.errors.terminal(inp).squeeze(-1).tanh()
        return {**local_scores(maps,torch.stack((pm.mean(-1),terr),-1)),'prefix_maps':pm}

    def revise(self,ctx,a,scores):
        f=torch.cat((a.flatten(2),scores['maps'].flatten(2)*100,
                     scores['correction']*100,scores['cost'][...,None]*100),-1).detach()
        ctx=self.ctx.revise(ctx,self.evidence(f))
        index=scores['cost'].detach().argmin(1)
        anchor=a[torch.arange(len(a),device=a.device),index].detach()
        return {**ctx,'anchor':anchor}

    @torch.no_grad()
    def propose(self,ctx,n,budget=None):
        if 'anchor' not in ctx:
            return self.flow.sample(ctx['condition'],n,self.s.flow_steps,budget).clamp(-1,1)
        cond=ctx['condition']+self.anchor(ctx['anchor'].flatten(1))
        u=self.repair.sample(cond,n-1,self.s.repair_steps,budget).clamp(-1,1)
        a=(ctx['anchor'][:,None]+self.s.repair_radius*u).clamp(-1,1)
        return torch.cat((ctx['anchor'][:,None],a),1)

    def loss(self,b,world,bank=None,step=1):
        hist=factual_history(b,world);z=b['z'];ctx=self.prepare(z[:,0],z[:,-1],b['goal'],hist)
        main=fm_loss(self.flow,ctx['condition'],b['a']);repair=main*0
        if step%2==0:
            if bank is None:raise ValueError('Retrieval repair requires a TRAIN-only bank')
            with torch.no_grad():
                a=self.propose(ctx,4)
                retrieved=bank.training_query(z[:,0],z[:,-1],b['episode'])
                enabled=b['index']%8!=7;a[enabled,-1]=retrieved[enabled]
                pred=world.rollout(z[:,0,None].expand(-1,4,-1,-1).flatten(0,1),a.flatten(0,1))
                pred=pred.reshape(len(z),4,self.s.horizon,*z.shape[2:]);scores=self.score(ctx,a,pred,z[:,-1])
            revised=self.revise(ctx,a,scores)
            teacher=((b['a']-revised['anchor'])/self.s.repair_radius).clamp(-1,1)
            repair=fm_loss(self.repair,revised['condition']+self.anchor(revised['anchor'].flatten(1)),teacher)
        cal=self.calibration(b,world,hist)
        return main+repair+.1*cal,dict(flow_loss=main.detach(),repair_loss=repair.detach(),calibration=cal.detach())


class SagePolicy(nn.Module):
    def __init__(self,s):
        super().__init__();self.s=s;self.subctx=Context(s)
        self.duration=nn.Linear(1,s.width);self.subout=nn.Linear(s.width,s.patches*s.dim)
        self.actor=GMMPolicy(s,history=False,mixtures=8)

    def subgoal(self,z,g,duration):
        h=self.subctx(z,g,g)['condition']+self.duration(duration[:,None]/60)
        return z+self.subout(h).reshape_as(z)

    def loss(self,b,world,bank=None,step=1):
        q=self.subgoal(b['z'][:,0],b['goal'],b['duration'].float())
        context=self.actor.prepare(b['z'][:,0],q.detach(),b['goal'])
        nll=self.actor.nll(context,b['a']);sg=prediction_loss(q,b['subgoal'])
        return nll+sg,dict(action_nll=nll.detach(),subgoal_loss=sg.detach())


class DiffusionPolicy(nn.Module):
    def __init__(self,s):
        super().__init__();self.s=s
        self.visual=nn.Linear(s.dim,s.width)
        self.context=nn.Sequential(nn.Linear(2*s.width,s.width),nn.LayerNorm(s.width),nn.SiLU())
        channels=128
        self.net=DenoiseUNet(s.width,channels)
        self.net.head=nn.Conv1d(s.ad,channels,3,padding=1)
        self.net.out[-1]=nn.Conv1d(channels,s.ad,1)
        self.register_buffer('alpha_bar',cosine_alpha_bar(100))

    def condition(self,z,g):
        return self.context(torch.cat((self.visual(z).mean(1),self.visual(g).mean(1)),-1))

    def loss(self,b,world=None,bank=None,step=1):
        a=b['a'];eps=torch.randn_like(a);t=torch.randint(0,100,(len(a),),device=a.device)
        ab=self.alpha_bar[t][:,None,None];x=ab.sqrt()*a+(1-ab).sqrt()*eps
        err=self.net(x,t.float()/99,self.condition(b['z'][:,0],b['goal']))-eps
        loss=err.square().mean();return loss,dict(noise_mse=loss.detach())

    @torch.no_grad()
    def sample(self,z,g,budget=None,guidance=None):
        x=torch.randn(len(z),self.s.horizon,self.s.ad,device=z.device)
        c=self.condition(z,g);ts=torch.linspace(99,0,16,device=z.device).long()
        for i,t in enumerate(ts.tolist()):
            if budget is not None:budget.check()
            eps=self.net(x,x.new_full((len(x),),t/99),c);ab=self.alpha_bar[t]
            clean=((x-(1-ab).sqrt()*eps)/ab.sqrt()).clamp(-1,1)
            if guidance is not None:clean=guidance(clean,i).clamp(-1,1)
            x=clean if i==15 else self.alpha_bar[ts[i+1]].sqrt()*clean+(1-self.alpha_bar[ts[i+1]]).sqrt()*eps
        return x.clamp(-1,1)


class LeFlowPolicy(nn.Module):
    def __init__(self,s):
        super().__init__();self.s=s
        self.flow=LatentPathFlow(s.dim,hidden_dim=512,depth=4,max_horizon=20,time_dim=64,dropout=0)
        self.inverse=InverseDynamics(s.dim,s.ad,hidden_dim=512,depth=3,dropout=0)

    def loss(self,b,world,bank=None,step=1):
        path=b['z'][:,:,0];fm=flow_matching_loss(self.flow,path)
        acts=self.inverse(path[:,:-1].flatten(0,1),path[:,1:].flatten(0,1)).reshape_as(b['a'])
        inverse=F.mse_loss(acts,world.to_world(b['a']))
        predicted=world(b['z'][:,:-1].flatten(0,1),world.from_world(acts.flatten(0,1)))
        consistency=F.mse_loss(predicted.reshape_as(b['z'][:,1:]),b['z'][:,1:])
        return fm+inverse+.1*consistency,dict(flow_loss=fm.detach(),inverse_loss=inverse.detach(),consistency=consistency.detach())

    @torch.no_grad()
    def sample(self,z,g,n=64,budget=None,world=None):
        start=z[:,0].repeat_interleave(n,0);goal=g[:,0].repeat_interleave(n,0)
        x=torch.randn(len(start),self.s.horizon-1,self.s.dim,device=z.device)
        for i in range(16):
            if budget is not None:budget.check()
            x=x+self.flow(x,x.new_full((len(x),),i/16),start,goal)/16
        path=torch.cat((start[:,None],x,goal[:,None]),1)
        a=self.inverse(path[:,:-1].flatten(0,1),path[:,1:].flatten(0,1))
        return world.from_world(a.reshape(len(z),n,self.s.horizon,self.s.ad))


def build(s):
    if s.method=='leflow':return LeFlowPolicy(s)
    if s.method in ('vanilla_action_flow','action_flow_search'):return ActionPolicy(s)
    if s.method=='execution_revision':return GMMPolicy(s)
    if s.method=='retrieval_repair_ttt':return RepairPolicy(s)
    if s.method=='sage_port':return SagePolicy(s)
    if s.method in ('diffusion_policy_port','feedback_wm_port'):return DiffusionPolicy(s)
    raise ValueError(s.method)
