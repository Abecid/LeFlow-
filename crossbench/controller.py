"""Common short-prefix execution, with isolated method-specific search logic."""
from __future__ import annotations
import torch
from flow_jepa.models import distance
from .models import local_scores, patch_distance


class Controller:
    def __init__(self, model, world, bank, spec):
        self.model,self.world,self.bank,self.s=model,world,bank,spec
        self.begin_episode()

    def begin_episode(self):
        self.history=[];self.pending=None;self.previous=None;self.stalls=[]
        self.feedback=None;self.trace=[]

    @torch.no_grad()
    def observe(self, actual):
        if self.pending is None:return
        z,a,pred,q,plan=self.pending
        self.history.append((z,actual.detach(),pred,a))
        self.history=self.history[-self.s.history:]
        residual=actual-pred
        self.feedback=residual if self.feedback is None else .7*self.feedback+.3*residual
        forecast=distance(z,q)-distance(pred,q)
        observed=distance(z,q)-distance(actual,q)
        self.stalls.append((z,q,(forecast-observed).clamp_min(0)*((forecast>0)&(observed<=0))))
        self.stalls=self.stalls[-8:];self.previous=plan;self.pending=None
        if self.trace:
            self.trace[-1].update(observed_prefix_error=float(residual.abs().mean()),
                                 observed_local_progress=float(observed[0]))

    def history_batch(self,z,n):
        s=self.s
        blank=z.new_zeros(1,s.history,s.patches,s.dim)
        out=dict(start=blank.clone(),next=blank.clone(),predicted=blank.clone(),
                 action=z.new_zeros(1,s.history,s.ad),
                 mask=torch.zeros(1,s.history,dtype=torch.bool,device=z.device))
        for slot,(start,nxt,pred,a) in enumerate(self.history,start=s.history-len(self.history)):
            out['start'][:,slot]=start;out['next'][:,slot]=nxt
            out['predicted'][:,slot]=pred;out['action'][:,slot]=a
            out['mask'][:,slot]=True
        return {k:v.expand(n,*v.shape[1:]) for k,v in out.items()}

    def stall_cost(self,z,targets):
        penalty=z.new_zeros(len(targets));counts=torch.zeros(len(targets),device=z.device)
        for i,(start,q,error) in enumerate(self.stalls):
            near=(distance(start,z)<=.005)&(distance(q,targets)<=.005)&(error>0)
            counts+=near;penalty+=near*error*(.8**(len(self.stalls)-1-i))
        return torch.where(counts>=3,penalty.clamp(max=.02),0.)

    @torch.no_grad()
    def plan(self,z,g,*,budget=None):
        s=self.s;m=self.model;method=s.method;before=self.world.calls
        if budget is not None:budget.check()
        if method=='vanilla_action_flow':
            a=m.propose(m.prepare(z,g,g),1,budget)[0,0]
            self.trace.append(dict(world_transitions=0,selected_round=0,local_cost=None))
            return a[0]
        if method in ('diffusion_policy_port','feedback_wm_port'):
            return self.diffusion_plan(z,g,budget)
        if method=='leflow':
            actions=m.sample(z,g,n=64,budget=budget,world=self.world)[0].clamp(-1,1)
            pred=self.world.rollout(z.expand(64,-1,-1),actions,budget=budget)
            scores=(pred[:,-1]-g).square().mean((1,2));j=int(scores.argmin())
            self.trace.append(dict(world_transitions=self.world.calls-before,selected_round=0,
                                   local_cost=float(scores[j])))
            return actions[j,0]
        n,per=s.targets,s.candidates
        if (n,per)!=(8,4):raise ValueError('Registered search requires eight targets and four proposals')
        zs=z.expand(n,-1,-1);gs=g.expand(n,-1,-1)
        if method=='sage_port':
            duration=z.new_tensor([5,5,10,10,20,20,40,60])
            q=m.subgoal(zs,gs,duration);routes=distance(q,gs)
            actor=m.actor;ctx=actor.prepare(zs,q,gs);recorded=None
        else:
            if self.bank is None:raise ValueError('Retrieval-based method requires a training bank')
            q,recorded,routes=self.bank.query(z,g,count=7)
            q=torch.cat((q,g));routes=torch.cat((routes,distance(z,g)))
            actor=m;ctx=m.prepare(zs,q,gs,self.history_batch(z,n))
        use_memory=method in ('execution_revision','retrieval_repair_ttt')
        stalls=self.stall_cost(z,q) if use_memory else z.new_zeros(n)
        rounds=2 if method=='retrieval_repair_ttt' else s.rounds
        a=actor.propose(ctx,per,budget)
        def insert(a):
            if recorded is not None:a[:7,-1]=recorded
            if use_memory and self.previous is not None:a[:,1,:-1]=self.previous[1:]
            return a
        a=insert(a)
        best=z.new_full((n,),float('inf'));plans=None;prefixes=None
        best_round=torch.zeros(n,device=z.device,dtype=torch.long)
        for r in range(rounds):
            if budget is not None:budget.check()
            a=a.clamp(-1,1)
            pred=self.world.rollout(z.expand(n*per,-1,-1),a.flatten(0,1),budget=budget)
            pred=pred.reshape(n,per,s.horizon,s.patches,s.dim)
            scores=m.score(ctx,a,pred,q) if use_memory else local_scores(patch_distance(pred,q[:,None,None]))
            costs=scores['cost'];v,j=costs.min(1);ix=torch.arange(n,device=z.device)
            if not torch.isfinite(costs).all():raise FloatingPointError('Nonfinite candidate score')
            if plans is None:
                plans=a[ix,j].clone();prefixes=pred[ix,j,0].clone()
            else:
                improved=v<best;plans[improved]=a[ix,j][improved];prefixes[improved]=pred[ix,j,0][improved]
            best_round[v<best]=r;best=torch.minimum(best,v)
            if r+1<rounds:
                if method=='retrieval_repair_ttt':
                    ctx=m.revise(ctx,a,scores);a=m.propose(ctx,per,budget)
                elif method=='action_flow_search':
                    a=insert(m.propose(ctx,per,budget))
                else:
                    ids=costs.topk(2,dim=1,largest=False).indices
                    elite=a[ix[:,None],ids];mean=elite.mean(1);std=elite.std(1,unbiased=False).clamp_min(.05)
                    a=(mean[:,None]+std[:,None]*torch.randn_like(a)).clamp(-1,1);a[:,0]=plans
        total=routes+best+stalls;j=int(total.argmin());plan=plans[j].detach()
        if use_memory:self.pending=(z.detach(),plan[0:1],prefixes[j:j+1].detach(),q[j:j+1].detach(),plan)
        self.trace.append(dict(world_transitions=self.world.calls-before,selected_round=int(best_round[j]),
            target_index=j,local_cost=float(best[j]),route_cost=float(routes[j]),
            history_steps=len(self.history),prefix_prediction=prefixes[j,0].cpu().tolist()))
        if budget is not None:budget.check()
        return plan[0]

    @torch.no_grad()
    def diffusion_plan(self,z,g,budget):
        before=self.world.calls;weights=None
        def guide(clean,step):
            nonlocal weights
            if step<12:return clean
            if budget is not None:budget.check()
            if weights is None:
                probes=(clean[0,0][None]+.1*torch.randn(4,self.s.ad,device=z.device)).clamp(-1,1)
                predictions=self.world(z.expand(4,-1,-1),probes)
                var=predictions.var(0,unbiased=False);mean=var.mean()
                weights=torch.ones_like(var) if mean<1.e-12 else (var/(mean+1.e-12)+1.e-6).sqrt()
                weights=(weights/(weights.mean()+1.e-12)).clamp(max=10).detach()
            with torch.enable_grad():
                plan=clean.detach().requires_grad_(True)
                prediction=self.world(z,plan[:,0].clamp(-1,1))
                corrected=prediction if self.feedback is None else prediction+.5*self.feedback
                energy=(weights[None]*(corrected-g).square()).mean()
                gradient=torch.autograd.grad(energy,plan)[0].detach()
            rms=gradient.square().mean().sqrt().clamp_min(1.e-8)
            return (clean-.05*gradient*(.25/rms).clamp(max=1)).clamp(-1,1)
        feedback=self.s.method=='feedback_wm_port'
        plan=self.model.sample(z,g,budget=budget,guidance=guide if feedback else None)[0]
        if feedback:
            if budget is not None:budget.check()
            pred=self.world(z,plan[0:1]);self.pending=(z.detach(),plan[0:1],pred.detach(),g.detach(),plan.detach())
        self.trace.append(dict(world_transitions=self.world.calls-before,selected_round=0,
                               feedback_updates=len(self.history),local_cost=None))
        return plan[0]
