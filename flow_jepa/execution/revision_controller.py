"""Bounded controller with factual-error-trained latent cost corrections."""
import time

import torch

from ..models import distance
from .controller import ExecutionController, local_cost


class RevisionController(ExecutionController):
    def begin_episode(self):
        super().begin_episode()
        self.history=[]

    @torch.no_grad()
    def observe(self, actual):
        if self.pending is not None:
            predicted,start,target=self.pending
            # Store only the observation reached AFTER the preceding action.
            self.history.append((start.clone(),actual.detach().clone(),predicted.clone()))
            self.history=self.history[-self.model.history_steps:]
            observed=distance(start,target)-distance(actual,target)
            last=self.trace[-1]
            last['corrected_prefix_progress_error']=float(last['corrected_prefix_target_progress']-observed)
        super().observe(actual)

    def model_history(self, start, count):
        history=self.model.empty_history(start)
        for i,(past,actual,predicted) in enumerate(self.history,start=self.model.history_steps-len(self.history)):
            history['start'][:,i]=past
            history['next'][:,i]=actual
            history['predicted'][:,i]=predicted
            history['mask'][:,i]=True
        return {k:v.expand(count,*v.shape[1:]) for k,v in history.items()}

    @torch.no_grad()
    def plan(self,start,goal,*,budget=None):
        def check():
            if budget is not None:budget.check()
        check(); then=time.perf_counter()
        targets,recorded,route_cost,info=self.bank.query(start,goal,count=7)
        targets=torch.cat((targets,goal));route_cost=torch.cat((route_cost,distance(start,goal)))
        info.append(dict(episode=None,start=None,span=None,direct_goal=True))
        retrieval_ms=1000*(time.perf_counter()-then);check()
        n,per=8,4
        context=self.model.prepare(start.expand(n,-1,-1),targets,goal.expand(n,-1,-1),self.model_history(start,n))
        actions=self.model.propose_context(context,per)
        actions[:7,-1]=recorded;actions[7,-1]=actions[7,0]
        initial_actions=actions.clone()
        best_cost=start.new_full((n,),float('inf'))
        raw_min=best_cost.clone()
        best_actions=best_prefix=best_raw=best_corrections=None
        best_round=torch.zeros(n,device=start.device,dtype=torch.long)
        thoughts=context['thought'][:,None].expand(n,per,-1).flatten(0,1)
        queries=targets[:,None].expand(n,per,*targets.shape[1:]).flatten(0,1)
        for iteration in range(3):
            check(); flat=actions.reshape(n*per,5,8).clamp(-1,1)
            predicted=self.world.rollout(start.expand(n*per,-1,-1),flat,budget=budget)
            raw=local_cost(predicted,queries).reshape(n,per)
            correction=self.model.corrections(thoughts,flat,predicted,queries).reshape(n,per,2)
            # Offline corrections are uncertain on generated actions. Penalize
            # predicted optimism; never reward a lower learned counterfactual
            # cost than the uncorrected frozen world supplies.
            costs=raw+correction[:,:,1].clamp_min(0)
            if not torch.isfinite(costs).all():raise FloatingPointError('Nonfinite revised score')
            raw_min=torch.minimum(raw_min,raw.min(1).values)
            values,ids=costs.topk(2,dim=1,largest=False);ix=torch.arange(n,device=start.device)
            proposed=actions[ix,ids[:,0]]
            prefix=predicted.reshape(n,per,5,*start.shape[1:])[ix,ids[:,0],0]
            selected_raw=raw[ix,ids[:,0]];selected_correction=correction[ix,ids[:,0]]
            improved=values[:,0]<best_cost
            if best_actions is None:
                best_actions,best_prefix=proposed.clone(),prefix.clone()
                best_raw,best_corrections=selected_raw.clone(),selected_correction.clone()
            else:
                best_actions[improved]=proposed[improved];best_prefix[improved]=prefix[improved]
                best_raw[improved]=selected_raw[improved];best_corrections[improved]=selected_correction[improved]
            best_round[improved]=iteration;best_cost=torch.minimum(best_cost,values[:,0])
            if iteration<2:
                elites=actions[ix[:,None],ids]
                mean,std=elites.mean(1),elites.std(1,unbiased=False).clamp_min(.05)
                actions=(mean[:,None]+std[:,None]*torch.randn_like(actions)).clamp(-1,1)
                actions[:,0]=best_actions
        scores=route_cost+best_cost;chosen=int(scores.argmin())
        target=targets[chosen:chosen+1];prefix=best_prefix[chosen:chosen+1]
        progress=distance(start,target)-distance(prefix,target)
        correction=best_corrections[chosen]
        self.trace.append(dict(anchor=info[chosen],selected_anchor=chosen,
            selected_after_refinement_round=int(best_round[chosen]),
            response_changed_retrieval_choice=chosen!=int(route_cost.argmin()),
            correction_changed_raw_pool_anchor=chosen!=int((route_cost+raw_min).argmin()),
            route_cost=float(route_cost[chosen]),local_cost=float(best_cost[chosen]),
            raw_local_cost=float(best_raw[chosen]),
            candidate_route_costs=route_cost.cpu().tolist(),candidate_response_costs=best_cost.cpu().tolist(),
            candidate_raw_response_costs=best_raw.cpu().tolist(),
            candidate_cost_corrections=best_corrections.cpu().tolist(),
            predicted_prefix_target_progress=float(progress),
            corrected_prefix_target_progress=float(distance(start,target)-
                (distance(prefix,target)+correction[0]).clamp(0,2)),
            selected_prefix_cost_correction=float(correction[0]),
            selected_terminal_cost_correction=float(correction[1]),
            applied_terminal_penalty=float(correction[1].clamp_min(0)),
            history_valid_steps=len(self.history),reasoning_depth=context['depth'],
            proposal_diversity=float(initial_actions.flatten(2).std(1,unbiased=False).mean()),
            retrieval_cpu_wall_ms=retrieval_ms,refinement_batches=3,candidate_world_transitions=480))
        self.pending=(prefix.detach(),start.detach(),target.detach())
        return best_actions[chosen,0],target[0],float(scores[chosen]),5
