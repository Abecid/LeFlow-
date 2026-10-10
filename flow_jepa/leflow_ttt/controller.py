"""Native LeFlow schedule with causal, episode-local outcome adaptation."""
import torch
from ..models import distance


class AdaptiveController:
    def __init__(self,model,world,c):
        self.model,self.world,self.c=model,world,c
        self.execution_blocks=c['baseline']['leflow']['receding_horizon']
        assert self.execution_blocks==model.horizon==5
        self.begin_episode()

    def begin_episode(self):
        self.history=[]; self.pending=None; self.decisions=[]

    @torch.no_grad()
    def observe(self,actual):
        if self.pending is None:return
        previous=self.pending
        self.history.append(dict(start=previous['start'],actions=previous['actions'],
            predicted=previous['predicted'],next=actual.detach().clone()))
        self.history=self.history[-self.c['leflow_ttt']['history_chunks']:]
        d=self.decisions[-1]
        d['actual_goal_cost']=float((actual-previous['goal']).square().mean())
        d['actual_progress']=d['start_goal_cost']-d['actual_goal_cost']
        d['raw_outcome_mse']=float((previous['predicted']-actual).square().mean())
        d['adapted_outcome_mse']=float((previous['corrected']-actual).square().mean())
        d['prior_outcome_mse']=float((previous['prior']-actual).square().mean())
        self.pending=None

    def batch_history(self,start):
        count=self.c['leflow_ttt']['history_chunks']
        states={k:start.new_zeros(1,count,*start.shape[1:]) for k in ('start','predicted','next')}
        actions=start.new_zeros(1,count,5,8); mask=torch.zeros(1,count,dtype=torch.bool,device=start.device)
        for j,item in enumerate(self.history,count-len(self.history)):
            for k in states:states[k][:,j]=item[k]
            actions[:,j]=item['actions'];mask[:,j]=True
        return dict(**states,actions=actions,mask=mask)

    @torch.no_grad()
    def plan(self,start,goal,*,budget=None):
        cfg=self.c['baseline']['leflow']; initial_calls=self.world.calls
        history=self.batch_history(start)
        weights,diag=self.model.memory.fit(history,self.model.inner_steps)
        if budget is not None:budget.check()
        paths,actions=self.model.sample_adapted(start,goal,weights,cfg['candidates'],cfg['flow_steps'],budget)
        starts=start.expand(len(paths),-1,-1)
        predicted=self.world.rollout(starts,actions,budget=budget)[:,-1]
        correction=self.model.memory.correction(weights,starts,actions,predicted)
        corrected=predicted+correction
        raw_costs=(predicted-goal).square().mean((1,2))
        costs=(corrected-goal).square().mean((1,2)); selected=int(costs.argmin())
        # The evaluator clips actions at the simulator boundary. Fit only the
        # actual executed action sequence, never raw out-of-bounds commands.
        executed=actions[selected:selected+1].clamp(-1,1)
        execution_prediction=self.world.rollout(start,executed,budget=budget)[:,-1]
        exec_correction=self.model.memory.correction(weights,start,executed,execution_prediction)
        prior=self.model.memory.correction(self.model.memory.prior(1),start,executed,execution_prediction)
        self.pending=dict(start=start.detach().clone(),actions=executed.detach().clone(),
            predicted=execution_prediction.detach().clone(),corrected=(execution_prediction+exec_correction).detach().clone(),
            prior=(execution_prediction+prior).detach().clone(),goal=goal.detach().clone())
        self.decisions.append(dict(history_chunks=len(self.history),inner_steps=self.model.inner_steps,
            support_loss_before=float(diag['support_before']),support_loss_after=float(diag['support_after']),
            fast_weight_delta=float(diag['fast_delta']),
            selected_candidate=selected,raw_selected_candidate=int(raw_costs.argmin()),
            adaptation_changed_selection=selected!=int(raw_costs.argmin()),
            raw_goal_cost=float(raw_costs[selected]),adapted_goal_cost=float(costs[selected]),
            start_goal_cost=float((start-goal).square().mean()),
            predicted_progress=float((start-goal).square().mean()-raw_costs[selected]),
            corrected_progress=float((start-goal).square().mean()-costs[selected]),
            mean_correction_norm=float(correction.square().mean().sqrt()),
            action_outside_bounds=float((actions.abs()>1).float().mean()),
            selected_action_clipping_fraction=float((actions[selected].abs()>1).float().mean()),
            latent_path_diversity=float(paths[:,1:-1].std(0,unbiased=False).mean()),
            action_diversity=float(actions.std(0,unbiased=False).mean()),
            world_transitions=self.world.calls-initial_calls,
            candidates=cfg['candidates'],flow_steps=cfg['flow_steps'],execution_blocks=self.execution_blocks,
            adaptation_scope='current_episode_observed_macro_transitions'))
        return actions[selected].flatten(),goal[0],float(costs[selected]),self.model.horizon

    def diagnostics(self):
        return dict(decisions=self.decisions,adaptation='nonlinear_fast_weights',
                    physical_states_and_memory_separate=True)
