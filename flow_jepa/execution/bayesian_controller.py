"""Posterior-relative repair acceptance with a fixed causal scoring context."""
import time
import torch

from ..models import distance
from .retrieval_repair import RetrievalRepairController


class BayesianRepairController(RetrievalRepairController):
    @torch.no_grad()
    def observe(self, actual):
        if self.pending is not None:
            predicted, _, target = self.pending
            error = float(distance(actual,target)-distance(predicted,target))
            trace = self.trace[-1]
            trace['observed_prefix_cost_error'] = error
            trace['applied_prefix_prediction_error'] = error-trace['selected_applied_prefix_correction']
            variance = trace['selected_prefix_predictive_variance']
            trace['posterior_standardized_error'] = (error-trace['selected_unbounded_prefix_mean'])**2/max(variance,1e-12)
        super().observe(actual)

    @torch.no_grad()
    def plan(self, start, goal, *, budget=None):
        def check():
            if budget is not None: budget.check()
        check(); began = time.perf_counter()
        targets, recorded, route_cost, info = self.bank.query(start,goal,count=7)
        targets = torch.cat((targets,goal)); route_cost = torch.cat((route_cost,distance(start,goal)))
        info.append(dict(episode=None,start=None,span=None,direct_goal=True))
        retrieval_ms = 1000*(time.perf_counter()-began); check()
        n, per = 8, self.model.candidates
        if per != 4: raise ValueError('Registered comparison uses four candidates per target')
        rounds = int(self.c['flow_reasoning']['rounds'])
        if not 1 <= rounds <= self.model.rounds: raise ValueError('Untrained inference depth')
        context = self.model.prepare(start.expand(n,-1,-1),targets,goal.expand(n,-1,-1),self.model_history(start,n))
        noise = torch.randn(n,per,5,8,device=start.device)
        actions = self.model.propose_context(context,per,noise=noise,budget=budget)
        actions[:7,-1] = recorded
        if self.previous_plan is not None: actions[:,1,:-1] = self.previous_plan[1:]
        context = self.initial_pool_context(context,actions,recorded)
        stalls, stall_counts = self.stall_penalties(start,targets)
        best_states = best_actions = None
        best_round = torch.zeros(n,device=start.device,dtype=torch.long)
        ix = torch.arange(n,device=start.device)
        logs, round_scores, diversities, changes = [], [], [], []
        previous_actions = actions.clone(); raw_min = start.new_full((n,),float('inf'))
        changed = batches = transitions = 0
        initial_diversity = float(actions.flatten(2).std(1,unbiased=False).mean())
        first_anchors = first_sources = None
        for iteration in range(rounds):
            check()
            if iteration == 0:
                states = self.world.rollout(start.expand(n*per,-1,-1),actions.reshape(n*per,5,8),budget=budget)
                states = states.reshape(n,per,5,*start.shape[1:]); transitions += n*per*5
            else:
                # Incumbent was already predicted under this exact frozen world.
                # Reuse its entire rollout; only the three new repairs need work.
                states = self.world.rollout(start.expand(n*(per-1),-1,-1),actions[:,1:].reshape(n*(per-1),5,8),budget=budget)
                states = torch.cat((best_states[:,None],states.reshape(n,per-1,5,*start.shape[1:])),1)
                transitions += n*(per-1)*5
            scores = self.model.score_candidates(context,actions,states,targets)
            if not torch.isfinite(scores['cost']).all(): raise FloatingPointError('Nonfinite Bayesian cost')
            selected, comparison = self.model.select(context,scores)
            if iteration == 0:
                first_anchors = actions[ix,selected].clone()
                first_sources = context['pool_provenance'][ix,selected].clone()
            else:
                best_round[selected!=0] = iteration
            best_actions, best_states = actions[ix,selected].clone(), states[ix,selected].clone()
            best = {k:v[ix,selected] for k,v in scores.items()}
            changed += int((scores['cost'].argmin(1)!=scores['raw'].argmin(1)).sum()); batches += n
            raw_min = torch.minimum(raw_min,scores['raw'].min(1).values)
            round_scores.append(float((route_cost+best['cost']+stalls).min()))
            diversities.append(float(actions.flatten(2).std(1,unbiased=False).mean()))
            changes.append(float((actions-previous_actions).abs().mean())); previous_actions = actions.clone()
            record = dict(round=iteration, selected=selected.cpu().tolist(),
                actions=actions.cpu().tolist(), raw_costs=scores['raw'].cpu().tolist(),
                costs=scores['cost'].cpu().tolist(), prefix_costs=scores['prefix'].cpu().tolist(),
                terminal_costs=scores['terminal'].cpu().tolist(),
                signed_corrections=scores['correction'].cpu().tolist(),
                applied_prefix=scores['prefix_penalty'].cpu().tolist(),
                predictive_variance=(scores['prefix_predictive_variance']*self.model.error_scale**2).cpu().tolist(),
                epistemic_variance=(scores['prefix_epistemic_variance']*self.model.error_scale**2).cpu().tolist())
            if comparison is not None:
                record.update({k:v.cpu().tolist() for k,v in comparison.items()})
            logs.append(record)
            if iteration < rounds-1:
                context = self.model.revise(context,actions,scores)
                context = {k:v for k,v in context.items() if k!='pool_provenance'}
                actions = self.model.propose_context(context,per,noise=noise,budget=budget)
        costs = route_cost+best['cost']+stalls; chosen = int(costs.argmin())
        target = targets[chosen:chosen+1]; prefix = best_states[chosen:chosen+1,0]
        correction = best['correction'][chosen]
        applied_prefix = best['prefix_penalty'][chosen]
        difference = (best_actions-first_anchors).abs().amax((1,2))
        prior = self.model.prefix_maps(context,best_actions[:,None],best_states[:,None,0],targets,prior=True)
        self.trace.append(dict(anchor=info[chosen],selected_anchor=chosen,
            selected_after_refinement_round=int(best_round[chosen]),
            response_changed_retrieval_choice=chosen!=int(route_cost.argmin()),
            correction_changed_raw_pool_anchor=chosen!=int((route_cost+raw_min+stalls).argmin()),
            stall_changed_pool_anchor=chosen!=int((route_cost+best['cost']).argmin()),
            correction_changed_action_batches=changed,evaluated_action_batches=batches,
            route_cost=float(route_cost[chosen]),local_cost=float(best['cost'][chosen]),
            raw_local_cost=float(best['raw'][chosen]),raw_terminal_cost=float(best['terminal'][chosen]),
            raw_prefix_cost=float(best['prefix'][chosen]),candidate_route_costs=route_cost.cpu().tolist(),
            candidate_response_costs=best['cost'].cpu().tolist(),candidate_raw_response_costs=best['raw'].cpu().tolist(),
            candidate_cost_corrections=best['correction'].cpu().tolist(),candidate_stall_penalties=stalls.cpu().tolist(),
            candidate_stall_counts=stall_counts.cpu().tolist(),applied_stall_penalty=float(stalls[chosen]),
            predicted_prefix_target_progress=float(distance(start,target)-distance(prefix,target)),
            corrected_prefix_target_progress=float(distance(start,target)-distance(prefix,target)-applied_prefix),
            selected_prefix_cost_correction=float(correction[0]),selected_terminal_cost_correction=float(correction[1]),
            selected_applied_prefix_correction=float(applied_prefix),
            applied_prefix_penalty=float(self.model.prefix_weight*applied_prefix),
            applied_terminal_penalty=float((1-self.model.prefix_weight)*correction[1].clamp_min(0)),
            selected_unbounded_prefix_mean=float(best['prefix_mean'][chosen]*self.model.error_scale),
            selected_prefix_predictive_variance=float(best['prefix_predictive_variance'][chosen]*self.model.error_scale**2),
            prior_prefix_cost_correction=float(prior[chosen].mean()),
            warm_start_available=self.previous_plan is not None,
            selected_plan_matches_shifted_prefix=bool(self.previous_plan is not None and torch.equal(best_actions[chosen,:4],self.previous_plan[1:])),
            history_valid_steps=len(self.history),fast_support_steps=int(context['fast_support_steps'][chosen]),
            fast_weight_delta_norm=float(context['fast_delta'][chosen].norm()),fast_weight_rank=self.model.rank,
            fast_fit_batches_per_decision=1,fast_weights_fixed_during_search=True,
            adaptation_scope='current_episode_executed_transitions',posterior_observation_unit='one_executed_transition',
            reasoning_depth=rounds,proposal_diversity=initial_diversity,round_best_scores=round_scores,
            round_action_changes=changes,round_proposal_diversity=diversities,
            flow_solver_steps=self.model.flow_steps,flow_evaluations=self.model.flow_steps+(rounds-1)*self.model.repair_steps,
            scoring_context_fixed=True,retrieval_cpu_wall_ms=retrieval_ms,refinement_batches=rounds,
            candidate_world_transitions=transitions,cached_world_transitions=(rounds-1)*n*5,
            repair_radius=self.model.repair_radius,repair_solver_steps=self.model.repair_steps,
            selected_distance_from_incumbent_linf=float(difference[chosen]),selected_is_unchanged_incumbent=bool(difference[chosen]==0),
            surrogate_gradient_evaluations=0,repair_incumbent_preserved=True,
            repair_anchor_source=('generated','retrieved','shifted')[int(first_sources[chosen])],
            acceptance_beta=self.model.acceptance_beta,posterior_comparisons=logs,
            uncertainty_scope='posterior_mean_prefix_cost_error_only; excludes task semantics and terminal-model bias'))
        self.proposed_plan = best_actions[chosen].detach().clone()
        self.pending = (prefix.detach(),start.detach(),target.detach())
        return best_actions[chosen,0],target[0],float(costs[chosen]),5
