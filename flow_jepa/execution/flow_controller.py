"""Bounded flow proposal / predicted consequence / latent revision controller."""
import time

import torch

from ..models import distance
from .aligned import AlignedController


class FlowReasoningController(AlignedController):
    def plan_diagnostics(self, context, start, targets, actions, prefixes, chosen):
        return {}

    @torch.no_grad()
    def plan(self, start, goal, *, budget=None):
        def check():
            if budget is not None:
                budget.check()
        check(); then = time.perf_counter()
        targets, recorded, route_cost, info = self.bank.query(start, goal, count=7)
        targets = torch.cat((targets, goal))
        route_cost = torch.cat((route_cost, distance(start, goal)))
        info.append(dict(episode=None, start=None, span=None, direct_goal=True))
        retrieval_ms = 1000 * (time.perf_counter()-then); check()
        n, per = 8, 4
        context = self.model.prepare(start.expand(n,-1,-1), targets, goal.expand(n,-1,-1), self.model_history(start,n))
        rounds = self.c['flow_reasoning']['rounds']
        noise = torch.randn(n, per, 5, 8, device=start.device)
        actions = self.model.propose_context(context, per, noise=noise, budget=budget)
        actions[:7, -1] = recorded
        if self.previous_plan is not None:
            actions[:, 1, :-1] = self.previous_plan[1:]
        initial_actions = actions.clone()
        stalls, stall_counts = self.stall_penalties(start, targets)
        best_cost = start.new_full((n,), float('inf'))
        raw_min = best_cost.clone()
        best_actions = best_prefix = best_raw = best_corrections = best_terminal = best_prefix_cost = None
        best_round = torch.zeros(n, device=start.device, dtype=torch.long)
        round_best_scores, round_action_changes, round_diversity = [], [], []
        previous_round_actions = actions.clone()
        changed_candidates = evaluated_batches = 0
        for iteration in range(rounds):
            check(); flat = actions.reshape(n*per,5,8).clamp(-1,1)
            predicted = self.world.rollout(start.expand(n*per,-1,-1), flat, budget=budget)
            candidate_states = predicted.reshape(n, per, 5, *start.shape[1:])
            evidence_scores = self.model.score_candidates(context, actions, candidate_states, targets)
            costs, raw, prefix_cost, terminal, correction = [evidence_scores[k]
                for k in ('cost', 'raw', 'prefix', 'terminal', 'correction')]
            round_diversity.append(float(actions.flatten(2).std(1, unbiased=False).mean()))
            round_action_changes.append(float((actions-previous_round_actions).abs().mean()))
            previous_round_actions = actions.clone()
            if not torch.isfinite(costs).all():
                raise FloatingPointError('Nonfinite execution-aligned score')
            changed_candidates += int((costs.argmin(1) != raw.argmin(1)).sum())
            evaluated_batches += n
            raw_min = torch.minimum(raw_min, raw.min(1).values)
            values, ids = costs.topk(2, dim=1, largest=False)
            ix = torch.arange(n, device=start.device)
            chosen = ids[:,0]
            prefix = predicted.reshape(n,per,5,*start.shape[1:])[ix,chosen,0]
            values_to_save = [actions[ix,chosen], prefix, raw[ix,chosen], correction[ix,chosen],
                              terminal[ix,chosen], prefix_cost[ix,chosen]]
            improved = values[:,0] < best_cost
            if best_actions is None:
                saved = [v.clone() for v in values_to_save]
            else:
                saved = [best_actions,best_prefix,best_raw,best_corrections,best_terminal,best_prefix_cost]
                for existing, value in zip(saved, values_to_save):
                    existing[improved] = value[improved]
            best_actions,best_prefix,best_raw,best_corrections,best_terminal,best_prefix_cost = saved
            best_round[improved] = iteration
            best_cost = torch.minimum(best_cost, values[:,0])
            round_best_scores.append(float((route_cost+best_cost+stalls).min()))
            if iteration < rounds-1:
                context = self.model.revise(context, actions, evidence_scores)
                # Common noise across rounds makes revised generations depend
                # on new reasoning evidence. Incumbents are retained separately.
                actions = self.model.propose_context(context, per, noise=noise, budget=budget)
        scores = route_cost + best_cost + stalls
        chosen = int(scores.argmin())
        target, prefix = targets[chosen:chosen+1], best_prefix[chosen:chosen+1]
        correction = best_corrections[chosen]
        self.trace.append(dict(anchor=info[chosen], selected_anchor=chosen,
            selected_after_refinement_round=int(best_round[chosen]),
            response_changed_retrieval_choice=chosen!=int(route_cost.argmin()),
            correction_changed_raw_pool_anchor=int((route_cost+best_cost+stalls).argmin())!=int((route_cost+raw_min+stalls).argmin()),
            stall_changed_pool_anchor=chosen!=int((route_cost+best_cost).argmin()),
            correction_changed_action_batches=changed_candidates, evaluated_action_batches=evaluated_batches,
            route_cost=float(route_cost[chosen]), local_cost=float(best_cost[chosen]),
            raw_local_cost=float(best_raw[chosen]), raw_terminal_cost=float(best_terminal[chosen]),
            raw_prefix_cost=float(best_prefix_cost[chosen]),
            candidate_route_costs=route_cost.cpu().tolist(), candidate_response_costs=best_cost.cpu().tolist(),
            candidate_raw_response_costs=best_raw.cpu().tolist(), candidate_cost_corrections=best_corrections.cpu().tolist(),
            candidate_stall_penalties=stalls.cpu().tolist(), candidate_stall_counts=stall_counts.cpu().tolist(),
            applied_stall_penalty=float(stalls[chosen]),
            predicted_prefix_target_progress=float(distance(start,target)-distance(prefix,target)),
            corrected_prefix_target_progress=float(distance(start,target)-(distance(prefix,target)+correction[0]).clamp(0,2)),
            selected_prefix_cost_correction=float(correction[0]), selected_terminal_cost_correction=float(correction[1]),
            applied_prefix_penalty=float(self.c['execution_revision']['prefix_weight']*correction[0].clamp_min(0)),
            applied_terminal_penalty=float((1-self.c['execution_revision']['prefix_weight'])*correction[1].clamp_min(0)),
            warm_start_available=self.previous_plan is not None,
            selected_plan_matches_shifted_prefix=bool(self.previous_plan is not None and torch.equal(best_actions[chosen,:4],self.previous_plan[1:])),
            history_valid_steps=len(self.history), reasoning_depth=context['depth'],
            proposal_diversity=float(initial_actions.flatten(2).std(1,unbiased=False).mean()),
            round_best_scores=round_best_scores, round_action_changes=round_action_changes,
            round_proposal_diversity=round_diversity, flow_solver_steps=self.model.flow_steps,
            flow_evaluations=rounds*self.model.flow_steps, scoring_context_fixed=True,
            retrieval_cpu_wall_ms=retrieval_ms, refinement_batches=rounds,
            candidate_world_transitions=n*per*rounds*5,
            **self.plan_diagnostics(context, start, targets, best_actions, best_prefix, chosen)))
        self.proposed_plan = best_actions[chosen].detach().clone()
        self.pending = (prefix.detach(),start.detach(),target.detach())
        return best_actions[chosen,0], target[0], float(scores[chosen]), 5
