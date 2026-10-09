"""Executed-prefix scoring, causal stall feedback and full local-window coverage.

This is a separately registered candidate. Historical controller behavior and
saved checkpoints retain their original classes/configuration.
"""
import time

import numpy as np
import torch

from ..models import distance
from .controller import local_cost
from .revision import RevisionSegments
from .revision_controller import RevisionController


class AlignedSegments(RevisionSegments):
    def sample_spec(self, index):
        # Keep task/episode draws; repair only local start and feasible goal draws.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        rows = self.tasks[self.names[int(rng.integers(len(self.names)))]]
        row = rows[int(rng.integers(len(rows)))]
        limit = row['steps'] - 5
        if row.get('first_success_action') is not None:
            limit = min(limit, row['first_success_action'] // self.c['data']['action_repeat'])
        start = int(rng.integers(limit + 1))
        offsets = [v for v in self.c['controller_grounded']['goal_offsets'] if v <= row['steps'] - start]
        grng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 1701]))
        return row, start, int(grng.choice(offsets))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        _, start, delta = self.sample_spec(index)
        item.update(sample_start=torch.tensor(start), sample_goal_offset=torch.tensor(delta))
        return item


class AlignedController(RevisionController):
    def begin_episode(self):
        super().begin_episode()
        self.previous_plan = self.proposed_plan = None
        self.stall_history = []

    @torch.no_grad()
    def observe(self, actual):
        if self.pending is not None:
            predicted, start, target = self.pending
            predicted_progress = distance(start, target) - distance(predicted, target)
            actual_progress = distance(start, target) - distance(actual, target)
            # Failed execution is evidence only AFTER its observation arrives.
            optimism = (predicted_progress - actual_progress).clamp_min(0)
            failed = (predicted_progress > 0) & (actual_progress <= 0)
            self.stall_history.append((start.clone(), target.clone(), optimism * failed))
            self.stall_history = self.stall_history[-self.c['execution_revision']['stall_memory']:]
            self.previous_plan = self.proposed_plan
        super().observe(actual)

    def stall_penalties(self, start, targets):
        cfg = self.c['execution_revision']
        if not self.stall_history:
            return start.new_zeros(len(targets)), torch.zeros(len(targets), device=start.device, dtype=torch.long)
        past = torch.cat([v[0] for v in self.stall_history])
        queries = torch.cat([v[1] for v in self.stall_history])
        errors = torch.cat([v[2] for v in self.stall_history])
        nearby = (distance(past, start) <= cfg['stall_state_radius'])[:, None]
        nearby = nearby & (distance(queries[:, None], targets[None]) <= cfg['stall_target_radius'])
        nearby = nearby & (errors[:, None] > 0)
        counts = nearby.sum(0)
        ages = torch.arange(len(errors)-1, -1, -1, device=start.device)
        weights = cfg['stall_decay'] ** ages
        # Cost-unit, recency-weighted observed optimism; no arbitrary task bonus.
        penalty = (nearby * (errors * weights)[:, None]).sum(0)
        penalty = penalty.clamp(max=cfg['stall_penalty_cap'])
        return torch.where(counts >= cfg['stall_min_failures'], penalty, 0.), counts

    def initialize_actions(self, context, recorded):
        actions = self.model.propose_context(context, 4)
        actions[:7, -1] = recorded
        if self.previous_plan is not None:
            # Reuse a stochastic slot for every target, retaining mean/expert and
            # fresh alternatives. A fresh proposed tail avoids a padded stop.
            actions[:, 1, :-1] = self.previous_plan[1:]
        # The direct goal's final slot stays fresh instead of duplicating its mean.
        return actions

    def candidate_costs(self, predicted, targets, correction):
        cfg = self.c['execution_revision']
        prefix = distance(predicted[:, 0], targets)
        terminal = local_cost(predicted, targets)
        weight = cfg['prefix_weight']
        raw = weight * prefix + (1-weight) * terminal
        # Conservative on generated actions: positive learned errors penalize
        # optimism; negative errors cannot create unsupported rewards.
        penalty = weight * correction[:, 0].clamp_min(0) + (1-weight) * correction[:, 1].clamp_min(0)
        return raw + penalty, raw, prefix, terminal

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
        actions = self.initialize_actions(context, recorded)
        initial_actions = actions.clone()
        stalls, stall_counts = self.stall_penalties(start, targets)
        best_cost = start.new_full((n,), float('inf'))
        raw_min = best_cost.clone()
        best_actions = best_prefix = best_raw = best_corrections = best_terminal = best_prefix_cost = None
        best_round = torch.zeros(n, device=start.device, dtype=torch.long)
        thoughts = context['thought'][:,None].expand(n,per,-1).flatten(0,1)
        queries = targets[:,None].expand(n,per,*targets.shape[1:]).flatten(0,1)
        changed_candidates = evaluated_batches = 0
        for iteration in range(3):
            check(); flat = actions.reshape(n*per,5,8).clamp(-1,1)
            predicted = self.world.rollout(start.expand(n*per,-1,-1), flat, budget=budget)
            correction = self.model.corrections(thoughts, flat, predicted, queries)
            costs, raw, prefix_cost, terminal = self.candidate_costs(predicted, queries, correction)
            costs, raw, prefix_cost, terminal = [v.reshape(n,per) for v in (costs,raw,prefix_cost,terminal)]
            correction = correction.reshape(n,per,2)
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
            if iteration < 2:
                elites = actions[ix[:,None], ids]
                mean, std = elites.mean(1), elites.std(1,unbiased=False).clamp_min(.05)
                actions = (mean[:,None] + std[:,None] * torch.randn_like(actions)).clamp(-1,1)
                actions[:,0] = best_actions
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
            retrieval_cpu_wall_ms=retrieval_ms, refinement_batches=3, candidate_world_transitions=480))
        self.proposed_plan = best_actions[chosen].detach().clone()
        self.pending = (prefix.detach(),start.detach(),target.detach())
        return best_actions[chosen,0], target[0], float(scores[chosen]), 5
