"""Select a subgoal jointly with the bounded controller's actual response."""
import time

import torch

from ..models import distance


def local_cost(predicted, targets):
    return 0.5 * (distance(predicted[:, -2], targets) + distance(predicted[:, -1], targets))


class ExecutionController:
    def __init__(self, model, world, c, bank):
        self.model, self.world, self.c, self.bank = model, world, c, bank
        self.begin_episode()

    def begin_episode(self):
        self.trace, self.pending = [], None

    @torch.no_grad()
    def observe(self, actual):
        if self.pending is not None:
            predicted, start, target = self.pending
            last = self.trace[-1]
            last['actual_prefix_prediction_error'] = float(distance(predicted, actual))
            last['actual_prefix_target_progress'] = float(distance(start, target) - distance(actual, target))
            self.pending = None

    def diagnostics(self):
        return dict(decisions=self.trace, full_chunk_executability_observed=False)

    @torch.no_grad()
    def plan(self, start, goal, *, budget=None):
        def check():
            if budget is not None:
                budget.check()
        check()
        then = time.perf_counter()
        targets, recorded, route_cost, info = self.bank.query(start, goal, count=7)
        # One direct-goal option provides a route-free local-control candidate.
        targets = torch.cat((targets, goal))
        route_cost = torch.cat((route_cost, distance(start, goal)))
        info.append(dict(episode=None, start=None, span=None, direct_goal=True))
        retrieval_ms = 1000 * (time.perf_counter() - then)
        check()
        n, per = 8, 4
        starts, goals = start.expand(n, -1, -1), goal.expand(n, -1, -1)
        actions = self.model.propose(starts, targets, goals, per)
        actions[:7, -1] = recorded
        actions[7, -1] = actions[7, 0]
        initial_actions = actions.clone()
        best_cost = start.new_full((n,), float('inf'))
        best_actions, best_prefix = None, None
        best_round = torch.zeros(n, device=start.device, dtype=torch.long)
        for iteration in range(3):
            check()
            flat = actions.reshape(n * per, 5, 8).clamp(-1, 1)
            predicted = self.world.rollout(start.expand(n * per, -1, -1), flat, budget=budget)
            costs = local_cost(predicted, targets[:, None].expand(n, per, *targets.shape[1:]).flatten(0, 1)).reshape(n, per)
            if not torch.isfinite(costs).all():
                raise FloatingPointError('Nonfinite controller candidate score')
            values, ids = costs.topk(2, dim=1, largest=False)
            ix = torch.arange(n, device=start.device)
            proposed = actions[ix, ids[:, 0]]
            prefix = predicted.reshape(n, per, 5, *start.shape[1:])[ix, ids[:, 0], 0]
            improved = values[:, 0] < best_cost
            if best_actions is None:
                best_actions, best_prefix = proposed.clone(), prefix.clone()
            else:
                best_actions[improved] = proposed[improved]
                best_prefix[improved] = prefix[improved]
            best_round[improved] = iteration
            best_cost = torch.minimum(best_cost, values[:, 0])
            if iteration < 2:
                elites = actions[ix[:, None], ids]
                mean, std = elites.mean(1), elites.std(1, unbiased=False).clamp_min(0.05)
                actions = (mean[:, None] + std[:, None] * torch.randn_like(actions)).clamp(-1, 1)
                actions[:, 0] = best_actions
        # The outer decision evaluates precisely the local search response that
        # supplies the executed prefix. No action replacement occurs afterwards.
        scores = route_cost + best_cost
        chosen = int(scores.argmin())
        target = targets[chosen:chosen+1]
        prefix = best_prefix[chosen:chosen+1]
        response_changed = chosen != int(route_cost.argmin())
        record = dict(anchor=info[chosen], selected_anchor=chosen,
                      selected_after_refinement_round=int(best_round[chosen]),
                      response_changed_retrieval_choice=response_changed,
                      route_cost=float(route_cost[chosen]), local_cost=float(best_cost[chosen]),
                      candidate_route_costs=route_cost.cpu().tolist(),
                      candidate_response_costs=best_cost.cpu().tolist(),
                      predicted_prefix_target_progress=float(distance(start, target)-distance(prefix, target)),
                      proposal_diversity=float(initial_actions.flatten(2).std(1, unbiased=False).mean()),
                      retrieval_cpu_wall_ms=retrieval_ms,
                      refinement_batches=3, candidate_world_transitions=480)
        self.trace.append(record)
        self.pending = (prefix.detach(), start.detach(), target.detach())
        return best_actions[chosen, 0], target[0], float(scores[chosen]), 5
