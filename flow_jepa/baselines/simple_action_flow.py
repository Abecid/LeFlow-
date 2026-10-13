"""Two controlled action-flow baselines for the shared image-goal MetaWorld benchmark.

vanilla_action_flow: one conditional rectified flow; no world search/retrieval/history.
action_flow_search: same action-flow architecture; eight local targets, three
independent candidate pools, frozen-world score; no recurrent workspace/TTT/repair.
Training labels are ALWAYS recorded actions from the same demonstration.
"""
from __future__ import annotations
import h5py
import numpy as np
import torch
from torch import nn

from ..models import Trunk, distance
from ..execution.flow_reasoning import ActionFlow
from ..execution.aligned import AlignedSegments


class BasicActionSegments(AlignedSegments):
    """Same recorded windows; avoid loading/transfer of unused TTT history."""
    def __getitem__(self, index):
        row, start, delta = self.sample_spec(index)
        with h5py.File(self.root / row['path'], 'r') as f:
            local = (f[self.state_key][start:start+6].astype(np.float32)-self.mean)/self.std
            goal = (f[self.state_key][start+delta].astype(np.float32)-self.mean)/self.std
            actions = f['actions'][start:start+5].astype(np.float32)
        states = np.stack((local[0], local[1], local[5], goal))
        return dict(z=torch.from_numpy(states.copy()), a=torch.from_numpy(actions.copy()),
                    sample_start=torch.tensor(start), sample_goal_offset=torch.tensor(delta))


class BasicActionFlowPolicy(nn.Module):
    def __init__(self, c):
        super().__init__()
        cfg = c["additional_baselines"]
        self.method = c["primary_method"]
        self.uses_subgoal = self.method == "action_flow_search"
        d = c["encoder"]["dim"]
        w = int(cfg["width"])
        self.visual = nn.Linear(d, w)
        self.types = nn.Parameter(torch.randn(3, w) * .02)
        self.trunk = Trunk(w, int(cfg["context_depth"]), 4)
        self.flow = ActionFlow(w, int(cfg["flow_depth"]))
        self.steps = int(cfg["flow_steps"])
        if self.steps < 1:
            raise ValueError("flow_steps must be positive")

    def condition(self, start, target, goal):
        if not self.uses_subgoal:
            # Do not silently train vanilla BC on oracle local targets.
            target = goal
        x = torch.cat([self.visual(v) + self.types[i]
                       for i, v in enumerate((start, target, goal))], dim=1)
        return self.trunk(x).mean(dim=1)

    def forward(self, batch, world=None, weight=0., depth=None):
        z = batch["z"]
        teacher = batch["a"]  # [B,5,8] recorded actions, never generated labels
        cond = self.condition(z[:, 0], z[:, 2], z[:, 3])
        eps = torch.randn_like(teacher)
        tau = torch.rand(len(teacher), device=teacher.device, dtype=teacher.dtype)
        noisy = (1 - tau[:, None, None]) * eps + tau[:, None, None] * teacher
        velocity = self.flow(noisy, tau, cond)
        loss = (velocity - (teacher - eps)).square().mean()
        return loss, {"flow_loss": loss.detach()}

    @torch.no_grad()
    def sample(self, start, target, goal, n, budget=None):
        b = len(start)
        context = self.condition(start, target, goal)[:, None]
        x = torch.randn(b, n, 5, 8, device=start.device, dtype=start.dtype)
        for i in range(self.steps):
            if budget is not None:
                budget.check()
            tau = x.new_full((b, n), i/self.steps)
            x = x + self.flow(x, tau, context) / self.steps
        return x.clamp(-1, 1)


class VanillaActionFlowPolicy(BasicActionFlowPolicy):
    pass


class ActionFlowSearchPolicy(BasicActionFlowPolicy):
    pass


class VanillaActionFlowController:
    execution_blocks = 1

    def __init__(self, model, world, c, bank):
        self.model = model

    def begin_episode(self):
        pass

    @torch.no_grad()
    def plan(self, start, goal, *, budget=None):
        actions = self.model.sample(start, goal, goal, 1, budget)
        if budget is not None:
            budget.check()
        return actions[0, 0, 0], goal[0], 0., 5


class ActionFlowSearchController:
    """Frozen-world sample-and-rank, independent pools, no workspace iteration."""
    execution_blocks = 1

    def __init__(self, model, world, c, bank):
        self.model, self.world, self.c, self.bank = model, world, c, bank
        self.rounds = int(c["additional_baselines"]["search_rounds"])
        self.per = int(c["additional_baselines"]["candidates_per_target"])
        if self.per != 4 or self.rounds != 3:
            raise ValueError("Pre-registered matched search uses 4 candidates x 3 rounds")

    def begin_episode(self):
        pass

    @torch.no_grad()
    def plan(self, start, goal, *, budget=None):
        if budget is not None:
            budget.check()
        targets, recorded, route_cost, _ = self.bank.query(start, goal, count=7)
        targets = torch.cat((targets, goal), 0)              # [8,32,D]
        route_cost = torch.cat((route_cost, distance(start, goal)), 0)
        starts = start.expand(8, -1, -1)
        goals = goal.expand(8, -1, -1)
        best_score = start.new_full((8,), float("inf"))
        best_action = start.new_zeros(8, 5, 8)
        for _ in range(self.rounds):
            actions = self.model.sample(starts, targets, goals, self.per, budget)
            actions[:7, -1] = recorded   # observed TRAIN actions, not query labels
            pred = self.world.rollout(starts[:, None].expand(-1,self.per,-1,-1)
                                      .reshape(8*self.per,*start.shape[1:]),
                                      actions.reshape(8*self.per,5,8),budget=budget)
            pred = pred.reshape(8,self.per,5,*start.shape[1:])
            q = targets[:, None]
            cost = .5*distance(pred[:,:,0],q)+.25*(distance(pred[:,:,3],q)
                                                       +distance(pred[:,:,4],q))
            values, choices = cost.min(dim=1)
            better = values < best_score
            candidates = actions[torch.arange(8,device=start.device),choices]
            best_action[better] = candidates[better]
            best_score = torch.minimum(best_score, values)
        chosen = int((route_cost + best_score).argmin())
        if budget is not None:
            budget.check()
        return best_action[chosen,0], targets[chosen], float((route_cost+best_score)[chosen]), 5
