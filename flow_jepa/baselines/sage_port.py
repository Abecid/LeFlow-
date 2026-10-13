"""SAGE-inspired MetaWorld port; NOT a reproduction of released SAGE checkpoints.

A goal/duration-conditional spatial subgoal predictor, an eight-component
40-D Gaussian-mixture action prior, and CEM refinement under the frozen world.
The observation/task interface differs from the SAGE authors' native benchmarks.
"""
from __future__ import annotations
import math
import numpy as np
import h5py
import torch
from torch import nn
from torch.nn import functional as F

from ..models import Trunk, distance, prediction_loss
from ..execution.model import ChunkPolicy
from .simple_action_flow import BasicActionSegments


class SageSegments(BasicActionSegments):
    """Sample a factual intermediate state at a horizon <= the final goal offset."""
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row, start, goal_offset = self.sample_spec(index)
        horizons = [h for h in self.c["sage_port"]["durations"] if h <= goal_offset]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 8107]))
        horizon = int(rng.choice(horizons))
        with h5py.File(self.root / row["path"], "r") as f:
            raw = f[self.state_key][start + horizon].astype(np.float32)
        item["sage_duration"] = torch.tensor(float(horizon), dtype=torch.float32)
        item["sage_subgoal"] = torch.from_numpy(((raw-self.mean)/self.std).copy())
        return item


class SageSubgoal(nn.Module):
    def __init__(self, c):
        super().__init__()
        width = c["controller_grounded"]["width"]
        dim = c["encoder"]["dim"]
        self.input = nn.Linear(dim, width)
        self.types = nn.Parameter(torch.randn(2, width)*.02)
        self.duration = nn.Sequential(nn.Linear(1,width), nn.SiLU(),nn.Linear(width,width))
        self.trunk = Trunk(width, 2, 4)
        self.out = nn.Linear(width,dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, start, goal, duration):
        if duration.shape != (len(start),):
            raise ValueError("duration must have one value per state")
        x = torch.cat((self.input(start)+self.types[0],
                       self.input(goal)+self.types[1]), dim=1)
        context = self.trunk(x + self.duration((duration/60.)[:,None])[:,None])
        n = start.shape[1]
        return start + self.out(context[:,:n])


class SagePolicy(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.subgoal = SageSubgoal(c)
        self.actor = ChunkPolicy(c)
        self.lambda_subgoal = float(c["sage_port"]["subgoal_weight"])

    def forward(self, batch, world=None, weight=0., depth=None):
        z,actions = batch["z"],batch["a"].flatten(1)
        q = self.subgoal(z[:,0],z[:,3],batch["sage_duration"])
        target = batch["sage_subgoal"]
        goal_loss = prediction_loss(q,target)
        # Train the actor on its *generated* subgoal, not a privileged expert
        # subgoal that is unavailable at inference. Recorded actions are the label.
        logits,means,scales = self.actor.distribution(z[:,0],q.detach(),z[:,3])
        logp = -.5*(((actions[:,None]-means)/scales).square()
                  +2*scales.log()+math.log(2*math.pi)).sum(-1)
        nll = -torch.logsumexp(F.log_softmax(logits,-1)+logp,-1).mean()/40.
        loss = nll+self.lambda_subgoal*goal_loss
        return loss,dict(action_nll=nll.detach(),subgoal_loss=goal_loss.detach())

    @torch.no_grad()
    def propose(self, start, subgoal, goal, count):
        return self.actor.propose(start,subgoal,goal,count)


class SageController:
    execution_blocks=1

    def __init__(self, model, world, c, bank):
        self.model, self.world = model,world
        self.durations = tuple(c["sage_port"]["deployment_durations"])
        self.per = int(c["sage_port"]["candidates_per_subgoal"])
        self.rounds = int(c["sage_port"]["cem_rounds"])
        if len(self.durations)!=8 or self.per!=4 or self.rounds!=3:
            raise ValueError("Registered SAGE port uses 8 durations x 4 plans x 3 CEM rounds")

    def begin_episode(self):
        pass

    @torch.no_grad()
    def plan(self,start,goal,*,budget=None):
        device=start.device
        starts=start.expand(8,-1,-1)
        goals=goal.expand(8,-1,-1)
        durations=torch.tensor(self.durations,device=device,dtype=start.dtype)
        if budget is not None: budget.check()
        subgoals=self.model.subgoal(starts,goals,durations)
        actions=self.model.propose(starts,subgoals,goals,self.per)
        best_cost=start.new_full((8,),float("inf"))
        best_plan=start.new_zeros((8,5,8))
        ix=torch.arange(8,device=device)
        for iteration in range(self.rounds):
            if budget is not None: budget.check()
            actions=actions.clamp(-1,1)
            predicted=self.world.rollout(starts[:,None].expand(-1,self.per,-1,-1).reshape(
                8*self.per,*start.shape[1:]),actions.reshape(8*self.per,5,8),budget=budget)
            pred=predicted.reshape(8,self.per,5,*start.shape[1:])
            target=subgoals[:,None]
            costs=.5*distance(pred[:,:,0],target)+.25*(
                distance(pred[:,:,3],target)+distance(pred[:,:,4],target))
            values,ids=costs.topk(2,dim=1,largest=False)
            candidates=actions[ix,ids[:,0]]
            better=values[:,0]<best_cost
            best_plan[better]=candidates[better]
            best_cost=torch.minimum(best_cost,values[:,0])
            if iteration+1<self.rounds:
                elites=actions[ix[:,None],ids]
                mean=elites.mean(dim=1)
                std=elites.std(dim=1,unbiased=False).clamp_min(.05)
                actions=(mean[:,None]+std[:,None]*torch.randn_like(actions)).clamp(-1,1)
                actions[:,0]=best_plan
        # Include a global goal incentive, since different generated subgoals
        # have different absolute local costs.
        total=best_cost+distance(subgoals,goals)
        selected=int(total.argmin())
        if budget is not None: budget.check()
        return best_plan[selected,0],subgoals[selected],float(total[selected]),5
