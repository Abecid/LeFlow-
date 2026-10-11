"""Offline causal previous-plan anchors; deployment is unchanged.

A plan is synthesized at the preceding RECORDED state using information ending
at that state. Its suffix is evidence, not an executed action or outcome label.
This covers one-step plan reuse, not the full on-policy state/target distribution.
"""
import h5py
import numpy as np
import torch

from .retrieval_repair import (
    RetrievalRepairSegments, TrainingRetrieval, RetrievalRepairPolicy,
    RetrievalRepairController,
)


class ShiftedRepairSegments(RetrievalRepairSegments):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row, start, _ = self.sample_spec(index)
        h = self.c['latent_revision']['history_steps']
        shape = (h, *item['z'].shape[1:])
        hs = np.zeros(shape, np.float32)
        hn = np.zeros_like(hs)
        ha = np.zeros((h, 8), np.float32)
        mask = np.zeros(h, bool)
        previous = np.zeros(shape[1:], np.float32)
        if start > 0:
            t = start-1
            left = max(0, t-h)
            # End at t-1 relative to the current sample: never read the action
            # that leads to the current state into the previous decision.
            with h5py.File(self.root/row['path'], 'r') as f:
                states = (f[self.state_key][left:t+1].astype(np.float32)-self.mean)/self.std
                actions = f['actions'][left:t].astype(np.float32)
            previous = states[-1]
            count = t-left
            if count:
                hs[-count:], hn[-count:] = states[:-1], states[1:]
                ha[-count:] = actions
                mask[-count:] = True
        item.update(previous_start=torch.from_numpy(previous.copy()),
                    previous_available=torch.tensor(start > 0))
        for name, value in dict(history_start=hs, history_next=hn,
                                history_action=ha, history_mask=mask).items():
            item['previous_'+name] = torch.from_numpy(value.copy())
        return item


class ShiftedTrainingRetrieval(TrainingRetrieval):
    def augment(self, batch):
        result = super().augment(batch)
        valid = batch['previous_available']
        actions = torch.zeros_like(batch['a'])
        episodes = torch.full_like(batch['bank_episode_index'], -1)
        if valid.any():
            actions[valid], episodes[valid] = self.query(
                batch['previous_start'][valid], batch['z'][valid, 2],
                batch['bank_episode_index'][valid])
        return {**result, 'previous_retrieved_actions': actions,
                'previous_retrieved_episode_index': episodes}


class ShiftedRepairPolicy(RetrievalRepairPolicy):
    @torch.no_grad()
    def previous_plans(self, batch, world):
        valid = batch['previous_available']
        result = torch.zeros_like(batch['a'])
        if not valid.any():
            return result
        if not (batch['previous_retrieved_episode_index'][valid]
                != batch['bank_episode_index'][valid]).all():
            raise ValueError('Previous retrieval must exclude the query episode')
        prior = {k: batch['previous_'+k][valid] for k in
                 ('history_start', 'history_next', 'history_action', 'history_mask')}
        start = batch['previous_start'][valid]
        target, goal = batch['z'][valid, 2], batch['z'][valid, 3]
        history = self.factual_history(prior, world)
        context = self.prepare(start, target, goal, history)
        actions = self.propose_context(context, self.candidates)
        enabled = batch['retrieval_enabled'][valid]
        actions[enabled, -1] = batch['previous_retrieved_actions'][valid][enabled]
        # Two bounded rounds select a previous plan. Neither current observation
        # nor teacher actions/outcomes are used. No recursive prior-plan chain.
        for round_index in range(2):
            predicted = world.rollout(
                start[:, None].expand(-1, self.candidates, -1, -1).flatten(0, 1),
                actions.flatten(0, 1))
            predicted = predicted.reshape(len(start), self.candidates, 5, *start.shape[1:])
            scores = self.score_candidates(context, actions, predicted, target)
            if round_index == 0:
                context = self.revise(context, actions, scores)
                actions = self.propose_context(context, self.candidates)
        selected = scores['cost'].argmin(1)
        result[valid] = actions[torch.arange(len(start), device=start.device), selected]
        return result.detach()

    def training_contexts(self, batch, world, history, depth):
        z = batch['z']
        context = self.prepare(z[:, 0], z[:, 2], z[:, 3], history)
        if depth == 1:
            return [context]
        if depth != 2 or 'retrieved_actions' not in batch:
            raise ValueError('Deep shifted repair requires cross-episode evidence')
        if not (batch['retrieved_episode_index'] != batch['bank_episode_index']).all():
            raise ValueError('Own-episode retrieval would leak the teacher into planning')
        with torch.no_grad():
            previous = self.previous_plans(batch, world)
            actions = self.propose_context(context, self.candidates)
            enabled, valid = batch['retrieval_enabled'], batch['previous_available']
            actions[enabled, -1] = batch['retrieved_actions'][enabled]
            # Exactly the deployment transformation: drop the executed block,
            # reuse four remaining blocks, and keep a freshly proposed tail.
            actions[valid, 1, :-1] = previous[valid, 1:]
            source = torch.zeros(actions.shape[:2], device=z.device, dtype=torch.long)
            source[enabled, -1] = 1
            source[valid, 1] = 2
            context = {**context, 'pool_provenance': source}
            predicted = world.rollout(z[:, 0, None].expand(-1, self.candidates, -1, -1).flatten(0, 1),
                                      actions.flatten(0, 1))
            predicted = predicted.reshape(len(z), self.candidates, 5, *z.shape[2:])
            scores = self.score_candidates(context, actions, predicted, z[:, 2])
        revised = self.revise(context, actions, scores)
        self._retrieved_anchor_fraction = (revised['repair_anchor_source'] == 1).float().mean()
        self._shifted_anchor_fraction = (revised['repair_anchor_source'] == 2).float().mean()
        return [context, revised]

    def forward(self, batch, world=None, weight=0., depth=None):
        self._shifted_anchor_fraction = batch['a'].new_tensor(0.)
        loss, parts = super().forward(batch, world, weight, depth)
        parts['shifted_eligible_fraction'] = batch['previous_available'].float().mean()
        parts['shifted_anchor_fraction'] = self._shifted_anchor_fraction.detach()
        return loss, parts


# No deployment behavior or additional inference work is introduced.
ShiftedRepairController = RetrievalRepairController
