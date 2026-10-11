"""Train the existing residual flow with a causal cross-episode retrieval slot.

Only the proposal/evidence distribution changes. Teacher actions still supervise
imitation residuals; recorded outcomes label only their own recorded actions.
No query-episode action or future observation is used as retrieved evidence.
"""
import torch
from ..models import distance
from .aligned import AlignedSegments
from .repair_flow_ttt import RepairFlowTTTPolicy, RepairFlowTTTController


class RetrievalRepairSegments(AlignedSegments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rows = [r for r in self.manifest['entries'] if r['split'] == 'train'
                and r['mode'] == 'expert' and r['expert_success'] and r['steps'] >= 60]
        self.bank_episode_indices = {r['id']: i for i, r in enumerate(rows)}

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row, _, _ = self.sample_spec(index)
        item['bank_episode_index'] = torch.tensor(self.bank_episode_indices[row['id']])
        # Across a global64 batch, exactly56 slots model retrieved targets and
        # eight model the direct goal. Independent of sampled labels/outcomes.
        item['retrieval_enabled'] = torch.tensor(int(index) % 8 != 7)
        return item


class TrainingRetrieval:
    """Five-block routes from the unchanged bank; exact reranking, own episode excluded."""
    def __init__(self, bank):
        self.bank = bank
        self.routes = bank.routes[bank.routes[:, 3] == 5]
        self.start_ids, self.end_ids = self.routes[:, 0], self.routes[:, 1]
        self.episodes = bank.episode.to(bank.raw.device)[self.start_ids]
        self.start_keys, self.end_keys = bank.keys[self.start_ids], bank.keys[self.end_ids]
        self.start_norms = self.start_keys.square().sum(-1)
        self.end_norms = self.end_keys.square().sum(-1)

    @torch.no_grad()
    def query(self, start, target, excluded_episodes):
        bank = self.bank
        sk, tk = bank.key(start), bank.key(target)
        scores = (self.start_norms[None] + sk.square().sum(-1)[:, None] - 2*sk@self.start_keys.T
                  + self.end_norms[None] + tk.square().sum(-1)[:, None] - 2*tk@self.end_keys.T)
        scores.masked_fill_(self.episodes[None] == excluded_episodes[:, None], float('inf'))
        shortlist = scores.topk(min(256, len(self.routes)), largest=False).indices
        b, n = shortlist.shape
        starts, ends = self.start_ids[shortlist], self.end_ids[shortlist]
        exact = (distance(bank.states(starts.flatten()).reshape(b,n,*start.shape[1:]), start[:,None])
                 + distance(bank.states(ends.flatten()).reshape(b,n,*target.shape[1:]), target[:,None]))
        exact.masked_fill_(self.episodes[shortlist] == excluded_episodes[:,None], float('inf'))
        if not torch.isfinite(exact.min(1).values).all():
            raise ValueError('No cross-episode training retrieval available')
        chosen = shortlist[torch.arange(b, device=start.device), exact.argmin(1)]
        episodes = self.episodes[chosen]
        assert (episodes != excluded_episodes).all()
        return bank.chunks[self.start_ids[chosen]].detach(), episodes

    def augment(self, batch):
        # The only query inputs are start, requested local target and an identity
        # exclusion; never teacher actions or recorded outcome-error labels.
        actions, episodes = self.query(batch['z'][:,0], batch['z'][:,2], batch['bank_episode_index'])
        return {**batch, 'retrieved_actions': actions, 'retrieved_episode_index': episodes}


class RetrievalRepairPolicy(RepairFlowTTTPolicy):
    def training_contexts(self, batch, world, history, depth):
        z = batch['z']
        context = self.prepare(z[:,0], z[:,2], z[:,3], history)
        if depth == 1:
            return [context]
        if depth != 2 or 'retrieved_actions' not in batch:
            raise ValueError('Deep retrieval repair requires verified cross-episode evidence')
        if not (batch['retrieved_episode_index'] != batch['bank_episode_index']).all():
            raise ValueError('Own-episode retrieval would leak the teacher into planning')
        with torch.no_grad():
            actions = self.propose_context(context, self.candidates)
            enabled = batch['retrieval_enabled']
            actions[enabled, -1] = batch['retrieved_actions'][enabled]
            provenance = torch.zeros(actions.shape[:2], device=z.device, dtype=torch.long)
            provenance[enabled, -1] = 1
            context = {**context, 'pool_provenance': provenance}
            predicted = world.rollout(z[:,0,None].expand(-1,self.candidates,-1,-1).flatten(0,1), actions.flatten(0,1))
            predicted = predicted.reshape(len(z),self.candidates,5,*z.shape[2:])
            scores = self.score_candidates(context, actions, predicted, z[:,2])
        revised = self.revise(context, actions, scores)
        # Scalar diagnostics are detached and cannot alter gradients or labels.
        self._retrieved_anchor_fraction = (revised['repair_anchor_source'] == 1).float().mean()
        return [context, revised]

    def revise(self, context, actions, scores):
        result = super().revise(context, actions, scores)
        provenance = context.get('pool_provenance')
        if provenance is not None:
            chosen = scores['cost'].detach().argmin(1)
            result['repair_anchor_source'] = provenance[torch.arange(len(actions),device=actions.device), chosen]
        return result

    def forward(self, batch, world=None, weight=0., depth=None):
        self._retrieved_anchor_fraction = batch['a'].new_tensor(0.)
        loss, parts = super().forward(batch, world, weight, depth)
        parts['retrieval_eligible_fraction'] = batch['retrieval_enabled'].float().mean()
        parts['retrieved_anchor_fraction'] = self._retrieved_anchor_fraction.detach()
        return loss, parts


class RetrievalRepairController(RepairFlowTTTController):
    def initial_pool_context(self, context, actions, recorded):
        source = torch.zeros(actions.shape[:2], device=actions.device, dtype=torch.long)
        source[:len(recorded), -1] = 1
        if self.previous_plan is not None:
            source[:,1] = 2
        return {**context, 'pool_provenance': source}

    def plan_diagnostics(self, context, start, targets, actions, prefixes, chosen):
        details = super().plan_diagnostics(context,start,targets,actions,prefixes,chosen)
        source = int(context['repair_anchor_source'][chosen])
        details['repair_anchor_source'] = ('generated','retrieved','shifted')[source]
        return details
