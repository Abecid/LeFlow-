"""LaCT-inspired nonlinear episodic memory on retrieval bounded flow repair.

The gated three-matrix cell and explicit differentiable MSE updates transfer
the earlier nonlinear-memory mechanism. Its task-specific value is observed
minus predicted *executed-prefix* goal cost. No imagined outcome is a label.
This small, four-transition adaptation is not LaCT's large-context regime.
"""
import math

import torch
from torch import nn

from ..leflow_ttt.model import read_memory, memory_gradients
from .flow_reasoning import FlowReasoningPolicy
from .progress_ttt import patch_cost
from .retrieval_repair import RetrievalRepairPolicy, RetrievalRepairController


class NonlinearRepairPolicy(RetrievalRepairPolicy):
    def __init__(self, c):
        super().__init__(c)
        cfg = c['nonlinear_repair_ttt']
        self.inner_steps = int(cfg['inner_steps'])
        self.inner_cap = float(cfg['gradient_cap'])
        self.elastic = float(cfg['elastic'])
        self.max_inner_lr = float(cfg['max_learning_rate'])
        rate = float(cfg['initial_learning_rate'])
        hidden = int(cfg['hidden'])
        if self.inner_steps < 1 or hidden < 1 or not 0 < rate < self.max_inner_lr or self.inner_cap <= 0 or self.elastic < 0:
            raise ValueError('Invalid nonlinear memory update configuration')
        del self.fast_prior
        self.memory_w0 = nn.Parameter(torch.randn(self.rank, hidden)/math.sqrt(self.rank))
        self.memory_w1 = nn.Parameter(torch.randn(hidden, 1)*.1/math.sqrt(hidden))
        self.memory_w2 = nn.Parameter(torch.randn(self.rank, hidden)/math.sqrt(self.rank))
        self.inner_lr_logit = nn.Parameter(torch.tensor(math.log(rate/(self.max_inner_lr-rate))))
        # Functional change in one readout per spatial patch conditions reasoning.
        self.fast_condition = nn.Linear(self.tokens, c['controller_grounded']['width'], bias=False)
        nn.init.normal_(self.fast_condition.weight, std=.02)

    def memory_prior(self, batch):
        return tuple(w[None].expand(batch, -1, -1) for w in
                     (self.memory_w0, self.memory_w1, self.memory_w2))

    def memory_keys(self, start, action, predicted, target):
        return math.sqrt(self.rank)*self.transition_keys(start, action, predicted, target)

    def fit_memory(self, history, target):
        mask = history['mask']; valid = mask[:, :, None, None]
        hs, hp, hn = [torch.where(valid, history[k], 0.) for k in ('start', 'predicted', 'next')]
        ha = torch.where(mask[:, :, None], history['action'], 0.)
        q = target[:, None].expand_as(hs)
        keys = self.memory_keys(hs, ha, hp, q).flatten(1, 2)
        values = ((patch_cost(hn, q)-patch_cost(hp, q))/self.error_scale).flatten(1)[..., None]
        rows = mask[:, :, None].expand(-1, -1, hs.shape[-2]).flatten(1)
        anchors = self.memory_prior(len(hs)); weights = anchors
        active = mask.any(1)[:, None, None]
        importance = tuple(torch.zeros_like(w) for w in weights)
        lr = self.max_inner_lr*self.inner_lr_logit.sigmoid()
        before = read_memory(weights, keys)
        for _ in range(self.inner_steps):
            # Explicit gradients preserve second-order outer differentiation and
            # also work inside the inference controller's no_grad context.
            grads = memory_gradients(weights, keys, values, rows)
            norm = sum(g.square().sum((1, 2)) for g in grads).clamp_min(1e-16).sqrt()
            factor = (self.inner_cap/norm).clamp_max(1)[:, None, None]
            updates = tuple(-lr*g*factor for g in grads)
            importance = tuple(.9*f+.1*u.detach().square() for f, u in zip(importance, updates))
            new = []
            for w, update, anchor, imp in zip(weights, updates, anchors, importance):
                precision = (imp/(imp.mean((1, 2), keepdim=True)+1e-8)).clamp_max(10)
                strength = self.elastic*precision
                adapted = (w+update+strength*anchor)/(1+strength)
                new.append(torch.where(active, adapted, anchor))
            weights = tuple(new)
        after = read_memory(weights, keys)
        denom = rows.sum(1).clamp_min(1)
        loss = lambda prediction: ((prediction-values).square()[..., 0]*rows).sum(1)/denom
        delta = sum((w-a).square().sum((1, 2)) for w, a in zip(weights, anchors)).clamp_min(1e-16).sqrt()
        return weights, dict(before=loss(before), after=loss(after),
                             delta=torch.where(mask.any(1), delta, 0.))

    def prepare(self, start, target, goal, history, depth=None):
        valid = history['mask'][:, :, None, None]
        history = {**history, **{k:torch.where(valid, history[k], 0.)
                                for k in ('start', 'next', 'predicted')}}
        context = FlowReasoningPolicy.prepare(self, start, target, goal, history)
        weights, diagnostic = self.fit_memory(history, target)
        # This query has no observed future outcome. It summarizes the fitted
        # function at the current state/target; action losses train its utility.
        query = self.memory_keys(start, start.new_zeros(len(start), 8), start, target)
        delta = (read_memory(weights, query)-read_memory(self.memory_prior(len(start)), query))[..., 0]
        state = context['state']+self.fast_condition(delta)[:, None]
        thought = self.workspace_norm(state.mean(1))
        return {**context, 'state':state, 'thought':thought, 'score_thought':thought,
                'fast_weights':weights, 'fast_delta':delta, 'memory_diagnostics':diagnostic,
                'start':start, 'fast_support_steps':history['mask'].sum(1)}

    def prefix_maps(self, context, actions, predicted_prefix, target, *, prior=False):
        b, n = actions.shape[:2]
        start = context['start'][:, None].expand(b, n, *target.shape[1:])
        q = target[:, None].expand_as(start)
        keys = self.memory_keys(start, actions[:, :, 0], predicted_prefix, q)
        weights = self.memory_prior(b) if prior else context['fast_weights']
        raw = self.error_scale*read_memory(weights, keys.flatten(1, 2))[..., 0].reshape(b, n, -1)
        return self.correction_bound*(raw/self.correction_bound).tanh()

    def forward(self, batch, world=None, weight=0., depth=None):
        self._memory_training_diagnostics = None
        loss, parts = super().forward(batch, world, weight, depth)
        diag = self._memory_training_diagnostics
        if diag is not None:
            parts.update(memory_support_before=diag['before'].mean().detach(),
                         memory_support_after=diag['after'].mean().detach(),
                         memory_parameter_delta=diag['delta'].mean().detach())
        parts.update(memory_inner_steps=loss.new_tensor(float(self.inner_steps)),
                     memory_inner_lr=(self.max_inner_lr*self.inner_lr_logit.sigmoid()).detach())
        return loss, parts

    def score_candidates(self, context, actions, predicted, target):
        if torch.is_grad_enabled():
            self._memory_training_diagnostics = context['memory_diagnostics']
        return super().score_candidates(context, actions, predicted, target)


class NonlinearRepairController(RetrievalRepairController):
    def plan_diagnostics(self, context, start, targets, actions, prefixes, chosen):
        details = super().plan_diagnostics(context, start, targets, actions, prefixes, chosen)
        details.pop('fast_weight_ridge')
        diag = context['memory_diagnostics']
        details.update(fast_fit='nonlinear_gated_three_matrix_mse',
            memory_inner_steps=self.model.inner_steps,
            memory_inner_learning_rate=float(self.model.max_inner_lr*self.model.inner_lr_logit.sigmoid()),
            memory_parameter_delta_norm=float(diag['delta'][chosen]),
            memory_support_before=float(diag['before'][chosen]),
            memory_support_after=float(diag['after'][chosen]))
        return details
