"""Episodic fast weights for action-conditioned, goal-relative progress errors.

Inner labels come only from executed transitions in the current episode. The
offline outer objective differentiates through the regularized fit on past
transitions, then supervises the NEXT recorded action and its observed outcome.
This is a new adaptation, not a reproduction of LaCT, SCOUT or AdaJEPA.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..models import distance, positions
from .flow_reasoning import FlowReasoningPolicy
from .flow_controller import FlowReasoningController


def patch_cost(state, target):
    return 1-F.cosine_similarity(state.float(), target.float(), dim=-1, eps=1e-8)


def ridge_update(keys, labels, mask, prior, ridge, patches):
    """Batched exact inner solve; sum of per-transition mean patch errors.

    min_w sum_{i,p} mask_i/P * (x_ip @ w - y_ip)^2
          + ridge * ||w-prior||^2.
    The solve is differentiable; it never mutates a parameter or optimizer.
    """
    if ridge <= 0 or patches < 1:
        raise ValueError('Fast-weight fit needs a positive prior and patch count')
    with torch.autocast(device_type=keys.device.type, enabled=False):
        valid = mask.bool()
        x = torch.where(valid[..., None], keys.float(), 0.)/math.sqrt(patches)
        y = torch.where(valid, labels.float(), 0.)/math.sqrt(patches)
        w0 = prior.float().expand(len(keys), -1)
        gram = x.transpose(-1, -2) @ x
        gram = gram+ridge*torch.eye(x.shape[-1], device=x.device)
        residual = y-(x*w0[:, None]).sum(-1)
        rhs = x.transpose(-1, -2) @ residual[..., None]
        delta = torch.cholesky_solve(rhs, torch.linalg.cholesky(gram)).squeeze(-1)
        return w0+delta


class ProgressTTTPolicy(FlowReasoningPolicy):
    def __init__(self, c):
        super().__init__(c)
        del self.prefix_error
        cfg = c['progress_ttt']; d = c['encoder']['dim']
        self.rank, self.ridge = cfg['rank'], cfg['ridge']
        if self.rank < 2 or self.ridge <= 0:
            raise ValueError('Invalid fast-weight dimensions or regularization')
        self.fast_keys = nn.Sequential(nn.Linear(3*d+8+16, cfg['key_width']),
            nn.LayerNorm(cfg['key_width']), nn.SiLU(), nn.Linear(cfg['key_width'], self.rank-1))
        self.fast_prior = nn.Parameter(torch.zeros(self.rank))
        self.fast_condition = nn.Linear(self.rank, c['controller_grounded']['width'], bias=False)
        nn.init.normal_(self.fast_condition.weight, std=.02)

    def empty_history(self, start):
        history = super().empty_history(start)
        history['action'] = start.new_zeros(len(start), self.history_steps, 8)
        return history

    @torch.no_grad()
    def factual_history(self, batch, world):
        return {**super().factual_history(batch, world), 'action':batch['history_action']}

    def transition_keys(self, start, action, predicted, target):
        """Pre-outcome features only, per spatial patch; no unexecuted suffix."""
        p = start.shape[-2]
        spatial = positions(p, 16, start.device, start.dtype).expand(*start.shape[:-2], p, 16)
        action = action[..., None, :].expand(*start.shape[:-1], 8)
        x = torch.cat((F.normalize(start.float(), dim=-1),
                       F.normalize(predicted.float(), dim=-1),
                       F.normalize(target.float(), dim=-1), action, spatial), -1)
        learned = self.fast_keys(x)
        # A constant coordinate allows a bias; unit norm bounds fit sensitivity.
        return F.normalize(torch.cat((learned, torch.ones_like(learned[..., :1])), -1), dim=-1)

    def fit_memory(self, history, target):
        mask = history['mask']; valid = mask[:, :, None, None]
        # Sanitize padding BEFORE nonlinearities, including nonfinite padding.
        hs, hp, hn = [torch.where(valid, history[k], 0.) for k in ('start','predicted','next')]
        ha = torch.where(mask[:, :, None], history['action'], 0.)
        q = target[:, None].expand_as(hs)
        keys = self.transition_keys(hs, ha, hp, q)
        labels = (patch_cost(hn, q)-patch_cost(hp, q))/self.error_scale
        return ridge_update(keys.flatten(1, 2), labels.flatten(1),
            mask[:, :, None].expand_as(labels).flatten(1), self.fast_prior, self.ridge, hs.shape[-2])

    def prepare(self, start, target, goal, history, depth=None):
        valid = history['mask'][:, :, None, None]
        history = {**history, **{k:torch.where(valid, history[k], 0.)
                                for k in ('start','next','predicted')}}
        context = super().prepare(start, target, goal, history)
        weights = self.fit_memory(history, target)
        delta = weights-self.fast_prior
        state = context['state']+self.fast_condition(delta)[:, None]
        thought = self.workspace_norm(state.mean(1))
        return {**context, 'state':state, 'thought':thought, 'score_thought':thought,
                'fast_weights':weights, 'fast_delta':delta, 'start':start,
                'fast_support_steps':history['mask'].sum(1)}

    def prefix_maps(self, context, actions, predicted_prefix, target, *, prior=False):
        b, n = actions.shape[:2]
        start = context['start'][:, None].expand(b, n, *target.shape[1:])
        q = target[:, None].expand_as(start)
        keys = self.transition_keys(start, actions[:, :, 0], predicted_prefix, q)
        weights = self.fast_prior.expand(b, -1) if prior else context['fast_weights']
        raw = self.error_scale*(keys*weights[:, None, None]).sum(-1)
        return self.correction_bound*(raw/self.correction_bound).tanh()

    def score_candidates(self, context, actions, predicted, target):
        b, n = actions.shape[:2]
        maps = patch_cost(predicted, target[:, None, None])
        prefix_maps = self.prefix_maps(context, actions, predicted[:, :, 0], target)
        prefix = maps[:, :, 0].mean(-1)
        terminal_maps = .5*(maps[:, :, -2]+maps[:, :, -1])
        terminal = terminal_maps.mean(-1)
        thought = context['score_thought'][:, None].expand(b, n, -1)
        inputs = torch.cat((thought, actions.flatten(2), maps[:, :, 0], terminal_maps), -1)
        terminal_error = self.correction_bound*self.terminal_error(inputs).squeeze(-1).tanh()
        correction = torch.stack((prefix_maps.mean(-1), terminal_error), -1)
        w = self.prefix_weight
        raw = w*prefix+(1-w)*terminal
        cost = raw+w*correction[:, :, 0].clamp_min(0)+(1-w)*correction[:, :, 1].clamp_min(0)
        return dict(cost=cost, raw=raw, prefix=prefix, terminal=terminal,
                    correction=correction, maps=maps, prefix_error_maps=prefix_maps)

    def corrections(self, *args, **kwargs):
        raise RuntimeError('Progress TTT requires a fitted causal context; use score_candidates')

    def forward(self, batch, world=None, weight=0., depth=None):
        if weight:
            raise ValueError('Generated actions cannot receive inherited expert-outcome consistency loss')
        depth = self.rounds if depth is None else depth
        z, actions = batch['z'], batch['a']
        history = self.factual_history(batch, world)
        contexts = self.deliberate(z[:, 0], z[:, 2], z[:, 3], history, world, depth)
        noise = torch.randn_like(actions); t = torch.rand(len(z), device=z.device)
        noisy = (1-t[:, None, None])*noise+t[:, None, None]*actions
        errors = torch.stack([(self.velocity(ctx, noisy, t)-(actions-noise)).square().mean((1, 2))
                              for ctx in contexts])
        flow_loss = errors.mean()
        useful = F.relu(errors[1:]-errors[0:1].detach()).mean() if depth > 1 else flow_loss*0
        choice = torch.randint(0, 3, (len(z),), device=z.device)
        q = torch.where((choice == 1)[:, None, None], z[:, 3], z[:, 2])
        q = torch.where((choice == 2)[:, None, None], z[:, 2].roll(1, 0), q)
        context = self.prepare(z[:, 0], q, z[:, 3], history)
        with torch.no_grad():
            predicted = world.rollout(z[:, 0], actions)
            prefix_labels = patch_cost(z[:, 1], q)-patch_cost(predicted[:, 0], q)
            terminal_labels = .5*(distance(batch['z4'], q)+distance(z[:, 2], q)
                                   -distance(predicted[:, -2], q)-distance(predicted[:, -1], q))
        scores = self.score_candidates(context, actions[:, None], predicted[:, None], q)
        prefix = scores['prefix_error_maps'][:, 0]
        terminal = scores['correction'][:, 0, 1]
        # Average patch error per example so 32 patches do not multiply the
        # prefix objective's weight relative to the inherited terminal objective.
        calibration = .5*(F.smooth_l1_loss(prefix/self.error_scale, prefix_labels/self.error_scale)
                         +F.smooth_l1_loss(terminal/self.error_scale, terminal_labels/self.error_scale))
        loss = flow_loss+self.usefulness_weight*useful+self.calibration_weight*calibration
        prior = self.prefix_maps(context, actions[:, None], predicted[:, None, 0], q, prior=True)[:, 0]
        metrics = dict(flow_loss=flow_loss.detach(), shallow_flow_loss=errors[0].mean().detach(),
            deepest_flow_loss=errors[-1].mean().detach(), paired_flow_regression=useful.detach(),
            paired_flow_improvement=(errors[0]-errors[-1]).mean().detach(), calibration_loss=calibration.detach(),
            recorded_raw_prefix_mae=prefix_labels.mean(-1).abs().mean().detach(),
            recorded_corrected_prefix_mae=(prefix_labels-prefix).mean(-1).abs().mean().detach(),
            recorded_prior_prefix_mae=(prefix_labels-prior).mean(-1).abs().mean().detach(),
            recorded_patch_prefix_mae=(prefix_labels-prefix).abs().mean().detach(),
            recorded_raw_terminal_mae=terminal_labels.abs().mean().detach(),
            recorded_corrected_terminal_mae=(terminal_labels-terminal).abs().mean().detach(),
            fast_weight_delta_norm=context['fast_delta'].norm(dim=-1).mean().detach(),
            reasoning_depth=loss.new_tensor(float(depth)), history_fraction=history['mask'].float().mean().detach())
        return loss, metrics


class ProgressTTTController(FlowReasoningController):
    def begin_episode(self):
        super().begin_episode()
        self.action_history = []

    @torch.no_grad()
    def observe(self, actual):
        if self.pending is not None:
            # Proposed actions become support ONLY after their outcome arrives.
            self.action_history.append(self.proposed_plan[0].detach().clone())
            self.action_history = self.action_history[-self.model.history_steps:]
        super().observe(actual)

    def model_history(self, start, count):
        history = super().model_history(start, count)
        actions = start.new_zeros(1, self.model.history_steps, 8)
        if len(self.action_history) != len(self.history):
            raise RuntimeError('Executed action/outcome histories are misaligned')
        for i, action in enumerate(self.action_history, start=self.model.history_steps-len(self.history)):
            actions[:, i] = action
        history['action'] = actions.expand(count, -1, -1)
        return history

    def plan_diagnostics(self, context, start, targets, actions, prefixes, chosen):
        prior = self.model.prefix_maps(context, actions[:, None], prefixes[:, None], targets, prior=True)
        return dict(fast_weight_delta_norm=float(context['fast_delta'][chosen].norm()),
                    fast_support_steps=int(context['fast_support_steps'][chosen]),
                    fast_weight_rank=self.model.rank, fast_weight_ridge=self.model.ridge,
                    prior_prefix_cost_correction=float(prior[chosen].mean()),
                    fast_fit_batches_per_decision=1, fast_weights_fixed_during_search=True,
                    adaptation_scope='current_episode_executed_transitions')
