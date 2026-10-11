"""Execution-conditioned local flow repair of a world-verified incumbent.

The second flow is trained on bounded ACTION corrections to recorded actions.
Generated actions never receive the recorded action's physical outcome label.
The incumbent and its scores are detached evidence, and the full world verifies
both pools under one fixed, episode-local calibrated scoring context.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..models import distance
from .flow_reasoning import ActionFlow
from .progress_ttt import ProgressTTTPolicy, ProgressTTTController, patch_cost


class RepairFlowTTTPolicy(ProgressTTTPolicy):
    def __init__(self, c):
        super().__init__(c)
        cfg = c['repair_flow_ttt']
        self.repair_radius = float(cfg['radius'])
        self.repair_steps = int(cfg['solver_steps'])
        self.repair_weight = float(cfg['loss_weight'])
        self.controller_flow_evaluations = self.flow_steps+self.repair_steps
        if not 0 < self.repair_radius <= 1 or self.repair_steps < 1 or self.rounds != 2:
            raise ValueError('Repair requires two rounds and a positive bounded radius/solver')
        width = c['controller_grounded']['width']
        self.repair_anchor = nn.Sequential(nn.Linear(40, width), nn.LayerNorm(width))
        self.repair_flow = ActionFlow(width, cfg['depth'])

    def revise(self, context, actions, scores):
        revised = super().revise(context, actions, scores)
        index = scores['cost'].detach().argmin(1)
        anchor = actions[torch.arange(len(actions), device=actions.device), index].detach()
        return {**revised, 'repair_anchor': anchor}

    def repair_velocity(self, context, noisy, time):
        condition = (context['base'] + self.policy_context(context['thought'])
                     + self.repair_anchor(context['repair_anchor'].flatten(1)))
        if noisy.ndim == 4:
            condition = condition[:, None]
        return self.repair_flow(noisy, time, condition)

    @torch.no_grad()
    def propose_context(self, context, count, *, noise=None, budget=None):
        if 'repair_anchor' not in context:
            return super().propose_context(context, count, noise=noise, budget=budget)
        b = len(context['base'])
        if noise is None:
            noise = torch.randn(b, count, 5, 8, device=context['base'].device)
        if noise.shape != (b, count, 5, 8) or count < 2:
            raise ValueError('Repair needs an incumbent slot and correctly shaped noise')
        # Slot zero is an exact incumbent; spend no solver work generating it.
        residual = noise[:, 1:].clone()
        for i in range(self.repair_steps):
            if budget is not None:
                budget.check()
            time = residual.new_full((b, count-1), i/self.repair_steps)
            residual = residual + self.repair_velocity(context, residual, time)/self.repair_steps
        anchor = context['repair_anchor'][:, None]
        repaired = (anchor + self.repair_radius*residual.clamp(-1, 1)).clamp(-1, 1)
        return torch.cat((anchor, repaired), 1)

    def forward(self, batch, world=None, weight=0., depth=None):
        if weight:
            raise ValueError('Generated actions cannot receive inherited expert-outcome labels')
        depth = self.rounds if depth is None else depth
        z, actions = batch['z'], batch['a']
        history = self.factual_history(batch, world)
        contexts = self.deliberate(z[:, 0], z[:, 2], z[:, 3], history, world, depth)
        noise = torch.randn_like(actions)
        time = torch.rand(len(z), device=z.device)
        noisy = (1-time[:, None, None])*noise + time[:, None, None]*actions
        flow_loss = (self.velocity(contexts[0], noisy, time)-(actions-noise)).square().mean()

        repair_loss = flow_loss*0
        before = after = saturation = flow_loss.detach()*0
        if depth == 2:
            context = contexts[1]
            anchor = context['repair_anchor']
            target = ((actions-anchor)/self.repair_radius).clamp(-1, 1)
            repair_noise = torch.randn_like(target)
            repair_time = torch.rand(len(z), device=z.device)
            interpolated = ((1-repair_time[:, None, None])*repair_noise
                            + repair_time[:, None, None]*target)
            velocity = self.repair_velocity(context, interpolated, repair_time)
            repair_loss = (velocity-(target-repair_noise)).square().mean()
            with torch.no_grad():
                estimate = (interpolated+(1-repair_time[:, None, None])*velocity).clamp(-1, 1)
                candidate = (anchor+self.repair_radius*estimate).clamp(-1, 1)
                before = (anchor-actions).square().mean()
                after = (candidate-actions).square().mean()
                saturation = ((actions-anchor).abs() > self.repair_radius).float().mean()

        # Only recorded actions have recorded next/terminal outcome labels.
        choice = torch.randint(0, 3, (len(z),), device=z.device)
        q = torch.where((choice == 1)[:, None, None], z[:, 3], z[:, 2])
        q = torch.where((choice == 2)[:, None, None], z[:, 2].roll(1, 0), q)
        calibration_context = self.prepare(z[:, 0], q, z[:, 3], history)
        with torch.no_grad():
            predicted = world.rollout(z[:, 0], actions)
            prefix_labels = patch_cost(z[:, 1], q)-patch_cost(predicted[:, 0], q)
            terminal_labels = .5*(distance(batch['z4'], q)+distance(z[:, 2], q)
                                   -distance(predicted[:, -2], q)-distance(predicted[:, -1], q))
        scores = self.score_candidates(calibration_context, actions[:, None], predicted[:, None], q)
        prefix, terminal = scores['prefix_error_maps'][:, 0], scores['correction'][:, 0, 1]
        calibration = .5*(F.smooth_l1_loss(prefix/self.error_scale, prefix_labels/self.error_scale)
                         + F.smooth_l1_loss(terminal/self.error_scale, terminal_labels/self.error_scale))
        prior = self.prefix_maps(calibration_context, actions[:, None], predicted[:, None, 0], q, prior=True)[:, 0]
        loss = flow_loss+self.repair_weight*repair_loss+self.calibration_weight*calibration
        metrics = dict(flow_loss=flow_loss.detach(), repair_flow_loss=repair_loss.detach(),
            repair_anchor_action_mse=before, repair_teacher_action_mse=after,
            repair_teacher_action_improvement=before-after, repair_target_clipped_fraction=saturation,
            calibration_loss=calibration.detach(),
            recorded_raw_prefix_mae=prefix_labels.mean(-1).abs().mean().detach(),
            recorded_corrected_prefix_mae=(prefix_labels-prefix).mean(-1).abs().mean().detach(),
            recorded_prior_prefix_mae=(prefix_labels-prior).mean(-1).abs().mean().detach(),
            recorded_patch_prefix_mae=(prefix_labels-prefix).abs().mean().detach(),
            recorded_raw_terminal_mae=terminal_labels.abs().mean().detach(),
            recorded_corrected_terminal_mae=(terminal_labels-terminal).abs().mean().detach(),
            fast_weight_delta_norm=calibration_context['fast_delta'].norm(dim=-1).mean().detach(),
            reasoning_depth=loss.new_tensor(float(depth)), history_fraction=history['mask'].float().mean().detach())
        return loss, metrics


class RepairFlowTTTController(ProgressTTTController):
    def plan_diagnostics(self, context, start, targets, actions, prefixes, chosen):
        details = super().plan_diagnostics(context, start, targets, actions, prefixes, chosen)
        difference = (actions-context['repair_anchor']).abs().amax((1, 2))
        details.update(repair_radius=self.model.repair_radius, repair_solver_steps=self.model.repair_steps,
            selected_distance_from_incumbent_linf=float(difference[chosen]),
            selected_is_unchanged_incumbent=bool(difference[chosen] == 0),
            surrogate_gradient_evaluations=0, refinement_source='world_verified_incumbent',
            repair_incumbent_preserved=True)
        return details
