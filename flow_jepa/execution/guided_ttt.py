"""Execution-calibrated clean-action flow guidance with causal fast weights.

The compact energy distils factual JEPA prefix costs and corrects them using
observed errors. Its action gradients also receive a behavioral denoising loss.
That loss is NOT a counterfactual outcome label or an IQL/Q-learning objective.
QGF's clean-action, identity-Jacobian sampler is adapted to a cost (negative Q).
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..models import positions
from .progress_ttt import ProgressTTTPolicy, ProgressTTTController, patch_cost, ridge_update
from .flow_reasoning import FlowReasoningPolicy


class GuidedTTTPolicy(ProgressTTTPolicy):
    def __init__(self, c):
        super().__init__(c)
        cfg = c['guided_ttt']; d = c['encoder']['dim']; w = cfg['width']
        self.guidance = cfg
        del self.fast_keys
        # State work is cached across solver steps. Action dependence stays small.
        self.energy_state = nn.Linear(3*d+16, w)
        self.energy_action = nn.Linear(8, w, bias=False)
        self.energy_features = nn.Sequential(nn.LayerNorm(w), nn.SiLU(), nn.Linear(w, w), nn.SiLU())
        self.energy_keys = nn.Linear(w, self.rank-1)
        self.energy_base = nn.Linear(w, 1)
        nn.init.normal_(self.energy_base.weight, std=.01)
        nn.init.zeros_(self.energy_base.bias)

    def state_features(self, start, target):
        p = start.shape[-2]
        pos = positions(p, 16, start.device, start.dtype).expand(*start.shape[:-2], p, 16)
        x = torch.cat((F.normalize(start.float(), dim=-1), F.normalize(target.float(), dim=-1),
                       F.normalize(target.float()-start.float(), dim=-1), pos), -1)
        return self.energy_state(x)

    def action_features(self, state, action):
        return self.energy_features(state+self.energy_action(action)[..., None, :])

    def feature_keys(self, features):
        learned = self.energy_keys(features)
        return F.normalize(torch.cat((learned, torch.ones_like(learned[..., :1])), -1), dim=-1)

    def transition_keys(self, start, action, predicted, target):
        # No future outcome/predicted-state dependence: the SAME action-sensitive
        # error features are used by factual fitting, ranking and guidance.
        return self.feature_keys(self.action_features(self.state_features(start, target), action))

    def prepare(self, start, target, goal, history, depth=None):
        valid = history['mask'][:, :, None, None]
        history = {**history, **{k:torch.where(valid, history[k], 0.)
                                for k in ('start','next','predicted')}}
        context = FlowReasoningPolicy.prepare(self, start, target, goal, history, depth)
        context.update(energy_state=self.state_features(start, target),
                       energy_start_cost=patch_cost(start, target))
        # A prequential check: fit earlier support, predict the newest observed
        # residual, and compare with the prior. Never use in-sample fit as trust.
        mask = history['mask']; valid = mask[:, :, None, None]
        hs, hp, hn = [torch.where(valid, history[k], 0.) for k in ('start','predicted','next')]
        ha = torch.where(mask[:, :, None], history['action'], 0.)
        q = target[:, None].expand_as(hs)
        keys = self.transition_keys(hs, ha, hp, q)
        labels = (patch_cost(hn, q)-patch_cost(hp, q))/self.error_scale
        indices = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
        last = torch.where(mask, indices, -1).max(1).values
        earlier = mask & (indices < last[:, None])
        # Batch both independent solves and reuse state/action projections.
        both_masks = torch.cat((mask, earlier))[:, :, None].expand(-1, -1, hs.shape[-2]).flatten(1)
        both_weights = ridge_update(keys.flatten(1, 2).repeat(2, 1, 1), labels.flatten(1).repeat(2, 1),
            both_masks, self.fast_prior, self.ridge, hs.shape[-2])
        raw_weights, fitted = both_weights.chunk(2)
        ix = torch.arange(len(start), device=start.device); last = last.clamp_min(0)
        last_keys, last_labels = keys[ix,last], labels[ix,last]
        prior_mse = ((last_keys*self.fast_prior).sum(-1)-last_labels).square().mean(-1)
        fit_mse = ((last_keys*fitted[:, None]).sum(-1)-last_labels).square().mean(-1)
        gain = ((prior_mse-fit_mse)/(prior_mse+self.guidance['trust_floor'])).clamp(0, 1)
        trust = torch.where(mask.sum(1) >= 2, gain, 0.).detach()
        # This is a historical reliability heuristic, not calibrated uncertainty.
        context.update(fast_raw_weights=raw_weights, fast_trust=trust,
            trust_prior_mse=prior_mse.detach(), trust_adapted_mse=fit_mse.detach())
        weights = self.fast_prior+trust[:, None]*(raw_weights-self.fast_prior)
        context['fast_weights'] = weights
        context['state'] = context['state'] + self.fast_condition(weights-self.fast_prior)[:, None]
        context['fast_delta'] = weights-self.fast_prior
        context['thought'] = self.workspace_norm(context['state'].mean(1))
        context['score_thought'] = context['thought']
        context['start'] = start
        context['fast_support_steps'] = mask.sum(1)
        return context

    def energy_maps(self, context, prefix, *, prior=False):
        features = self.action_features(context['energy_state'][:, None], prefix)
        base = context['energy_start_cost'][:, None]+self.error_scale*self.energy_base(features).squeeze(-1)
        keys = self.feature_keys(features)
        weights = self.fast_prior.expand(len(prefix), -1) if prior else context['fast_weights']
        raw = self.error_scale*(keys*weights[:, None, None]).sum(-1)
        correction = self.correction_bound*(raw/self.correction_bound).tanh()
        return base, correction

    def energy_gradient(self, context, clean_prefix, *, create_graph=False, prior=False):
        # Detaching here is QGF's identity-Jacobian approximation: gradients are
        # evaluated at clean estimates, never through the velocity/ODE chain.
        with torch.enable_grad():
            action = clean_prefix.detach().requires_grad_(True)
            base, correction = self.energy_maps(context, action, prior=prior)
            energy = (base+correction).mean(-1)/self.error_scale
            gradient = torch.autograd.grad(energy.sum(), action, create_graph=create_graph)[0]
        return gradient

    def bounded_gradient(self, gradient):
        rms = gradient.square().mean(-1, keepdim=True).add(1e-12).sqrt()
        return gradient*(self.guidance['gradient_rms_cap']/rms).clamp(max=1.)

    @torch.no_grad()
    def propose_context(self, context, count, *, noise=None, budget=None):
        b = len(context['base'])
        if noise is None:
            noise = torch.randn(b, count, 5, 8, device=context['base'].device)
        if noise.shape != (b, count, 5, 8):
            raise ValueError('Action noise must preserve batch/candidate/chunk axes')
        actions = noise.clone()
        # Half the pool retains plain flow proposals. No extra candidate budget.
        guided = torch.zeros(count, device=actions.device, dtype=torch.bool)
        guided[1:1+count//2] = True
        for i in range(self.flow_steps):
            if budget is not None: budget.check()
            t = actions.new_full((b, count), i/self.flow_steps)
            velocity = self.velocity(context, actions, t)
            if i >= self.guidance['first_guided_step']:
                clean = (actions+(1-i/self.flow_steps)*velocity).clamp(-1, 1)
                gradient = self.energy_gradient(context, clean[:, guided, 0])
                drift = self.guidance['strength']*self.bounded_gradient(gradient)
                velocity[:, guided, 0] -= drift
            actions = actions+velocity/self.flow_steps
        return actions.clamp(-1, 1)

    def forward(self, batch, world=None, weight=0., depth=None):
        # Includes factual calibration and flow training through causal memory.
        loss, metrics = super().forward(batch, world, weight, depth)
        z, actions = batch['z'], batch['a']
        history = self.factual_history(batch, world)
        context = self.prepare(z[:, 0], z[:, 2], z[:, 3], history)
        # Critic value targets exist ONLY for recorded actions. Its base predicts
        # JEPA costs; the correction predicts their observed factual residual.
        with torch.no_grad():
            predicted = world.rollout(z[:, 0], actions[:, :1])[:, 0]
            world_maps = patch_cost(predicted, z[:, 2])
        base, _ = self.energy_maps(context, actions[:, None, 0])
        surrogate = F.smooth_l1_loss(base[:, 0]/self.error_scale, world_maps/self.error_scale)
        # A clean estimate from a teacher denoising view; this is an action-label
        # loss. Neither generated action is assigned the expert's outcome.
        t = torch.rand(len(z), device=z.device)*.5+.5
        noise = torch.randn_like(actions)
        noisy = (1-t[:, None, None])*noise+t[:, None, None]*actions
        with torch.no_grad():
            clean = (noisy+(1-t[:, None, None])*self.velocity(context, noisy, t)).clamp(-1, 1)
        gradient = self.energy_gradient(context, clean[:, None, 0], create_graph=True)
        eta = self.guidance['outer_step']
        revised = (clean[:, None, 0]-eta*self.bounded_gradient(gradient)).clamp(-1, 1)
        expert = actions[:, None, 0]
        before = (clean[:, None, 0]-expert).square().mean(-1)
        after = (revised-expert).square().mean(-1)
        # Compare against the identical prior-guided candidate; train the memory
        # for decision usefulness after a factual inner fit, not just lower MAE.
        prior_grad = self.energy_gradient(context, clean[:, None, 0], prior=True).detach()
        prior_action = (clean[:, None, 0]-eta*self.bounded_gradient(prior_grad)).clamp(-1, 1)
        prior_error = (prior_action-expert).square().mean(-1)
        meta_regression = F.relu(after-prior_error.detach()).mean()
        guided_bc = after.mean()
        anchor_gradient = self.energy_gradient(context, expert, create_graph=True)
        anchor = anchor_gradient.square().mean()
        cfg = self.guidance
        loss = loss+cfg['surrogate_weight']*surrogate+cfg['guided_bc_weight']*guided_bc
        loss = loss+cfg['meta_weight']*meta_regression+cfg['anchor_weight']*anchor
        metrics.update(critic_surrogate_loss=surrogate.detach(), guided_action_mse=guided_bc.detach(),
            unguided_action_mse=before.mean().detach(), prior_guided_action_mse=prior_error.mean().detach(),
            guided_action_improvement=(before-after).mean().detach(),
            adapted_guidance_improvement=(prior_error-after).mean().detach(),
            guidance_gradient_rms=gradient.square().mean().sqrt().detach(),
            expert_gradient_penalty=anchor.detach(), fast_trust=context['fast_trust'].mean().detach(),
            adaptation_regression=meta_regression.detach())
        return loss, metrics


class GuidedTTTController(ProgressTTTController):
    def plan_diagnostics(self, context, start, targets, actions, prefixes, chosen):
        details = super().plan_diagnostics(context, start, targets, actions, prefixes, chosen)
        base, correction = self.model.energy_maps(context, actions[:, None, 0])
        cfg = self.model.guidance
        details.update(fast_trust=float(context['fast_trust'][chosen]),
            trust_prior_mse=float(context['trust_prior_mse'][chosen]),
            trust_adapted_mse=float(context['trust_adapted_mse'][chosen]),
            guidance_surrogate_prefix_cost=float(base[chosen].mean()),
            guidance_corrected_prefix_cost=float((base+correction)[chosen].mean()),
            guidance_gradient_evaluations=self.model.rounds*(self.model.flow_steps-cfg['first_guided_step']),
            guidance_prefix_only=True, guidance_gradient_rms_cap=cfg['gradient_rms_cap'],
            fast_fit_batches_per_decision=1, fast_logical_fits=2,
            guidance_critic='factual_prefix_cost_surrogate',
            trust_is_calibrated_uncertainty=False)
        return details
