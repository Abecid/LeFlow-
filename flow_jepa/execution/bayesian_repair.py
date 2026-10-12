"""ALPaCA posterior adaptation for incumbent-relative bounded flow repair.

Reference: Harrison et al., arXiv:1807.08912, equations 5--12; official
StanfordASL/ALPaCA@d06391a1. One observation is one executed transition,
not 32 independent patch measurements. The posterior describes errors in
the visual cost proxy; it is not a probability of physical task success.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..models import distance
from .flow_reasoning import FlowReasoningPolicy
from .progress_ttt import ProgressTTTPolicy, patch_cost
from .retrieval_repair import RetrievalRepairPolicy


def bayesian_update(features, labels, mask, prior, precision):
    """Exact ALPaCA normal equations, solved without an explicit inverse."""
    with torch.autocast(device_type=features.device.type, enabled=False):
        x = torch.where(mask[..., None], features.float(), 0.)
        y = torch.where(mask, labels.float(), 0.)
        mean0 = prior.float().expand(len(x), -1)
        gram = precision.float()[None] + x.transpose(-1, -2) @ x
        factor = torch.linalg.cholesky(gram)
        residual = y - (x * mean0[:, None]).sum(-1)
        delta = torch.cholesky_solve(x.transpose(-1, -2) @ residual[..., None], factor)[..., 0]
        return mean0 + delta, factor


def predictive_moments(features, mean, factor, noise_variance):
    """Mean, epistemic variance, and observation predictive variance."""
    solved = torch.linalg.solve_triangular(factor, features.transpose(-1, -2), upper=False)
    epistemic = noise_variance * solved.square().sum(-2)
    return (features * mean[:, None]).sum(-1), epistemic, epistemic + noise_variance


def joint_weight_points(mean, factor, noise_variance):
    """Deterministic shared Gaussian cubature: exact first/second moments.

    All candidates use the SAME posterior points. Independent points would
    invent uncertainty even when comparing an action with itself. Nonlinear
    clipping moments are a cubature approximation, not an analytic guarantee.
    """
    rank = mean.shape[-1]
    eye = torch.eye(rank, device=mean.device, dtype=mean.dtype).expand(len(mean), -1, -1)
    root = torch.linalg.solve_triangular(factor.transpose(-1, -2), eye, upper=True)
    offsets = (math.sqrt(rank) * noise_variance.sqrt() * root).transpose(-1, -2)
    return mean[:, None] + torch.cat((offsets, -offsets), 1)


def choose_with_margin(cost, joint_costs, beta):
    """Slot zero is the incumbent; compare shared-posterior cost differences."""
    gain_samples = joint_costs[:, :1] - joint_costs
    gain = cost[:, :1] - cost
    uncertainty = gain_samples.std(-1, unbiased=False)
    lower = gain - beta * uncertainty
    eligible = lower > 0
    eligible[:, 0] = True
    chosen = cost.masked_fill(~eligible, float('inf')).argmin(1)
    return chosen, dict(gain=gain, uncertainty=uncertainty, lower=lower, eligible=eligible)


class BayesianRepairPolicy(RetrievalRepairPolicy):
    def __init__(self, c):
        # Leave all previously released method constructors/weights unchanged.
        initialization = copy.deepcopy(c)
        initialization['flow_reasoning']['rounds'] = 2
        super().__init__(initialization)
        self.c = c
        cfg = c['bayesian_repair_ttt']
        self.rounds = int(cfg['training_rounds'])
        self.eval_rounds = int(c['flow_reasoning']['rounds'])
        if not 1 <= self.eval_rounds <= self.rounds:
            raise ValueError('Inference depth must be covered by training')
        self.acceptance_beta = float(cfg['acceptance_beta'])
        if self.acceptance_beta < 0:
            raise ValueError('Negative uncertainty penalty')
        self.nll_weight = float(cfg['nll_weight'])
        self.deployed_weight = float(cfg['deployed_loss_weight'])
        self.noise_floor = float(cfg['noise_floor'])
        self.precision_floor = float(cfg['precision_floor'])
        self.prior_tril = nn.Parameter(torch.eye(self.rank) * math.sqrt(self.ridge))
        self.log_noise_variance = nn.Parameter(torch.tensor(math.log(cfg['initial_noise_variance'])))
        self.controller_flow_evaluations = self.flow_steps + (self.eval_rounds - 1)*self.repair_steps

    def prior_precision(self):
        lower = self.prior_tril.tril()
        return lower @ lower.T + self.precision_floor*torch.eye(self.rank, device=lower.device)

    def noise_variance(self):
        return self.noise_floor + self.log_noise_variance.clamp(-12, 6).exp()

    def fit_posterior(self, history, target):
        mask = history['mask']; valid = mask[:, :, None, None]
        hs, hp, hn = [torch.where(valid, history[k], 0.) for k in ('start','predicted','next')]
        ha = torch.where(mask[:, :, None], history['action'], 0.)
        q = target[:, None].expand_as(hs)
        # Pool BEFORE fitting: each physical transition supplies one likelihood.
        features = self.transition_keys(hs, ha, hp, q).mean(-2)
        labels = (patch_cost(hn, q)-patch_cost(hp, q)).mean(-1)/self.error_scale
        return bayesian_update(features, labels, mask, self.fast_prior, self.prior_precision())

    def fit_memory(self, history, target):
        return self.fit_posterior(history, target)[0]

    def prepare(self, start, target, goal, history, depth=None):
        valid = history['mask'][:, :, None, None]
        history = {**history, **{k:torch.where(valid, history[k], 0.) for k in ('start','next','predicted')}}
        context = FlowReasoningPolicy.prepare(self, start, target, goal, history)
        mean, factor = self.fit_posterior(history, target)
        delta = mean-self.fast_prior
        state = context['state']+self.fast_condition(delta)[:, None]
        thought = self.workspace_norm(state.mean(1))
        return {**context, 'state':state, 'thought':thought, 'score_thought':thought,
                'fast_weights':mean, 'fast_delta':delta, 'posterior_factor':factor,
                'posterior_points':joint_weight_points(mean, factor, self.noise_variance()),
                'start':start, 'fast_support_steps':history['mask'].sum(1)}

    def candidate_features(self, context, actions, prefix, target):
        b, n = actions.shape[:2]
        start = context['start'][:, None].expand(b, n, *target.shape[1:])
        return self.transition_keys(start, actions[:, :, 0], prefix, target[:, None].expand_as(start)).mean(-2)

    def bounded(self, value):
        return self.correction_bound*(self.error_scale*value/self.correction_bound).tanh()

    def prefix_maps(self, context, actions, predicted_prefix, target, *, prior=False):
        features = self.candidate_features(context, actions, predicted_prefix, target)
        mean = self.fast_prior.expand(len(features), -1) if prior else context['fast_weights']
        value = self.bounded((features*mean[:, None]).sum(-1))
        return value[..., None].expand(*value.shape, target.shape[-2])

    def score_candidates(self, context, actions, predicted, target):
        scores = ProgressTTTPolicy.score_candidates(self, context, actions, predicted, target)
        features = self.candidate_features(context, actions, predicted[:, :, 0], target)
        mean, epistemic, predictive = predictive_moments(features, context['fast_weights'],
            context['posterior_factor'], self.noise_variance())
        point_errors = self.bounded(torch.einsum('bnr,bkr->bnk', features, context['posterior_points']))
        penalties = point_errors.clamp_min(0)
        terminal_penalty = (1-self.prefix_weight)*scores['correction'][:, :, 1].clamp_min(0)
        joint = scores['raw'][..., None] + terminal_penalty[..., None] + self.prefix_weight*penalties
        # Deployed objective and training use the same bounded optimism penalty.
        scores.update(cost=joint.mean(-1), joint_costs=joint,
            prefix_penalty=penalties.mean(-1), prefix_mean=mean,
            prefix_epistemic_variance=epistemic, prefix_predictive_variance=predictive,
            prefix_features=features)
        return scores

    def select(self, context, scores):
        if 'repair_anchor' not in context:
            return scores['cost'].argmin(1), None
        return choose_with_margin(scores['cost'], scores['joint_costs'], self.acceptance_beta)

    def revise(self, context, actions, scores):
        selected, _ = self.select(context, scores)
        revised = FlowReasoningPolicy.revise(self, context, actions, scores)
        ix = torch.arange(len(actions), device=actions.device)
        anchor = actions[ix, selected].detach()
        provenance = context.get('pool_provenance')
        source = provenance[ix, selected] if provenance is not None else context['repair_anchor_source']
        return {**revised, 'repair_anchor':anchor,
                'repair_root':context.get('repair_root', anchor), 'repair_anchor_source':source}

    @torch.no_grad()
    def propose_context(self, context, count, *, noise=None, budget=None):
        actions = super().propose_context(context, count, noise=noise, budget=budget)
        if 'repair_root' in context:
            # Total displacement is bounded across ALL rounds, not per-round only.
            root = context['repair_root'][:, None]
            actions = torch.maximum(torch.minimum(actions, root+self.repair_radius), root-self.repair_radius).clamp(-1,1)
            actions[:, 0] = context['repair_anchor']
        return actions

    def training_contexts(self, batch, world, history, depth):
        if not 1 <= depth <= self.rounds:
            raise ValueError('Unsupported trained refinement depth')
        z = batch['z']; context = self.prepare(z[:,0], z[:,2], z[:,3], history)
        contexts = [context]
        if depth == 1:
            self._retrieved_anchor_fraction = z.new_zeros(())
            return contexts
        if 'retrieved_actions' not in batch or not (batch['retrieved_episode_index'] != batch['bank_episode_index']).all():
            raise ValueError('Cross-episode retrieval required; teacher leakage forbidden')
        noise = torch.randn(len(z), self.candidates, 5, 8, device=z.device)
        for iteration in range(depth-1):
            with torch.no_grad():
                actions = self.propose_context(context, self.candidates, noise=noise)
                if iteration == 0:
                    enabled = batch['retrieval_enabled']; actions[enabled, -1] = batch['retrieved_actions'][enabled]
                    source = torch.zeros(actions.shape[:2], device=z.device, dtype=torch.long)
                    source[enabled, -1] = 1
                    context = {**context, 'pool_provenance':source}
                else:
                    context = {k:v for k,v in context.items() if k != 'pool_provenance'}
                predicted = world.rollout(z[:,0,None].expand(-1,self.candidates,-1,-1).flatten(0,1), actions.flatten(0,1))
                predicted = predicted.reshape(len(z),self.candidates,5,*z.shape[2:])
                scores = self.score_candidates(context, actions, predicted, z[:,2])
            context = self.revise(context, actions, scores)
            contexts.append(context)
        self._retrieved_anchor_fraction = (contexts[1]['repair_anchor_source']==1).float().mean()
        return contexts

    def forward(self, batch, world=None, weight=0., depth=None):
        if weight:
            raise ValueError('Generated actions cannot inherit physical outcome labels')
        depth = self.rounds if depth is None else depth
        z, actions = batch['z'], batch['a']; history = self.factual_history(batch, world)
        contexts = self.training_contexts(batch, world, history, depth)
        noise = torch.randn_like(actions); time = torch.rand(len(z), device=z.device)
        noisy = (1-time[:,None,None])*noise + time[:,None,None]*actions
        flow_loss = (self.velocity(contexts[0], noisy, time)-(actions-noise)).square().mean()
        repairs, saturations, before, after = [], [], [], []
        for context in contexts[1:]:
            anchor, root = context['repair_anchor'], context['repair_root']
            feasible = torch.maximum(torch.minimum(actions, root+self.repair_radius), root-self.repair_radius).clamp(-1,1)
            target = ((feasible-anchor)/self.repair_radius).clamp(-1,1)
            epsilon = torch.randn_like(target); t = torch.rand(len(z), device=z.device)
            xt = (1-t[:,None,None])*epsilon+t[:,None,None]*target
            velocity = self.repair_velocity(context, xt, t)
            repairs.append((velocity-(target-epsilon)).square().mean())
            with torch.no_grad():
                estimate = (anchor+self.repair_radius*(xt+(1-t[:,None,None])*velocity).clamp(-1,1)).clamp(-1,1)
                estimate = torch.maximum(torch.minimum(estimate,root+self.repair_radius),root-self.repair_radius)
                before.append((anchor-actions).square().mean()); after.append((estimate-actions).square().mean())
                saturations.append(((actions-anchor).abs()>self.repair_radius).float().mean())
        average = lambda xs:torch.stack(xs).mean() if xs else flow_loss*0
        repair_loss = average(repairs)
        choice = torch.randint(0,3,(len(z),),device=z.device)
        q = torch.where((choice==1)[:,None,None],z[:,3],z[:,2])
        q = torch.where((choice==2)[:,None,None],z[:,2].roll(1,0),q)
        context = self.prepare(z[:,0],q,z[:,3],history)
        with torch.no_grad():
            predicted = world.rollout(z[:,0],actions)
            prefix_label = (patch_cost(z[:,1],q)-patch_cost(predicted[:,0],q)).mean(-1)
            terminal_label = .5*(distance(batch['z4'],q)+distance(z[:,2],q)-distance(predicted[:,-2],q)-distance(predicted[:,-1],q))
        scores = self.score_candidates(context,actions[:,None],predicted[:,None],q)
        mean = scores['prefix_mean'][:,0]; variance = scores['prefix_predictive_variance'][:,0]
        standardized = (prefix_label/self.error_scale-mean).square()/variance
        nll = .5*(variance.log()+standardized).mean()
        prefix_applied = scores['prefix_penalty'][:,0]
        terminal = scores['correction'][:,0,1]
        applied_loss = .5*(F.smooth_l1_loss(prefix_applied/self.error_scale,prefix_label.clamp(0,self.correction_bound)/self.error_scale)
            +F.smooth_l1_loss(terminal.clamp_min(0)/self.error_scale,terminal_label.clamp(0,self.correction_bound)/self.error_scale))
        terminal_fit = F.smooth_l1_loss(terminal/self.error_scale,terminal_label/self.error_scale)
        calibration = self.nll_weight*nll+self.deployed_weight*applied_loss+.5*terminal_fit
        loss = flow_loss+self.repair_weight*repair_loss+self.calibration_weight*calibration
        prior = self.prefix_maps(context,actions[:,None],predicted[:,None,0],q,prior=True)[:,0].mean(-1)
        return loss, dict(flow_loss=flow_loss.detach(),repair_flow_loss=repair_loss.detach(),
            calibration_loss=calibration.detach(),posterior_nll=nll.detach(),deployed_calibration_loss=applied_loss.detach(),
            posterior_standardized_error=standardized.mean().detach(),
            posterior_90_coverage=(standardized<=1.644854**2).float().mean().detach(),
            posterior_noise_std=self.noise_variance().sqrt().detach()*self.error_scale,
            recorded_raw_prefix_mae=prefix_label.abs().mean().detach(),
            recorded_corrected_prefix_mae=(prefix_label-self.bounded(mean)).abs().mean().detach(),
            recorded_applied_prefix_mae=(prefix_label-prefix_applied).abs().mean().detach(),
            recorded_prior_prefix_mae=(prefix_label-prior).abs().mean().detach(),
            repair_anchor_action_mse=average(before).detach(),repair_teacher_action_mse=average(after).detach(),
            repair_target_clipped_fraction=average(saturations).detach(),
            fast_weight_delta_norm=context['fast_delta'].norm(dim=-1).mean().detach(),
            reasoning_depth=loss.new_tensor(float(depth)),history_fraction=history['mask'].float().mean().detach(),
            retrieved_anchor_fraction=self._retrieved_anchor_fraction.detach())
