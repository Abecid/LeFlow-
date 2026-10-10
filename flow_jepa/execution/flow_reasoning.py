"""Candidate-conditioned latent deliberation and conditional action flow.

Physical states, imagined outcomes and internal workspace states remain distinct.
Only recorded actions receive recorded outcome labels. Frozen-world rollouts are
detached evidence for the reasoner, never physical supervision for generated actions.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..models import distance, positions
from .revision import RevisionPolicy


class ActionFlow(nn.Module):
    """Rectified-flow velocity on a complete five-block, 40-dimensional action."""
    def __init__(self, width, depth):
        super().__init__()
        self.action = nn.Linear(40, width)
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width),
                          nn.SiLU(), nn.Linear(2*width, width)) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 40))
        nn.init.normal_(self.out[-1].weight, std=.001)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, noisy, time, condition):
        h = self.action(noisy.flatten(-2)) + condition + self.time(time[..., None])
        for layer in self.layers:
            h = h + layer(h)
        return self.out(h).reshape_as(noisy)


class FlowReasoningPolicy(RevisionPolicy):
    def __init__(self, c):
        super().__init__(c)
        del self.output  # No GMM head or GMM loss in this candidate.
        cfg = c['flow_reasoning']; width = c['controller_grounded']['width']
        self.rounds, self.flow_steps = cfg['rounds'], cfg['solver_steps']
        self.candidates = cfg['candidates']
        self.usefulness_weight = cfg['usefulness_weight']
        self.prefix_weight = c['execution_revision']['prefix_weight']
        self.flow = ActionFlow(width, cfg['flow_depth'])
        # Each token preserves one action and its ordered spatial cost maps.
        self.evidence = nn.Sequential(nn.Linear(40+5*self.tokens+3, width),
                                      nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.imagined_type = nn.Parameter(torch.randn(width)*.02)
        nn.init.eye_(self.policy_context.weight)
        with torch.no_grad():
            self.policy_context.weight.mul_(.1)

    def prepare(self, start, target, goal, history, depth=None):
        """One causal initialization; further computation requires candidates."""
        b, p, _ = start.shape
        x = torch.cat([self.input(v)+self.types[i]
                       for i, v in enumerate((start, target, goal))], 1)
        context = self.trunk(x); base = context.mean(1)
        hs, hn, hp = (history[k] for k in ('start', 'next', 'predicted'))
        mask = history['mask']; valid = mask[:, :, None, None]
        residual = torch.where(valid, hn-hp, 0.)
        motion = torch.where(valid, hn-hs, 0.)
        q = target[:, None]
        summary = torch.stack((distance(hs, q)-distance(hn, q),
                               distance(hs, q)-distance(hp, q), distance(hp, hn)), -1)
        summary = torch.where(mask[:, :, None], summary, 0.)
        memory = self.error_input(residual*10)+self.movement_input(motion*10)
        memory = memory+self.progress_input(summary*100)[:, :, None]+self.history_type
        memory = memory+positions(p, memory.shape[-1], memory.device, memory.dtype)[None, None]
        memory = memory+positions(self.history_steps, memory.shape[-1], memory.device, memory.dtype)[None, :, None]
        keys = torch.cat((context, memory.flatten(1, 2)), 1)
        padding = torch.cat((torch.zeros(b, 3*p, device=start.device, dtype=torch.bool),
                             (~mask)[:, :, None].expand(-1, -1, p).flatten(1)), 1)
        state = self.workspace_queries[None].expand(b, -1, -1)+base[:, None]
        state = self.core(state, keys, padding)
        thought = self.workspace_norm(state.mean(1))
        return dict(base=base, state=state, thought=thought, score_thought=thought,
                    keys=keys, padding=padding, depth=1)

    def velocity(self, context, noisy, time):
        condition = context['base']+self.policy_context(context['thought'])
        if noisy.ndim == 4:
            condition = condition[:, None]
        return self.flow(noisy, time, condition)

    @torch.no_grad()
    def propose_context(self, context, count, *, noise=None, budget=None):
        b = len(context['base'])
        if noise is None:
            noise = torch.randn(b, count, 5, 8, device=context['base'].device)
        if noise.shape != (b, count, 5, 8):
            raise ValueError('Action noise must preserve batch/candidate/chunk axes')
        actions = noise.clone()  # Full Gaussian source, never restricted/truncated.
        for i in range(self.flow_steps):
            if budget is not None:
                budget.check()
            t = actions.new_full((b, count), i/self.flow_steps)
            actions = actions+self.velocity(context, actions, t)/self.flow_steps
        # The world only sees the completed, actuator-bounded clean proposal.
        return actions.clamp(-1, 1)

    def score_candidates(self, context, actions, predicted, target):
        b, n = actions.shape[:2]
        queries = target[:, None].expand(b, n, *target.shape[1:]).flatten(0, 1)
        thought = context['score_thought'][:, None].expand(b, n, -1).flatten(0, 1)
        correction = self.corrections(thought, actions.flatten(0, 1),
                                      predicted.flatten(0, 1), queries).reshape(b, n, 2)
        maps = 1-F.cosine_similarity(predicted.float(), target[:, None, None].float(), dim=-1, eps=1e-8)
        prefix = maps[:, :, 0].mean(-1)
        terminal = .5*(maps[:, :, -2].mean(-1)+maps[:, :, -1].mean(-1))
        w = self.prefix_weight
        raw = w*prefix+(1-w)*terminal
        cost = raw+w*correction[:, :, 0].clamp_min(0)+(1-w)*correction[:, :, 1].clamp_min(0)
        return dict(cost=cost, raw=raw, prefix=prefix, terminal=terminal,
                    correction=correction, maps=maps)

    def revise(self, context, actions, scores):
        # Atomic candidate tokens keep action/outcome correspondence. The set is
        # permutation equivariant; arbitrary candidate indices are not evidence.
        facts = torch.cat((actions.flatten(2), scores['maps'].flatten(2)*100,
                           scores['correction']*100, scores['cost'][:, :, None]*100), -1).detach()
        evidence = self.evidence(facts)+self.imagined_type
        keys = torch.cat((context['keys'], evidence), 1)
        padding = F.pad(context['padding'], (0, actions.shape[1]), value=False)
        state = self.core(context['state'], keys, padding)
        return {**context, 'state':state, 'thought':self.workspace_norm(state.mean(1)),
                'depth':context['depth']+1}

    def deliberate(self, start, target, goal, history, world, depth, *, noise=None):
        """Deployment-shaped computation: no teacher action/outcome argument."""
        if not 1 <= depth <= self.rounds:
            raise ValueError('Unsupported trained deliberation depth')
        context = self.prepare(start, target, goal, history)
        if noise is None:
            noise = torch.randn(len(start), self.candidates, 5, 8, device=start.device)
        contexts = [context]
        for _ in range(depth-1):
            with torch.no_grad():
                actions = self.propose_context(context, self.candidates, noise=noise)
                predicted = world.rollout(start[:, None].expand(-1, self.candidates, -1, -1).flatten(0, 1),
                                           actions.flatten(0, 1))
                predicted = predicted.reshape(len(start), self.candidates, 5, *start.shape[1:])
                scores = self.score_candidates(context, actions, predicted, target)
            context = self.revise(context, actions, scores)
            contexts.append(context)
        return contexts

    def forward(self, batch, world=None, weight=0., depth=None):
        if weight:
            raise ValueError('Generated actions cannot receive inherited expert-outcome consistency loss')
        depth = self.rounds if depth is None else depth
        z, actions = batch['z'], batch['a']
        history = self.factual_history(batch, world)
        contexts = self.deliberate(z[:, 0], z[:, 2], z[:, 3], history, world, depth)
        # Independent from all planning noise. No teacher-forced denoising view
        # is ever passed into the recurrent context above.
        noise = torch.randn_like(actions)
        t = torch.rand(len(z), device=z.device)
        noisy = (1-t[:, None, None])*noise+t[:, None, None]*actions
        target_velocity = actions-noise
        errors = torch.stack([(self.velocity(ctx, noisy, t)-target_velocity).square().mean((1, 2))
                              for ctx in contexts])
        flow_loss = errors.mean()
        useful = F.relu(errors[1:]-errors[0:1].detach()).mean() if depth > 1 else flow_loss*0

        # Only this factual branch sees the recorded action; outcomes are targets.
        choice = torch.randint(0, 3, (len(z),), device=z.device)
        q = torch.where((choice == 1)[:, None, None], z[:, 3], z[:, 2])
        q = torch.where((choice == 2)[:, None, None], z[:, 2].roll(1, 0), q)
        calibration_context = self.prepare(z[:, 0], q, z[:, 3], history)
        with torch.no_grad():
            predicted = world.rollout(z[:, 0], actions)
            labels = torch.stack((distance(z[:, 1], q)-distance(predicted[:, 0], q),
                .5*(distance(batch['z4'], q)+distance(z[:, 2], q)
                    -distance(predicted[:, -2], q)-distance(predicted[:, -1], q))), -1)
        corrections = self.corrections(calibration_context['score_thought'], actions, predicted, q)
        calibration = F.smooth_l1_loss(corrections/self.error_scale, labels/self.error_scale)
        loss = flow_loss+self.usefulness_weight*useful+self.calibration_weight*calibration
        metrics = dict(flow_loss=flow_loss.detach(), shallow_flow_loss=errors[0].mean().detach(),
            deepest_flow_loss=errors[-1].mean().detach(), paired_flow_regression=useful.detach(),
            paired_flow_improvement=(errors[0]-errors[-1]).mean().detach(),
            calibration_loss=calibration.detach(),
            recorded_raw_prefix_mae=labels[:, 0].abs().mean().detach(),
            recorded_corrected_prefix_mae=(labels[:, 0]-corrections[:, 0]).abs().mean().detach(),
            recorded_raw_terminal_mae=labels[:, 1].abs().mean().detach(),
            recorded_corrected_terminal_mae=(labels[:, 1]-corrections[:, 1]).abs().mean().detach(),
            reasoning_depth=loss.new_tensor(float(depth)), history_fraction=history['mask'].float().mean().detach())
        return loss, metrics
