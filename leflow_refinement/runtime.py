"""Frozen native checkpoint adapter. All variants share action clipping and scoring."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import torch
from torch import Tensor

from .core import PlannerOps, RefinementConfig
from .noise_search import noise_variants, plan as plan_candidates


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def variants(*, include_noise=False):
    """Fixed, untuned first comparison; equal flow work != equal total work."""
    result = {
        'leflow': RefinementConfig(mode='euler'),
        'more_samples': RefinementConfig(mode='euler', samples=96),
        'more_steps': RefinementConfig(mode='euler', steps=24),
        'pnp': RefinementConfig(mode='pnp'),
        'world_gradient': RefinementConfig(mode='gradient'),
        'world_pnp': RefinementConfig(mode='world_pnp'),
    }
    if include_noise:
        result.update(noise_variants())
    return result


class NativeRuntime:
    def __init__(self, runtime, *, history_size=3, lower=None, upper=None):
        if history_size < 1:
            raise ValueError('history_size must be positive')
        self.model = runtime.eval().requires_grad_(False)
        self.history_size = history_size
        self.device = next(runtime.parameters()).device
        if (lower is None) != (upper is None):
            raise ValueError('Provide both normalized action bounds or neither')
        self.lower = None if lower is None else torch.as_tensor(lower, device=self.device).float()
        self.upper = None if upper is None else torch.as_tensor(upper, device=self.device).float()
        if lower is not None and (self.lower.shape != self.upper.shape or
                not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all() or
                (self.upper <= self.lower).any()):
            raise ValueError('Invalid normalized action bounds')
        self.provenance = {}

    @classmethod
    def load(cls, checkpoint, *, world_checkpoint=None, device='cuda:0',
             lower=None, upper=None, history_size=3):
        from latent_planner import (LatentPlannerRuntime, LatentPathFlow,
                                    InverseDynamics, load_lewm)
        device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or 'arch' not in payload:
            raise ValueError('Expected a native LeFlow planner checkpoint')
        arch = payload['arch']
        flow = LatentPathFlow(**arch['flow'])
        inverse = InverseDynamics(**arch['inverse_dynamics'])
        flow.load_state_dict(payload['flow_state_dict'], strict=True)
        inverse.load_state_dict(payload['inverse_dynamics_state_dict'], strict=True)
        reference = world_checkpoint or payload['lewm_checkpoint']
        lewm = load_lewm(reference)
        runtime = LatentPlannerRuntime(lewm, flow, inverse, payload.get('action_block', 1))
        adapter = cls(runtime.to(device), lower=lower, upper=upper, history_size=history_size)
        adapter.provenance = dict(checkpoint_sha256=sha256(checkpoint),
            world_checkpoint=str(reference), world_sha256=sha256(reference) if Path(reference).is_file() else None,
            original_world_reference=str(payload.get('lewm_checkpoint')),
            action_block=runtime.action_block, architecture=arch,
            action_clipping='normalized_actuator_bounds' if lower is not None else 'none_native_legacy')
        return adapter

    def decode(self, paths: Tensor) -> Tensor:
        actions = self.model.decode_actions(paths)
        if self.lower is not None:
            if actions.shape[-1] != self.lower.numel():
                raise ValueError('Action bounds do not match checkpoint block width')
            actions = torch.maximum(torch.minimum(actions, self.upper), self.lower)
        return actions

    def rollout(self, start, actions, check=None):
        """Autoregressive native rollout WITHOUT the old no_grad decorator."""
        check = check or (lambda: None)
        states = start[:, None]
        for k in range(actions.shape[1]):
            check()
            ah = actions[:, max(0, k-self.history_size+1):k+1]
            zh = states[:, -ah.shape[1]:]
            predicted = self.model.lewm.predict(zh, self.model.lewm.action_encoder(ah))[:, -1:]
            states = torch.cat((states, predicted), dim=1)
        return states

    def ops(self, check=None):
        return PlannerOps(self.model.flow, self.decode,
                          lambda s, a: self.rollout(s, a, check), check)

    def plan(self, start, goal, cfg, *, seed=1234, initial_noise=None, check=None):
        synchronize(self.device)
        started = time.perf_counter()
        ops = self.ops(check)
        base_rng = torch.Generator(device=self.device).manual_seed(seed)
        perturb_rng = torch.Generator(device=self.device).manual_seed(seed + 1000000007)
        result = plan_candidates(start, goal, ops, cfg, initial_noise=initial_noise,
                                 generator=base_rng, perturb_generator=perturb_rng)
        synchronize(self.device)
        result['telemetry'] = dict(**ops.compute.as_dict(),
            planner_seconds=time.perf_counter()-started, configuration=asdict(cfg),
            initial_noise_seed=seed, perturbation_seed=seed+1000000007)
        if 'noise_search' in result:
            result['telemetry']['noise_search'] = result['noise_search']
        return result
