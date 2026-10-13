"""Drop-in stable-worldmodel solver; imports native dependencies only on load."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .core import RefinementConfig
from .native import PlannerEngine


class SelfRefiningLatentPathSolver:
    def __init__(self, checkpoint: str, *, refinement: dict | None = None,
                 batch_size: int = 1, num_samples: int = 64, flow_steps: int = 16,
                 device: str = "cuda", seed: int = 1234, history_size: int = 3,
                 telemetry_path: str | None = None, runtime: Any = None):
        if min(batch_size, num_samples, flow_steps) < 1:
            raise ValueError("batch size, samples and flow steps must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; refusing silent CPU benchmarking")
        if runtime is None:
            from latent_planner import LatentPlannerRuntime
            runtime = LatentPlannerRuntime.from_checkpoint(checkpoint, device=self.device)
        else:
            runtime = runtime.to(self.device)
        self.config = RefinementConfig.from_dict(refinement or {})
        self.config.validate(flow_steps)
        self.model = PlannerEngine(runtime, self.config, history_size)
        self.batch_size, self.num_samples, self.flow_steps = batch_size, num_samples, flow_steps
        self.initial_seed = int(seed)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(self.initial_seed)
        # A second stream: additional inner loops cannot consume candidate seeds.
        self.perturb_gen = torch.Generator(device=self.device).manual_seed(self.initial_seed ^ 0x5E1F1E)
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        if self.telemetry_path:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.solve_calls = 0
        self.telemetry: list[dict] = []

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        from gymnasium.spaces import Box
        if not isinstance(action_space, Box):
            raise TypeError("SelfRefiningLatentPathSolver requires a Box action space")
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        if config.action_block != self.model.runtime.action_block:
            raise ValueError("execution action_block does not match the frozen checkpoint")
        self._action_space, self._n_envs, self._config = action_space, n_envs, config
        # Exact convention used by native LearnedLatentPathSolver (batched space).
        self._action_dim = int(math.prod(action_space.shape[1:]))

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action  # Preserve native LeFlow's no-warm-start policy.
        if not info_dict:
            raise ValueError("empty observation dictionary")
        total_envs = len(next(iter(info_dict.values())))
        if total_envs < 1:
            raise ValueError("empty environment batch")
        all_actions, all_costs, all_goal_costs = [], [], []
        records = []
        for lo in range(0, total_envs, self.batch_size):
            hi = min(lo + self.batch_size, total_envs)
            batch = {key: value[lo:hi] for key, value in info_dict.items()}
            out = self.model.plan(
                batch, horizon=self.horizon, num_samples=self.num_samples,
                flow_steps=self.flow_steps, generator=self.torch_gen,
                perturb_generator=self.perturb_gen,
            )
            all_actions.append(out["actions"])
            all_costs.append(out["costs"])
            all_goal_costs.append(out["goal_costs"])
            records.append({"solve": self.solve_calls, "env_start": lo, "env_end": hi,
                            "mode": self.config.mode, "samples": self.num_samples,
                            "flow_steps": self.flow_steps, "horizon": self.horizon,
                            "selected_candidates": out["selected_candidates"].tolist(),
                            "selected_model_cost": out["costs"].tolist(),
                            **out["work"]})
        self.solve_calls += 1
        self.telemetry.extend(records)
        if self.telemetry_path:
            with self.telemetry_path.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        return {"actions": torch.cat(all_actions),
                "costs": torch.cat(all_costs).tolist(),
                "goal_costs": torch.cat(all_goal_costs),
                "rollout_count": self.model.rollout_count}
