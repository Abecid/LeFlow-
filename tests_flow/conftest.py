import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture
def small_config():
    c = json.loads(
        (Path(__file__).resolve().parents[1] / "config/flow_metaworld.json").read_text()
    )
    c["training_tasks"], c["heldout_tasks"] = ["reach"], ["push"]
    c["encoder"].update(dim=8, token_grid=[1, 1, 2])
    c["model"].update(width=16, depth=1, heads=2, segments=2, chunk_steps=2)
    c["data"].update(
        action_repeat=1,
        train_episodes_per_task=3,
        validation_episodes_per_task=2,
        test_episodes_per_task=4,
    )
    c["training"].pop("budget_seconds", None)
    c["training"].update(
        world_steps=2,
        planner_steps=2,
        global_batch=4,
        micro_batch=2,
        workers=0,
        log_every=1,
        checkpoint_every=1,
        eval_every=2,
        consistency_steps=2,
        consistency_warmup=0,
    )
    c["evaluation"].update(
        candidates=3,
        flow_steps=2,
        cem_candidates=4,
        cem_elites=2,
        cem_iterations=2,
        refine_candidates=4,
        refine_elites=2,
        refine_iterations=2,
        short_horizon=2,
        long_horizon=4,
        bootstrap_samples=100,
    )
    return c
