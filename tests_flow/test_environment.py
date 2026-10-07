"""Physics tests run on CPU and do not acquire a renderer or a GPU."""

import json
from pathlib import Path

import numpy as np

from flow_jepa.common import episode_seed
from flow_jepa.environment import make_env


def test_expert_goal_not_clipped_by_cached_observation_bounds():
    from metaworld.policies import ENV_POLICY_MAP

    c = json.loads(
        (Path(__file__).resolve().parents[1] / "config/flow_metaworld.json").read_text()
    )["data"]
    env = make_env("reach", episode_seed("reach", "test", 0), c)
    try:
        obs, _ = env.reset()
        policy = ENV_POLICY_MAP["reach-v3"]()
        success = False
        for _ in range(200):
            np.testing.assert_allclose(obs[-3:], env._target_pos)
            obs, _, _, _, info = env.step(np.clip(policy.get_action(obs), -1, 1))
            if info["success"]:
                success = True
                break
        assert success, (
            "Broken expert observations would corrupt training data and goal images"
        )
    finally:
        env.close()


def test_seeded_reset_and_transition_are_paired():
    c = json.loads(
        (Path(__file__).resolve().parents[1] / "config/flow_metaworld.json").read_text()
    )["data"]
    a, b = make_env("push", 123, c), make_env("push", 123, c)
    try:
        x, _ = a.reset()
        y, _ = b.reset()
        np.testing.assert_array_equal(x, y)
        for action in np.random.default_rng(18).uniform(-1, 1, (10, 4)):
            x = a.step(action)[0]
            y = b.step(action)[0]
            np.testing.assert_array_equal(x, y)
    finally:
        a.close()
        b.close()
