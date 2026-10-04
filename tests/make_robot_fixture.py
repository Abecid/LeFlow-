"""Collect tiny random PushT episodes for end-to-end wiring checks, not training results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401

from btm_jepa.data import atomic_json, episode_split, file_sha256, inspect_source


def main(root):
    root = Path(root).resolve()
    path = root / "datasets/pusht_random_fixture.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        k: []
        for k in [
            "pixels",
            "action",
            "state",
            "proprio",
            "seed",
            "episode_idx",
            "step_idx",
        ]
    }
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", disable_env_checker=True)
    lengths = []
    for ep in range(10):
        obs, _ = env.reset(seed=100 + ep)
        env.action_space.seed(500 + ep)
        for step in range(36):
            action = env.action_space.sample()
            for k, v in dict(
                pixels=env.render(),
                action=action,
                state=obs["state"],
                proprio=obs["proprio"],
                seed=100 + ep,
                episode_idx=ep,
                step_idx=step,
            ).items():
                columns[k].append(v)
            obs, _, _, _, _ = env.step(action)
        lengths.append(36)
    env.close()
    with h5py.File(path, "w") as f:
        for k, v in columns.items():
            f.create_dataset(k, data=np.asarray(v), compression="lzf")
        f["ep_len"] = np.asarray(lengths)
        f["ep_offset"] = np.r_[0, np.cumsum(lengths)[:-1]]
    source = inspect_source(path)
    ckpt = root / "checkpoints/pusht/weights.pt"
    atomic_json(
        root / "prepared/pusht/source.json",
        dict(
            task="pusht",
            source=source,
            split=episode_split(source),
            lewm_checkpoint=str(ckpt),
            checkpoint_sha256=file_sha256(ckpt),
            test_fixture=True,
        ),
    )
    print(path)


if __name__ == "__main__":
    main(sys.argv[1])
