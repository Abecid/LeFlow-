"""Small synthetic latent fixture for trainer/DDP tests; NOT a robotics dataset."""

from pathlib import Path

import h5py
import numpy as np
import torch

from btm_jepa.data import atomic_json, episode_split, file_sha256


def create(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "placeholder.pt"
    torch.save({"test_fixture": True}, checkpoint)
    source = dict(
        path=str(folder / "not_a_robotics_dataset.h5"),
        signature="synthetic-integration-v1",
        lengths=[60] * 10,
        episodes=10,
        episode_ids=list(range(10)),
        action_dim=2,
        action_mean=[0, 0],
        action_std=[1, 1],
    )
    split = episode_split(source)
    rng = np.random.default_rng(123)
    with h5py.File(folder / "cache.h5", "w") as f:
        for ep in range(10):
            g = f.create_group(f"episodes/{ep}")
            a = rng.normal(size=(60, 2)).astype(np.float32)
            z = np.cumsum(np.tile(a, (1, 4)), axis=0).astype(np.float32) * 0.01
            g["z"], g["action"] = z, a
        f.attrs["complete"] = True
    manifest = dict(
        source=source,
        split=split,
        latent_dim=8,
        task="pusht",
        lewm_checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        episode_shards={str(i): "cache.h5" for i in range(10)},
    )
    atomic_json(folder / "manifest.json", manifest)
    return folder / "manifest.json"


if __name__ == "__main__":
    import sys

    print(create(sys.argv[1]))
