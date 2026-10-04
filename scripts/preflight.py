#!/usr/bin/env python
"""Fail before a training job if hardware, data, APIs, or W&B are unavailable."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import stable_worldmodel as swm
from btm_jepa.data import cache_root, LatentSegments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="pusht")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-wandb", action="store_true")
    args = p.parse_args()
    if not 1 <= args.gpus <= 4:
        raise ValueError("GPU limit is 1--4")
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(
            f"Requested {args.gpus} GPUs; found {torch.cuda.device_count()}"
        )
    if not hasattr(swm.World, "evaluate_from_dataset"):
        raise RuntimeError(
            "stable-worldmodel API mismatch; install requirements.txt exactly"
        )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpus": [torch.cuda.get_device_name(i) for i in range(args.gpus)],
            }
        )
    )
    if not args.skip_data:
        manifest = cache_root() / "latents" / args.task / "manifest.json"
        for split in ("train", "val", "test"):
            data = LatentSegments(manifest, split=split)
            first = data[0]
            print(
                split,
                len(data),
                {k: list(v.shape) for k, v in first.items() if torch.is_tensor(v)},
            )
    if not args.skip_wandb:
        import wandb

        # Authentication check, no dummy experiment or secret output.
        api = wandb.Api(timeout=20)
        _ = api.viewer
        print("W&B authentication verified")


if __name__ == "__main__":
    main()
