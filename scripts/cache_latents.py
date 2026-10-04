#!/usr/bin/env python
"""Encode each source frame once; torchrun optionally distributes episodes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from btm_jepa.data import TASKS, atomic_json, cache_root, image_transform, file_sha256
from btm_jepa.distributed import initialize, barrier
from latent_planner import load_lewm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    rank, world, device = initialize(args.device)
    root = cache_root()
    source_path = root / "prepared" / args.task / "source.json"
    meta = json.loads(source_path.read_text())
    if file_sha256(meta["lewm_checkpoint"]) != meta["checkpoint_sha256"]:
        raise ValueError("Encoder checkpoint changed after data preparation")
    folder = root / "latents" / args.task
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        old = json.loads((folder / "manifest.json").read_text())
        if old["checkpoint_sha256"] != meta["checkpoint_sha256"] or old["source"]["signature"] != meta["source"]["signature"] or old["split"] != meta["split"]:
            raise ValueError("Existing latent cache differs; use a new STABLEWM_HOME")
        if rank == 0:
            print(f"Verified existing cache: {folder / 'manifest.json'}", flush=True)
        return
    path = folder / f"rank_{rank:02d}_of_{world:02d}.h5"
    tmp = path.with_suffix(".h5.partial")
    model = load_lewm(meta["lewm_checkpoint"]).to(device).eval().requires_grad_(False)
    transform = image_transform()
    mean, std = np.asarray(meta["source"]["action_mean"]), np.asarray(meta["source"]["action_std"])
    with h5py.File(meta["source"]["path"], "r") as src, h5py.File(tmp, "w") as out:
        for ep in tqdm(range(rank, meta["source"]["episodes"], world), desc=f"cache rank {rank}"):
            offset, length = meta["source"]["offsets"][ep], meta["source"]["lengths"][ep]
            g = out.create_group(f"episodes/{ep}")
            actions = (src["action"][offset:offset+length].astype(np.float32) - mean) / std
            g.create_dataset("action", data=actions.astype(np.float32), compression="lzf")
            for start in range(0, length, args.batch_size):
                pixels = torch.from_numpy(src["pixels"][offset+start:offset+min(length,start+args.batch_size)])
                pixels = transform(pixels.permute(0,3,1,2)).to(device)
                with torch.no_grad():
                    z = model.encode({"pixels": pixels[:, None]})["emb"][:, 0].float().cpu().numpy()
                if not np.isfinite(z).all():
                    raise ValueError(f"Nonfinite encoded latents in episode {ep}")
                if "z" not in g:
                    g.create_dataset("z", (length, z.shape[-1]), dtype="f4", chunks=True, compression="lzf")
                g["z"][start:start+len(z)] = z
        out.attrs["complete"] = True
        out.attrs["checkpoint_sha256"] = meta["checkpoint_sha256"]
    tmp.replace(path)
    barrier()
    if rank == 0:
        with h5py.File(path, "r") as f:
            latent_dim = f["episodes/0/z"].shape[-1]
        manifest = dict(**meta, latent_dim=latent_dim, preprocessing="imagenet_resize224_v1",
                        episode_shards={str(ep): f"rank_{ep%world:02d}_of_{world:02d}.h5" for ep in range(meta["source"]["episodes"])})
        atomic_json(folder / "manifest.json", manifest)
        print(f"Training-ready latent cache: {folder / 'manifest.json'}", flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
