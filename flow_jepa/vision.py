from __future__ import annotations

import contextlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import file_hash


class Encoder:
    """Frozen official V-JEPA 2.1; past-only clips -> [B,32,1024]."""

    def __init__(self, c, root, device):
        self.c, self.device = c, device
        root = Path(root)
        code = root / "vendor" / "vjepa2"
        if not code.exists():
            code.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/" + c["repository"] + ".git",
                    str(code),
                ],
                check=True,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != c["revision"]:
            subprocess.run(
                ["git", "-C", str(code), "fetch", "origin", c["revision"]], check=True
            )
            subprocess.run(
                ["git", "-C", str(code), "checkout", "--detach", c["revision"]],
                check=True,
            )
        sys.path.insert(0, str(code))
        module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
        # Instantiate only the encoder, not the large unused pretraining predictor.
        self.model = getattr(module, c["architecture"])(
            patch_size=16,
            img_size=(384, 384),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        weights = root / "weights" / Path(c["checkpoint_url"]).name
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            tmp = weights.with_suffix(".partial")
            torch.hub.download_url_to_file(c["checkpoint_url"], str(tmp))
            tmp.replace(weights)
        state = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)[
            c["checkpoint_key"]
        ]
        state = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state.items()
        }
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.fingerprint = file_hash(weights)
        if c.get("checkpoint_sha256") and self.fingerprint != c["checkpoint_sha256"]:
            raise ValueError("Official encoder checkpoint hash mismatch")

    @torch.inference_mode()
    def __call__(self, clips):
        # uint8 [B,T,H,W,3], exclusively observations <= current physical time.
        x = (
            torch.as_tensor(np.asarray(clips), device=self.device)
            .permute(0, 4, 1, 2, 3)
            .float()
            / 255
        )
        b, _, t, _, _ = x.shape
        x = F.interpolate(
            x.transpose(1, 2).reshape(b * t, 3, *x.shape[-2:]),
            (self.c["input_size"],) * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = x.reshape(b, t, 3, self.c["input_size"], self.c["input_size"]).transpose(
            1, 2
        )
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None, None]
        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with ctx:
            z = self.model((x - mean) / std)
        if not torch.is_tensor(z):
            raise TypeError(
                "Expected the official final-layer tensor, not hierarchical outputs"
            )
        p = self.c["input_size"] // 16
        z = z.reshape(b, t // 2, p, p, -1).permute(0, 4, 1, 2, 3).float()
        z = F.adaptive_avg_pool3d(z, self.c["token_grid"]).flatten(2).transpose(1, 2)
        if z.shape[-1] != self.c["dim"] or not torch.isfinite(z).all():
            raise ValueError("Unexpected/nonfinite encoder output")
        return z


def history_clip(frames, index, history):
    return np.stack([frames[max(0, j)] for j in range(index - history + 1, index + 1)])
