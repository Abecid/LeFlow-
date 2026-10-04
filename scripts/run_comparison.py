#!/usr/bin/env python
"""Run a frozen, matched FM/BTM training campaign sequentially on 1--4 GPUs."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("flow", "btm")
PROTECTED = {"method", "seed", "run_dir", "resume", "wandb", "hydra", "defaults"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Use letters, digits, underscore, or hyphen for --name")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Training seeds must be distinct nonnegative integers")
    method_order = tuple(getattr(args, "method_order", METHODS))
    if len(method_order) != len(METHODS) or set(method_order) != set(METHODS):
        raise ValueError("Method order must contain flow and btm exactly once each")
    for value in args.overrides:
        key = value.split("=", 1)[0].split(".", 1)[0]
        if key in PROTECTED:
            raise ValueError(
                f"Campaign controls {key}; only shared training overrides are allowed"
            )
    cfg = OmegaConf.load(ROOT / "config/train/subgoals.yaml")
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if (
        cfg.device != "cuda"
        or not cfg.evaluation.enabled
        or cfg.evaluation.split != "val"
    ):
        raise ValueError(
            "Campaign requires CUDA and intermediate validation evaluation"
        )
    if cfg.global_batch_size % (args.gpus * cfg.micro_batch_size):
        raise ValueError(
            "global_batch_size must be divisible by gpus * micro_batch_size"
        )
    manifest = Path(cfg.data.manifest).expanduser().resolve()
    meta = json.loads(manifest.read_text())
    if meta.get("test_fixture", False):
        raise ValueError("A research campaign cannot use a test fixture")
    out = Path(os.environ["STABLEWM_HOME"]).expanduser().resolve() / "runs" / args.name
    jobs = []
    for seed in args.seeds:
        for method in method_order:
            job = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            job.method, job.seed = method, seed
            job.run_dir = str(out / f"{method}_{seed}")
            job.wandb.group = args.name
            job.wandb.name = f"{args.name}_{method}_{seed}"
            # The trainer fails rather than falling back to offline W&B.
            job.wandb.enabled, job.wandb.mode = True, "online"
            jobs.append(OmegaConf.to_container(job, resolve=True))
    return dict(
        format_version=1,
        name=args.name,
        gpus=args.gpus,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        seeds=args.seeds,
        method_order=list(method_order),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        manifest=str(manifest),
        manifest_sha256=sha256(manifest),
        split=meta["split"],
        encoder_sha256=meta["checkpoint_sha256"],
        campaign_dir=str(out),
        primary_goal_offset=max(cfg.evaluation.goal_offsets),
        targets={
            "pilot": "Match flow task success with lower measured full-controller latency; this is a target, not an established result.",
            "confirmation": "Use at least three training seeds before claiming a repeatable improvement; develop on validation only.",
        },
        jobs=jobs,
    )


def run(args):
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError(
            "Commit the working tree before a campaign so both methods have fixed code"
        )
    out = Path(plan["campaign_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "campaign.lock").open("w") as lock:
        # Prevent duplicate supervisors for this campaign.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = out / "comparison.json"
        if saved.exists():
            if not args.resume:
                raise FileExistsError(f"Campaign exists: {out}; use --resume")
            if json.loads(saved.read_text()) != plan:
                raise ValueError(
                    "Resume changed the code, data, seeds, GPU count, or comparison settings"
                )
        else:
            saved.write_text(json.dumps(plan, indent=2) + "\n")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/preflight.py"),
                "--task",
                plan["jobs"][0]["task"],
                "--manifest",
                plan["manifest"],
                "--gpus",
                str(args.gpus),
            ],
            cwd=ROOT,
            check=True,
        )
        configs = out / "configs"
        configs.mkdir(exist_ok=True)
        for cfg in plan["jobs"]:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
            if revision != plan["git_commit"] or dirty:
                raise ValueError(
                    "Code changed during the campaign; restore its committed revision"
                )
            if sha256(plan["manifest"]) != plan["manifest_sha256"]:
                raise ValueError("Shared dataset manifest changed during the campaign")
            run_dir = Path(cfg["run_dir"])
            done = run_dir / "training_complete.json"
            if done.exists():
                continue
            job = OmegaConf.create(cfg)
            if args.resume and (run_dir / "last.pt").exists():
                job.resume = str(run_dir / "last.pt")
            name = run_dir.name
            OmegaConf.save(job, configs / f"{name}.yaml")
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                f"--nproc_per_node={args.gpus}",
                "--master-addr=127.0.0.1",
                f"--master-port={args.port}",
                str(ROOT / "train_subgoals.py"),
                "--config-path",
                str(configs),
                "--config-name",
                name,
            ]
            print(f"Starting {name}; log: {out / (name + '.log')}", flush=True)
            env = os.environ.copy()
            for key, value in {
                "OMP_NUM_THREADS": "4",
                "MUJOCO_GL": "egl",
                "SDL_VIDEODRIVER": "dummy",
                "PYTHONUNBUFFERED": "1",
            }.items():
                env.setdefault(key, value)
            with (out / f"{name}.log").open("a") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            done.write_text(
                json.dumps(
                    {
                        "manifest_sha256": plan["manifest_sha256"],
                        "git_commit": plan["git_commit"],
                    }
                )
                + "\n"
            )
        print(
            f"Training complete: {saved}. Choose checkpoints using validation before final test evaluation.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072])
    parser.add_argument(
        "--method-order",
        nargs=2,
        choices=METHODS,
        default=METHODS,
        help="Run both methods once per seed in this order (default: flow btm)",
    )
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Shared Hydra overrides, e.g. epochs=10 global_batch_size=128",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
