"""Non-preemptive GPU queue, frozen protocol, sequential training and held-out eval."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import config, digest, file_hash, git_revision, save_json


def visible_gpus():
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    active = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    busy = {r[0].strip() for r in csv.reader(active.splitlines()) if r}
    allowed = os.getenv("CUDA_VISIBLE_DEVICES")
    allowed = {x.strip() for x in allowed.split(",")} if allowed is not None else None
    result = []
    for row in csv.reader(query.splitlines()):
        index, uuid, memory, util = [x.strip() for x in row]
        if allowed is not None and index not in allowed and uuid not in allowed:
            continue
        if uuid not in busy and int(memory) < 1024 and int(util) < 5:
            result.append((index, uuid))
    return result


def acquire_gpus(maximum, root, wait_hours, required=None):
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("No NVIDIA GPU runtime: launch on the authorized server")
    if shutil.which("sinfo") and not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError(
            "Slurm detected: submit this command inside an allocated GPU job"
        )
    deadline = time.monotonic() + 3600 * wait_hours
    previous, steady = None, 0
    lock_dir = Path("/tmp") / f"flow-jepa-{os.getuid()}"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    while time.monotonic() < deadline:
        free = visible_gpus()
        previous, steady = free, steady + 1 if free == previous else 1
        save_json(
            root / "queue.json",
            dict(status="waiting_for_idle_gpus", candidates=free, idle_checks=steady),
        )
        if steady >= 3:
            held, selected = [], []
            for index, uuid in free[:maximum]:
                f = (lock_dir / (uuid + ".lock")).open("a+")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                held.append(f)
                selected.append((index, uuid))
            # Preserve the registered global batch on 1, 2 or 4 GPUs.
            count = next((n for n in (4, 2, 1) if n <= len(selected)), 0)
            if required is not None:
                count = required if len(selected) >= required else 0
            for f in held[count:]:
                f.close()
            held, selected = held[:count], selected[:count]
            if count and all(g in visible_gpus() for g in selected):
                save_json(
                    root / "queue.json",
                    dict(
                        status="acquired_idle_gpus",
                        devices=selected,
                        locking="advisory per-user locks; not a cluster-scheduler reservation",
                    ),
                )
                return selected, held
            for f in held:
                f.close()
        print(
            "Waiting for idle authorized GPUs; existing jobs are never stopped.",
            flush=True,
        )
        time.sleep(30)
    raise TimeoutError("GPU wait deadline exceeded; no training started")


def verify_manifest(root, c):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["protocol"] != digest(c) or manifest.get("fixture", True):
        raise ValueError("Wrong protocol or fixture data")
    for entry in manifest["entries"]:
        if file_hash(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Cached episode changed: {entry['id']}")
    return file_hash(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/flow_metaworld.json")
    p.add_argument("--root", required=True)
    p.add_argument("--gpus", type=int, default=4, choices=[1, 2, 4])
    p.add_argument("--wait-hours", type=float, default=168)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    c, root = config(a.config), Path(a.root).expanduser().resolve()
    plan = dict(
        code=git_revision(),
        configuration=c,
        protocol=digest(c),
        max_gpus=a.gpus,
        rendering={
            "backend": os.getenv("MUJOCO_GL", "egl"),
            "egl_devices_override": os.getenv("FLOW_EGL_DEVICES"),
            "software": os.getenv("LIBGL_ALWAYS_SOFTWARE", "0"),
            "gallium_driver": os.getenv("GALLIUM_DRIVER"),
        },
        independent_test_resets=len(c["training_tasks"] + c["heldout_tasks"])
        * c["data"]["test_episodes_per_task"],
    )
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit and push code before launching a campaign")
    root.mkdir(parents=True, exist_ok=True)
    campaign_lock = (root / "campaign.lock").open("a+")
    fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "campaign.json").exists() and json.loads(
        (root / "campaign.json").read_text()
    ) != plan:
        raise ValueError("Campaign already registered with another revision/protocol")
    save_json(root / "campaign.json", plan)
    # Verify online authentication before waiting days for GPUs.
    import wandb

    check = wandb.init(
        project=c["wandb"]["project"],
        mode="online",
        group=c["name"],
        job_type="launch-check",
        config=plan,
    )
    save_json(root / "wandb.json", dict(url=check.url, project=c["wandb"]["project"]))
    check.finish()
    allocation = root / "allocation.json"
    required = (
        json.loads(allocation.read_text())["world_size"]
        if allocation.exists()
        else None
    )
    if required is not None and required > a.gpus:
        raise ValueError("Resume needs the original number of GPUs")
    selected, held = acquire_gpus(a.gpus, root, a.wait_hours, required)
    save_json(allocation, dict(world_size=len(selected)))
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(x[0] for x in selected),
        FLOW_EGL_DEVICES=os.getenv(
            "FLOW_EGL_DEVICES", ",".join(x[0] for x in selected)
        ),
        MUJOCO_GL=plan["rendering"]["backend"],
        OMP_NUM_THREADS="2",
        WANDB_MODE="online",
        FLOW_DEVICE="cuda",
        CUDA_MODULE_LOADING="LAZY",
    )
    count = len(selected)

    def run(module, arguments, log_name):
        if (
            git_revision() != plan["code"]
            or subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        ):
            raise ValueError("Code changed during campaign")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:0",
            f"--nproc_per_node={count}",
            "-m",
            module,
            "--config",
            str(Path(a.config).resolve()),
            "--root",
            str(root),
            *map(str, arguments),
        ]
        (root / "logs").mkdir(exist_ok=True)
        save_json(
            root / "status.json",
            dict(status="running", stage=log_name, command=cmd, gpus=selected),
        )
        with (root / "logs" / f"{log_name}.log").open("a") as log:
            subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    try:
        run("flow_jepa.preflight", [], "gpu_preflight")
        if not (root / "manifest.json").exists():
            # The feature cache may live on scratch through this symlink while
            # manifests, checkpoints and evaluation records remain persistent.
            episode_storage = root / "episodes"
            episode_storage.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(episode_storage).free < 150 * 2**30:
                raise RuntimeError(
                    "Use a data volume with at least 150 GiB free for the full dense-feature cache and goal screening"
                )
            run("flow_jepa.data", [], "prepare_data")
        manifest_hash = verify_manifest(root, c)
        for seed in c["seeds"]:
            world_dir = root / "runs" / f"world_{seed}"
            if not (world_dir / "complete.json").exists():
                run(
                    "flow_jepa.train",
                    ["--run-dir", world_dir, "--method", "world", "--seed", seed],
                    f"world_{seed}",
                )
            world = world_dir / "best.pt"
            if not world.exists():
                raise FileNotFoundError("No validation-selected world checkpoint")
            for method in c["methods"]:
                if file_hash(root / "manifest.json") != manifest_hash:
                    raise ValueError("Manifest changed during campaign")
                out = root / "runs" / f"{method}_{seed}"
                if (
                    not method.startswith("cem_")
                    and not (out / "complete.json").exists()
                ):
                    run(
                        "flow_jepa.train",
                        [
                            "--run-dir",
                            out,
                            "--method",
                            method,
                            "--seed",
                            seed,
                            "--world",
                            world,
                        ],
                        f"{method}_{seed}",
                    )
        # Test only after every model finishes; no adaptive tuning against test results.
        for seed in c["seeds"]:
            world = root / "runs" / f"world_{seed}" / "best.pt"
            for method in c["methods"]:
                report = root / "test" / f"{method}_{seed}.json"
                # Evaluator validates hashes before skipping complete reports;
                # unfinished online uploads reuse the durable local results.
                arguments = [
                    "--world",
                    world,
                    "--method",
                    method,
                    "--seed",
                    seed,
                    "--output",
                    report,
                ]
                if not method.startswith("cem_"):
                    arguments += [
                        "--checkpoint",
                        root / "runs" / f"{method}_{seed}" / "best.pt",
                    ]
                run("flow_jepa.evaluate", arguments, f"test_{method}_{seed}")
        from .compare import compare

        reports = [
            json.loads((root / "test" / f"{m}_{s}.json").read_text())
            for m in c["methods"]
            for s in c["seeds"]
        ]
        comparison = compare(c, reports)
        save_json(root / "comparison.json", comparison)
        result = wandb.init(
            project=c["wandb"]["project"],
            mode="online",
            group=c["name"],
            job_type="comparison",
            config=plan,
        )
        for method, stats in comparison["table"].items():
            result.summary[f"{method}/success"] = stats["success"]
        artifact = wandb.Artifact(c["name"] + "-comparison", type="evaluation")
        artifact.add_file(str(root / "comparison.json"))
        result.log_artifact(artifact)
        result.finish()
        save_json(
            root / "status.json",
            dict(status="completed", comparison=str(root / "comparison.json")),
        )
    except BaseException as exc:
        save_json(root / "status.json", dict(status="failed", reason=str(exc)))
        raise
    finally:
        for lock in held:
            lock.close()


if __name__ == "__main__":
    main()
