"""Native LeFlow evaluation with matched cases and structured, audited results.

Run from the complete LeFlow repository. This entry point does not modify
native eval.py, completed reports, or the custom MetaWorld sealed final set.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .protocol import (atomic_json, digest_json, jsonable, select_cases,
                       sha256_file, summarize_telemetry, validate_metrics)

PINNED_SWM = "06b1f4c24ccb237a9b2d6bb65bf010a6a5b41e83"
BASE_REPO_REVISION = "8fea74f96c867c9aa19ac763b85b22f11d62d3d0"


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",required=True,choices=["tworoom","pusht","reacher","cube"])
    p.add_argument("--checkpoint",required=True,help="Existing native LeFlow planner checkpoint")
    p.add_argument("--method-config",required=True)
    p.add_argument("--run-dir",required=True,help="NEW directory; never overwrite completed experiments")
    p.add_argument("--repo-root",default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--policy-seed",type=int,default=1234)
    p.add_argument("--num-eval",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--device",default="cuda")
    p.add_argument("--cache-dir")
    p.add_argument("--split-file",help="Optional explicit native heldout split, not the custom sealed final set")
    p.add_argument("--wandb-mode",choices=["online","offline","disabled"],default="online")
    p.add_argument("--no-video",action="store_true")
    p.add_argument("--allow-dirty",action="store_true",help="Development only; recorded in manifest")
    return p


def execute(args) -> dict:
    run_dir=Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True,exist_ok=False)
    started=time.time()
    meta={"status":"preflight","benchmark_family":"native_leflow","task":args.task,
          "seed":args.seed,"training_updates":0,"arguments":vars(args),
          "started_unix":started,"inspected_base_revision":BASE_REPO_REVISION}
    atomic_json(run_dir/"status.json",meta)
    world=dataset=wb=None
    try:
        root=Path(args.repo_root).resolve()
        for name in ["latent_planner.py","jepa.py","eval.py","config/eval"]:
            if not (root/name).exists():
                raise FileNotFoundError(f"complete native repository required; missing {root/name}")
        sys.path.insert(0,str(root))
        os.environ.setdefault("MUJOCO_GL","egl")
        # Refuse a silently different stable-worldmodel success/execution API.
        dist=importlib.metadata.distribution("stable-worldmodel")
        direct=json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info",{}).get("commit_id")!=PINNED_SWM:
            raise RuntimeError("install the repository's pinned stable-worldmodel requirement before evaluation")
        import numpy as np
        import torch
        import hydra
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from sklearn.preprocessing import StandardScaler
        import stable_worldmodel as swm
        from btm_jepa.env import World
        from eval import get_dataset, img_transform, load_heldout_episodes
        from latent_planner import _resolve_checkpoint
        from .core import RefinementConfig
        from .native import state_fingerprint
        from .preflight import validate_native_runtime
        from .solver import SelfRefiningLatentPathSolver

        device=torch.device(args.device)
        if device.type=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device available; no GPU benchmark was started")
        torch.manual_seed(args.policy_seed)
        np.random.seed(args.seed)
        method=json.loads(Path(args.method_config).read_text())
        if set(method)!={"name","num_samples","flow_steps","refinement"}:
            raise ValueError("method config must contain name, num_samples, flow_steps, refinement")
        RefinementConfig.from_dict(method["refinement"]).validate(method["flow_steps"])
        checkpoint=_resolve_checkpoint(Path(args.checkpoint).expanduser())
        checkpoint_sha=sha256_file(checkpoint)
        revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
        dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=normal"],cwd=root,text=True).strip()
        if dirty and not args.allow_dirty:
            raise RuntimeError("tracked source is dirty; commit it before running, or explicitly mark a development run")
        files=[root/x for x in ["latent_planner.py","jepa.py","eval.py","module.py","btm_jepa/env.py"]]
        files+=sorted((root/"self_refine").glob("*.py"))
        files+=sorted((root/"config/eval").rglob("*.yaml"))
        source={str(f.relative_to(root)):sha256_file(f) for f in files}
        with initialize_config_dir(version_base=None,config_dir=str(root/"config/eval")):
            cfg=compose(config_name=args.task,overrides=["solver=latent_flow"])
        cfg.seed=args.seed; cfg.eval.num_eval=args.num_eval; cfg.policy=str(checkpoint)
        if args.cache_dir:
            from omegaconf import open_dict
            with open_dict(cfg): cfg.cache_dir=args.cache_dir
        if args.split_file:
            cfg.episode_split.enabled=True; cfg.episode_split.mode="eval"
            cfg.episode_split.split_file=str(Path(args.split_file).resolve())
        if cfg.plan_config.horizon*cfg.plan_config.action_block>cfg.eval.eval_budget:
            raise ValueError("native planning horizon exceeds execution budget")
        cfg.world.max_episode_steps=2*cfg.eval.eval_budget
        dataset=get_dataset(cfg,cfg.eval.dataset_name)
        data_sha=sha256_file(dataset.h5_path)
        col="episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        cases=select_cases(dataset.get_col_data(col),dataset.get_col_data("step_idx"),
            goal_offset=cfg.eval.goal_offset_steps,num_eval=args.num_eval,seed=args.seed,
            heldout=load_heldout_episodes(cfg))
        atomic_json(run_dir/"cases.json",cases)
        process={}; stats={}
        # Preserve native evaluation's normalization source exactly. Do not call
        # this an independent train-only-normalization experiment.
        for key in cfg.dataset.keys_to_cache:
            if key=="pixels": continue
            values=dataset.get_col_data(key)
            values=values[~np.isnan(values).any(axis=1)]
            scaler=StandardScaler().fit(values)
            process[key]=scaler
            stats[key]={"mean":scaler.mean_,"scale":scaler.scale_}
            if key!="action": process[f"goal_{key}"]=scaler
        solver=SelfRefiningLatentPathSolver(str(checkpoint),device=str(device),
            seed=args.policy_seed,batch_size=args.batch_size,
            num_samples=method["num_samples"],flow_steps=method["flow_steps"],
            refinement=method["refinement"],telemetry_path=str(run_dir/"controller.jsonl"))
        gate=validate_native_runtime(solver.model.runtime,device=device,
                                    horizon=cfg.plan_config.horizon)
        atomic_json(run_dir/"preflight.json",gate)
        protocol={"task":args.task,"family":"native_leflow", "cases":cases,
            "data_sha256":data_sha,"checkpoint_sha256":checkpoint_sha,
            "model_state_sha256":gate["state_sha256"],"world":OmegaConf.to_container(cfg.world,resolve=True),
            "plan_config":OmegaConf.to_container(cfg.plan_config,resolve=True),
            "evaluation":OmegaConf.to_container(cfg.eval,resolve=True),
            "normalization":stats,"normalization_source":"native evaluation dataset",
            "policy_seed":args.policy_seed,"batch_size":args.batch_size,"swm_commit":PINNED_SWM,
            "seed":args.seed,"image_transform":"native eval.img_transform",
            "torch":torch.__version__,"device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
            "rank_score":"native terminal rollout MSE; no extra consistency score",
            "native_source_hashes":{key:source[key] for key in ["latent_planner.py","eval.py","jepa.py","module.py","btm_jepa/env.py"]}}
        meta.update({"method":method,"status":"running","protocol":jsonable(protocol),
                     "protocol_hash":digest_json(protocol),"source_revision":revision,
                     "source_files":source,"source_sha256":digest_json(source),"dirty":bool(dirty),
                     "weights_before":gate["state_sha256"],"checkpoint_sha256":checkpoint_sha})
        atomic_json(run_dir/"manifest.json",meta)
        atomic_json(run_dir/"status.json",meta)
        if args.wandb_mode!="disabled":
            import wandb
            wb=wandb.init(project="leflow-self-refine",name=f"{args.task}-{method['name']}-s{args.seed}",
                group=f"native-{revision[:8]}",mode=args.wandb_mode,
                dir=str(run_dir),config={"method":method,"protocol_hash":meta["protocol_hash"],
                                       "task":args.task,"seed":args.seed,"training_updates":0})
        world=World(**cfg.world,image_shape=(224,224))
        callable_config=cfg.eval.get("callables")
        callables=OmegaConf.to_container(callable_config,resolve=True) if callable_config is not None else []
        # Upstream evaluator merely warns if a reset callable is absent. That
        # would invalidate benchmark starts: fail instead of skipping resets.
        for env in world.envs.unwrapped.envs:
            for spec in callables:
                if not hasattr(env.unwrapped,spec["method"]):
                    raise RuntimeError(f"native reset method {spec['method']} is unavailable")
        config=swm.PlanConfig(**cfg.plan_config)
        policy=swm.policy.WorldModelPolicy(solver=solver,config=config,process=process,
                         transform={"pixels":img_transform(cfg),"goal":img_transform(cfg)})
        world.set_policy(policy)
        eval_started=time.perf_counter()
        metrics=world.evaluate_from_dataset(dataset,episodes_idx=[c["episode"] for c in cases],
            start_steps=[c["start"] for c in cases],goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,callables=callables,
            save_video=not args.no_video,video_path=run_dir/"videos")
        wall=time.perf_counter()-eval_started
        outcomes=validate_metrics(metrics,cases)
        timing=summarize_telemetry(solver.telemetry)
        after=state_fingerprint(solver.model.runtime)
        if after!=meta["weights_before"] or sha256_file(checkpoint)!=checkpoint_sha:
            raise RuntimeError("frozen weights/checkpoint changed during evaluation")
        summary={**meta,"status":"completed","cases":outcomes,"metrics":jsonable(metrics),
                 "successes":sum(c["success"] for c in outcomes),"evaluations":len(cases),
                 "success_rate_percent":float(metrics["success_rate"]),"timing":timing,
                 "evaluation_wall_seconds":wall,"weights_after":after,
                 "completed_unix":time.time(),"training_updates":0,
                 "scope":"Native protocol; not a custom MetaWorld baseline or sealed final test"}
        if wb:
            wb.log({"success_rate_percent":summary["success_rate_percent"],
                    "successes":summary["successes"],**timing})
            summary["wandb_url"]=wb.url
            wb.finish(); wb=None
        atomic_json(run_dir/"result.json",summary)
        atomic_json(run_dir/"status.json",{"status":"completed","result":"result.json",
            "protocol_hash":meta["protocol_hash"],"completed_unix":time.time()})
        return summary
    except Exception as exc:
        blocked=isinstance(exc,(ImportError,FileNotFoundError,importlib.metadata.PackageNotFoundError))
        atomic_json(run_dir/"status.json",{**meta,"status":"blocked" if blocked else "failed",
                    "error_type":type(exc).__name__,"error":str(exc),"ended_unix":time.time(),
                    "result_available":False})
        (run_dir/"failure.txt").write_text(traceback.format_exc())
        if wb:
            wb.finish(exit_code=1); wb=None
        raise
    finally:
        if world is not None:
            world.envs.close()
        if dataset is not None and getattr(dataset,"h5_file",None) is not None:
            dataset.h5_file.close()


def main():
    args=parser().parse_args()
    result=execute(args)
    print(json.dumps({"status":result["status"],"successes":result["successes"],
                      "evaluations":result["evaluations"]}))


if __name__=="__main__": main()
