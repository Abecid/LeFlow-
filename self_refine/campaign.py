"""Finite, restart-safe native comparison; <=4 GPUs, no training/replays by default."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time

from .core import RefinementConfig
from .protocol import atomic_json, digest_json, sha256_file


def build_jobs(config: dict, root: Path, checkpoint_map: dict) -> list[dict]:
    if config.get("schema_version")!=1 or config.get("training_updates")!=0:
        raise ValueError("campaign must declare schema_version=1 and training_updates=0")
    if not 1<=int(config["job_timeout_seconds"])<=3600:
        raise ValueError("each job must have a finite timeout of at most one hour")
    tasks=config["tasks"]; methods=config["methods"]
    if not tasks or len(set(tasks))!=len(tasks) or any(t not in {"tworoom","pusht","reacher","cube"} for t in tasks):
        raise ValueError("tasks must be unique native benchmark names")
    if len(set(m["name"] for m in methods))!=len(methods):
        raise ValueError("method names must be unique")
    if sum(m["name"]=="leflow" for m in methods)!=1:
        raise ValueError("include exactly one native leflow reference")
    jobs=[]
    for task in tasks:
        if task not in checkpoint_map:
            raise ValueError(f"no checkpoint specified for {task}")
        for seed in config["seeds"]:
            for method in methods:
                name=method["name"]
                if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
                    raise ValueError("unsafe method name")
                if method["num_samples"]<1:
                    raise ValueError("num_samples must be positive")
                RefinementConfig.from_dict(method["refinement"]).validate(method["flow_steps"])
                job_id=f"{task}-s{seed}-{name}"
                jobs.append({"id":job_id,"task":task,"seed":seed,"method":method,
                             "checkpoint":checkpoint_map[task],"run_dir":str(root/"jobs"/job_id)})
    return jobs


def map_gpu_lanes(gpus: list[str], visible_mask: str | None, device_count: int) -> list[str]:
    """Map requested logical indices without escaping an existing GPU allocation."""
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus) or any(not x.isdigit() for x in gpus):
        raise ValueError("provide one to four distinct nonnegative GPU indices")
    if any(int(g) >= device_count for g in gpus):
        raise ValueError("selected GPU is outside the visible allocation")
    if visible_mask is None:
        return gpus
    available = [v.strip() for v in visible_mask.split(",") if v.strip()]
    if any(int(g) >= len(available) for g in gpus):
        raise ValueError("CUDA_VISIBLE_DEVICES and device count disagree")
    return [available[int(g)] for g in gpus]


def run_bounded(cmd, *, cwd, env, log, timeout):
    """Terminate the entire job process group on timeout, not just its parent."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="config/self_refine/native_comparison.json")
    p.add_argument("--checkpoint-map",required=True,help="JSON mapping native task -> existing checkpoint")
    p.add_argument("--run-root",required=True)
    p.add_argument("--repo-root",default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--gpus",default="0",help="One process per listed GPU; maximum four")
    p.add_argument("--cache-dir")
    p.add_argument("--wandb-mode",choices=["online","offline","disabled"],default="online")
    p.add_argument("--execute",action="store_true",help="Absent: print a plan only; launch nothing")
    p.add_argument("--resume",action="store_true",help="Reuse only completed identical jobs; never replay failed evaluations")
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    cmap=json.loads(Path(args.checkpoint_map).read_text())
    root=Path(args.run_root).resolve()
    jobs=build_jobs(cfg,root,cmap)
    gpus=args.gpus.split(",")
    if not 1<=len(gpus)<=4 or len(set(gpus))!=len(gpus) or any(not x.isdigit() for x in gpus):
        p.error("provide one to four distinct nonnegative GPU indices")
    manifest={"config":cfg,"checkpoints":cmap,"jobs":jobs,"repo_root":str(Path(args.repo_root).resolve()),
              "cache_dir":args.cache_dir,"gpus":gpus,"wandb_mode":args.wandb_mode}
    if not args.execute:
        print(json.dumps({"status":"planned_not_started","jobs":len(jobs),"gpus":gpus,
                          "training_updates":0,"manifest":manifest},indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("campaign not started: this environment has no CUDA")
    gpu_lanes=map_gpu_lanes(gpus,os.environ.get("CUDA_VISIBLE_DEVICES"),torch.cuda.device_count())
    sys.path.insert(0,str(Path(args.repo_root).resolve()))
    from latent_planner import _resolve_checkpoint
    resolved={task:str(_resolve_checkpoint(Path(value).expanduser()).resolve()) for task,value in cmap.items()}
    source_root=Path(args.repo_root).resolve()
    source_files=list((source_root/"self_refine").glob("*.py"))
    source_files+=list((source_root/"config/eval").rglob("*.yaml"))
    source_files += [source_root/n for n in ["latent_planner.py","eval.py","jepa.py","module.py","btm_jepa/env.py"]]
    manifest.update({"resolved_checkpoints":resolved,
        "checkpoint_hashes":{t:sha256_file(v) for t,v in resolved.items()},
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=args.repo_root,text=True).strip(),
        "source_hashes":{str(f.relative_to(source_root)):sha256_file(f) for f in sorted(source_files)}})
    jobs=build_jobs(cfg,root,resolved)
    manifest["jobs"]=jobs
    if root.exists() and not args.resume:
        raise FileExistsError("run root already exists; use --resume for completed jobs or a NEW run root")
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/"campaign.lock").open("a")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("another coordinator owns this campaign")
    manifest_hash=digest_json(manifest)
    old=root/"campaign.json"
    if old.exists() and json.loads(old.read_text())["manifest_hash"]!=manifest_hash:
        raise ValueError("resume manifest differs; cannot mix checkpoints/methods/protocols")
    atomic_json(old,{"manifest_hash":manifest_hash,"manifest":manifest})
    (root/"logs").mkdir(exist_ok=True); (root/"methods").mkdir(exist_ok=True)
    for method in cfg["methods"]:
        atomic_json(root/"methods"/(method["name"]+".json"),method)
    stop=threading.Event()
    record_lock=threading.Lock(); outcomes=[]

    def lane(index):
        for job in jobs[index::len(gpus)]:
            if stop.is_set(): return
            job_dir=Path(job["run_dir"])
            if job_dir.exists():
                result=job_dir/"result.json"
                if result.exists() and json.loads(result.read_text()).get("status")=="completed":
                    outcome={"id":job["id"],"status":"reused_completed"}
                else:
                    outcome={"id":job["id"],"status":"blocked_existing_incomplete_no_replay"}
                    stop.set()
            else:
                cmd=[sys.executable,"-m","self_refine.evaluate","--task",job["task"],
                     "--checkpoint",job["checkpoint"],"--method-config",str(root/"methods"/(job["method"]["name"]+".json")),
                     "--run-dir",str(job_dir),"--repo-root",args.repo_root,"--seed",str(job["seed"]),
                     "--policy-seed",str(cfg["policy_seed"]),"--num-eval",str(cfg["num_eval"]),
                     "--batch-size",str(cfg["batch_size"]),"--device","cuda:0","--wandb-mode",args.wandb_mode]
                if args.cache_dir: cmd += ["--cache-dir",args.cache_dir]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_lanes[index],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
                start=time.time()
                with (root/"logs"/(job["id"]+".log")).open("ab") as log:
                    try:
                        rc=run_bounded(cmd,cwd=args.repo_root,env=env,log=log,timeout=cfg["job_timeout_seconds"])
                        result_file=job_dir/"result.json"
                        complete=(rc==0 and result_file.exists() and
                                  json.loads(result_file.read_text()).get("status")=="completed")
                        outcome={"id":job["id"],"status":"completed" if complete else "failed","returncode":rc}
                    except subprocess.TimeoutExpired:
                        outcome={"id":job["id"],"status":"job_timeout","returncode":None}
                        atomic_json(job_dir/"status.json",{**outcome,"result_available":False})
                outcome.update({"start_unix":start,"end_unix":time.time(),"gpu":gpus[index]})
                if outcome["status"]!="completed": stop.set()
            with record_lock:
                outcomes.append(outcome)
                atomic_json(root/"progress.json",{"outcomes":outcomes,"planned":len(jobs),
                    "unstarted":len(jobs)-len(outcomes),"halted_after_failure":stop.is_set()})
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as workers:
            futures=[workers.submit(lane,i) for i in range(len(gpus))]
            for future in futures: future.result()
        if stop.is_set():
            raise SystemExit("campaign halted after a failure; completed results preserved, no failed job replayed")
        print(json.dumps({"status":"completed","jobs":len(outcomes)}))
    finally:
        lock.close()


if __name__=="__main__": main()
