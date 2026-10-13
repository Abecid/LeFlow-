"""Protocol tests use synthetic outcomes only inside pytest temporary directories."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from self_refine.protocol import (atomic_json, digest_json, paired_difference,
    select_cases, sha256_file, summarize_telemetry, validate_metrics)
from self_refine.campaign import build_jobs, map_gpu_lanes, run_bounded
from self_refine.core import Work
from self_refine.evaluate import execute, parser


@pytest.fixture
def case_inputs():
    return np.repeat(np.arange(4),20),np.tile(np.arange(20),4)


def test_case_sampling_matches_native_row_algorithm(case_inputs):
    eps,steps=case_inputs
    cases=select_cases(eps,steps,goal_offset=5,num_eval=10,seed=42)
    valid=np.nonzero(steps<=14)[0]
    positions=np.random.default_rng(42).choice(len(valid),size=10,replace=False)
    expected=np.sort(valid[positions]).tolist()
    assert [c["row"] for c in cases]==expected
    assert all(c["goal"]==c["start"]+5 for c in cases)


def test_heldout_selection_never_uses_train_episodes(case_inputs):
    eps,steps=case_inputs
    cases=select_cases(eps,steps,goal_offset=5,num_eval=10,seed=42,heldout={2,3})
    assert {c["episode"] for c in cases} <= {2,3}


def test_too_few_cases_fail_instead_of_resampling(case_inputs):
    with pytest.raises(ValueError):
        select_cases(*case_inputs,goal_offset=19,num_eval=5,seed=0)


def test_empty_heldout_fails(case_inputs):
    with pytest.raises(ValueError):
        select_cases(*case_inputs,goal_offset=5,num_eval=5,seed=0,heldout=set())


def test_seed_changes_cases_without_changing_count(case_inputs):
    a=select_cases(*case_inputs,goal_offset=5,num_eval=10,seed=1)
    b=select_cases(*case_inputs,goal_offset=5,num_eval=10,seed=2)
    assert a!=b and len(a)==len(b)==10


@pytest.fixture
def cases(case_inputs):
    return select_cases(*case_inputs,goal_offset=5,num_eval=4,seed=42)


def test_success_comes_from_physical_episode_successes(cases):
    rows=validate_metrics({"episode_successes":[0,1,1,0],"success_rate":50},cases)
    assert [r["success"] for r in rows]==[False,True,True,False]


@pytest.mark.parametrize("bad", [
    {"success_rate":90},
    {"episode_successes":[1,0],"success_rate":50},
    {"episode_successes":[1,0,0,0],"success_rate":.25},
    {"episode_successes":[1,0,0,0],"success_rate":float('nan')},
    {"episode_successes":[1,0,0,.5],"success_rate":37.5},
    {"episode_successes":[1,0,0,float('nan')],"success_rate":25},
])
def test_invalid_success_is_rejected(cases,bad):
    with pytest.raises(ValueError): validate_metrics(bad,cases)


def test_atomic_json_and_hashes(tmp_path):
    path=tmp_path/"result.json"
    atomic_json(path,{"a":np.array([1,2]),"v":np.int64(3)})
    assert json.loads(path.read_text())=={"a":[1,2],"v":3}
    a=sha256_file(path)
    atomic_json(path,{"a":[2,3]})
    assert a!=sha256_file(path)
    assert digest_json({"b":2,"a":1})==digest_json({"a":1,"b":2})
    with pytest.raises(ValueError): atomic_json(path,{"not_valid":float('nan')})


def run(cases,values):
    return {"status":"completed","benchmark_family":"native_leflow","protocol_hash":"same",
            "cases":validate_metrics({"episode_successes":values,"success_rate":100*np.mean(values)},cases)}


def test_paired_improvements_and_regressions(cases):
    a=run(cases,[0,1,1,0]); b=run(cases,[1,0,1,1])
    assert paired_difference(a,b)=={"cases":4,"wins":2,"regressions":1,"success_delta_pp":25.}


@pytest.mark.parametrize("field,value",[("status","failed"),("protocol_hash","different"),
                                      ("benchmark_family","custom_metaworld")])
def test_incompatible_results_are_not_mixed(cases,field,value):
    a=run(cases,[0,1,1,0]); b=copy.deepcopy(a); b[field]=value
    with pytest.raises(ValueError): paired_difference(a,b)


def test_case_identity_and_order_are_verified_even_with_same_hash(cases):
    a=run(cases,[0,1,1,0]); b=copy.deepcopy(a)
    b["cases"]=list(reversed(b["cases"]))
    with pytest.raises(ValueError): paired_difference(a,b)


def test_latency_is_not_divided_by_batch_size():
    records=[]
    for ms in [10.,20.,30.]:
        row={**Work(controller_ms=ms,flow_calls=2,flow_samples=8).as_dict(),"env_start":0,"env_end":4}
        records.append(row)
    out=summarize_telemetry(records)
    assert out["mean_batch_controller_ms"]==20.
    assert out["batch_sizes"]==[4] and out["flow_samples"]==24
    assert out["total_controller_seconds"]==.06


def test_no_telemetry_cannot_be_a_latency_result():
    with pytest.raises(ValueError): summarize_telemetry([])


@pytest.fixture
def campaign_config():
    root=Path(__file__).resolve().parents[2]
    # tests/self_refine/test_protocol.py -> repo root is parents[2]
    return json.loads((root/"config/self_refine/native_comparison.json").read_text())


def test_campaign_is_finite_and_training_free(campaign_config,tmp_path):
    jobs=build_jobs(campaign_config,tmp_path,{t:"existing/checkpoint" for t in campaign_config["tasks"]})
    assert len(jobs)==24 and len({j["id"] for j in jobs})==24
    assert all(j["task"]!="metaworld" for j in jobs)
    assert campaign_config["training_updates"]==0


def test_missing_checkpoint_mapping_fails(campaign_config,tmp_path):
    with pytest.raises(ValueError): build_jobs(campaign_config,tmp_path,{})


def test_duplicate_names_rejected(campaign_config,tmp_path):
    campaign_config["methods"].append(campaign_config["methods"][0])
    with pytest.raises(ValueError): build_jobs(campaign_config,tmp_path,{t:"x" for t in campaign_config["tasks"]})


def test_unbounded_job_rejected(campaign_config,tmp_path):
    campaign_config["job_timeout_seconds"]=0
    with pytest.raises(ValueError): build_jobs(campaign_config,tmp_path,{})


@pytest.mark.parametrize("selected,mask,count,expected",[
    (["0","1"],"2,5,6,7",4,["2","5"]),
    (["1"],"GPU-aaa,GPU-bbb",2,["GPU-bbb"]),
    (["2"],None,4,["2"]),
])
def test_gpu_selection_respects_existing_allocation(selected,mask,count,expected):
    assert map_gpu_lanes(selected,mask,count)==expected


@pytest.mark.parametrize("selected,mask,count",[
    (["0","0"],None,4),(["0","1","2","3","4"],None,8),
    (["2"],"0,3",2),(["1"],"GPU-aaa",2),(["-1"],None,4),
])
def test_invalid_gpu_allocation_rejected(selected,mask,count):
    with pytest.raises(ValueError): map_gpu_lanes(selected,mask,count)


def test_run_timeout_is_real_and_process_exits(tmp_path):
    with (tmp_path/"log").open("wb") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_bounded([sys.executable,"-c","import time;time.sleep(30)"],
                        cwd=tmp_path,env=None,log=log,timeout=.05)


def test_evaluation_missing_repository_creates_blocked_not_results(tmp_path):
    args=parser().parse_args(["--task","pusht","--checkpoint","missing.pt",
       "--method-config","missing.json","--run-dir",str(tmp_path/"run"),
       "--repo-root",str(tmp_path/"no-repo"),"--wandb-mode","disabled"])
    with pytest.raises(FileNotFoundError): execute(args)
    state=json.loads((tmp_path/"run/status.json").read_text())
    assert state["status"]=="blocked" and state["result_available"] is False
    assert not (tmp_path/"run/result.json").exists()


def test_completed_or_partial_run_is_never_overwritten(tmp_path):
    (tmp_path/"run").mkdir(); (tmp_path/"run/result.json").write_text("PRESERVE")
    args=parser().parse_args(["--task","pusht","--checkpoint","missing.pt",
       "--method-config","missing.json","--run-dir",str(tmp_path/"run")])
    with pytest.raises(FileExistsError): execute(args)
    assert (tmp_path/"run/result.json").read_text()=="PRESERVE"
