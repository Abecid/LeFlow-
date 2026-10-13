"""Two-rank CPU/Gloo integration of the same trainer. No benchmark scores."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crossbench.spec import Spec, METHODS
from crossbench.train import run
from crossbench.world import LeWMWorld, ActionUnits
from flow_jepa.common import save_json
from test_crossbench_core import TinyLeWM
from test_crossbench_data_eval import make_cache


def evaluate_fixture(model,world,bank,s,root,out,sha,device,rank,size):
    metrics=dict(successes=0,episodes=1,success_rate=0.,timeouts=0)
    if rank==0:save_json(Path(out)/'report.json',dict(metrics=metrics,fixture=True))
    dist.barrier()
    return metrics


def main():
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(1);dist.init_process_group('gloo')
    value=[tempfile.mkdtemp(prefix='crossbench-ddp-') if rank==0 else None]
    dist.broadcast_object_list(value,0);root=Path(value[0])
    try:
        if rank==0:make_cache(root)
        dist.barrier()
        for method in METHODS:
            s=Spec('pusht',method,dim=16,width=32,updates=4,global_batch=4,eval_episodes=1)
            world=LeWMWorld(TinyLeWM(),ActionUnits([-1]*2,[1]*2,[0]*2,[1]*2))
            result=run(s,root,root/method,'cpu',rank,size,world=world,eval_fn=evaluate_fixture,
                       allow_fixture=True,online=False)
            assert result['step']==4 and result['validation_steps']==[1,2,3,4]
            if rank==0:print('DDP_VERIFIED',method,'ranks',size,flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if rank==0:shutil.rmtree(root)


if __name__=='__main__':main()
