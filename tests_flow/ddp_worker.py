"""Run explicitly with torchrun --standalone --nproc_per_node=2; CPU collective gate."""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_jepa.models import System


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = int(os.environ["RANK"])
    c = json.loads(Path("config/flow_metaworld.json").read_text())
    c["encoder"].update(dim=8, token_grid=[1, 1, 2])
    c["model"].update(width=16, depth=1, heads=2, segments=2, chunk_steps=2)
    c["data"]["action_repeat"] = 1
    torch.manual_seed(42)
    world = System(c, "world").world.eval().requires_grad_(False)
    torch.nn.init.normal_(world.out.weight, std=0.1)
    for method in [
        "joint_flow_consistent",
        "joint_deterministic_consistent",
        "leflow_adapted",
        "hwm_adapted",
        "world",
    ]:
        model = DDP(System(c, method))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(100 + rank)
        for _ in range(2):
            batch = dict(
                z=torch.randn(2, 3, 2, 8),
                a=torch.randn(2, 2, 2, 4),
                local=torch.randn(2, 2, 2, 8),
            )
            if method in ("world", "hwm_adapted"):
                batch["a"] = batch["a"][:, 0]
            opt.zero_grad(set_to_none=True)
            loss, _ = model(
                batch,
                world,
                consistency_weight=0.1 if method.endswith("_consistent") else 0.0,
                consistency_steps=2,
            )
            loss.backward()
            opt.step()
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        all_params = [torch.empty_like(params) for _ in range(dist.get_world_size())]
        dist.all_gather(all_params, params)
        assert all(torch.equal(params, x) for x in all_params)
        if rank == 0:
            print(method, "two-rank parameters synchronized", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
