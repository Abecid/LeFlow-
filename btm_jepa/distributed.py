"""One-node 1--4 GPU training, with Gloo CPU support for integration tests."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def initialize(device="cuda"):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if not 1 <= world_size <= 4:
        raise ValueError("This experiment supports at most four processes/GPUs")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use device=cpu only for tests")
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device(device)
    if world_size > 1:
        # Rank zero can spend several minutes doing actual environment evaluation.
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo", timeout=timedelta(hours=2))
    return rank, world_size, dev


def barrier():
    if dist.is_initialized():
        dist.barrier()


class EvalShard(Sampler):
    """No duplicated/padded validation samples, unlike DistributedSampler."""
    def __init__(self, size, rank, world_size):
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
