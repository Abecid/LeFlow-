"""Small real-checkpoint/real-simulator interface test; no benchmark outcomes.

Uses one rendered image and five imagined action blocks per dataset. No
training dataset or policy checkpoint is downloaded and no optimizer runs.
"""
import os
from pathlib import Path
import sys
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download
from latent_planner import load_lewm
from crossbench.environment import NativeEnv
from crossbench.world import ActionUnits, LeWMWorld


def main():
    torch.set_num_threads(1)
    for task in ('pusht','cube'):
        checkpoint=hf_hub_download('quentinll/lewm-'+task,'weights.pt')
        model=load_lewm(checkpoint).eval().requires_grad_(False)
        env=NativeEnv(task)
        try:
            env.env.action_space.seed(3072);env.env.reset(seed=3072)
            image=env.render();low=env.action_space.low;high=env.action_space.high
            pd=len(low);ad=model.action_encoder.patch_embed.weight.shape[1]
            assert ad==5*pd,(task,ad,pd)
            # Synthetic action statistics for an interface-only probe. Formal
            # comparisons instead use verified statistics from source.json.
            units=ActionUnits(low,high,(low+high)/2,(high-low)/2,block=5)
            world=LeWMWorld(model,units)
            z=world.encode(image[None]);assert z.shape[0:2]==(1,1)
            actions=torch.zeros(1,5,ad,requires_grad=True)
            pred=world.rollout(z,actions)
            assert pred.shape==(1,5,1,z.shape[-1]) and torch.isfinite(pred).all()
            grad=torch.autograd.grad(pred.square().mean(),actions)[0]
            assert torch.isfinite(grad).all() and grad.abs().sum()>0
            assert all(p.grad is None for p in world.parameters())
            print('RELEASED_MODEL_VERIFIED',task,'latent_dim',z.shape[-1],
                  'primitive_dim',pd,'block_dim',ad,'action_gradient_norm',float(grad.norm()),flush=True)
        finally:env.close()


if __name__=='__main__':main()
