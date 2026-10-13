"""Actuator-box policy units and the frozen, released compact LeWM interface."""
from __future__ import annotations
import torch
from torch import nn


class ActionUnits(nn.Module):
    def __init__(self, low, high, mean, std, block=5):
        super().__init__()
        for key, value in dict(low=low, high=high, mean=mean, std=std).items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32).flatten())
        self.block = int(block)
        if not (self.low.shape == self.high.shape == self.mean.shape == self.std.shape):
            raise ValueError('Action bounds/statistics have different dimensions')
        if self.block < 1 or not all(torch.isfinite(v).all() for v in (self.low,self.high,self.mean,self.std)):
            raise ValueError('Invalid action normalization')
        if (self.high <= self.low).any() or (self.std <= 0).any():
            raise ValueError('Degenerate actuator range or action standard deviation')

    def _expand(self, value, x):
        if x.shape[-1] == len(value):
            return value
        if x.shape[-1] == len(value)*self.block:
            return value.repeat(self.block)
        raise ValueError(f'Expected primitive or block actions, got {tuple(x.shape)}')

    def physical(self, unit):
        low,high = self._expand(self.low,unit), self._expand(self.high,unit)
        return low + (unit.clamp(-1,1)+1)*.5*(high-low)

    def unit(self, physical, *, strict=False, clip=True):
        low,high = self._expand(self.low,physical), self._expand(self.high,physical)
        value = 2*(physical-low)/(high-low)-1
        if strict and (not torch.isfinite(value).all() or (value.abs() > 1.0001).any()):
            raise ValueError('Recorded actions exceed declared actuator bounds')
        return value.clamp(-1,1) if clip else value

    def to_world(self, unit):
        a = self.physical(unit)
        return (a-self._expand(self.mean,a))/self._expand(self.std,a)

    def from_world(self, normalized):
        a = normalized*self._expand(self.std,normalized)+self._expand(self.mean,normalized)
        return self.unit(a)


class LeWMWorld(nn.Module):
    """[B,P=1,D], [B,H,block*primitive_dim] -> [B,H,1,D].

    The world sees exactly the bounded actions sent to the simulator. Gradients
    with respect to actions remain available; model parameters are frozen.
    """
    def __init__(self, lewm, units, history_size=3):
        super().__init__()
        self.lewm = lewm.eval().requires_grad_(False)
        self.units = units
        self.history_size = int(history_size)
        self.calls = 0

    def train(self, mode=True):
        super().train(False)
        return self

    def to_world(self, unit):
        return self.units.to_world(unit)

    def from_world(self, normalized):
        return self.units.from_world(normalized)

    def forward(self, z, action):
        return self.rollout(z, action[:,None])[:,0]

    def rollout(self, z, actions, *, budget=None):
        if z.ndim != 3 or z.shape[1] != 1 or actions.ndim != 3 or len(z) != len(actions):
            raise ValueError('Expected [B,1,D] states and [B,H,A] actions')
        states = z[:,0,None]
        converted = self.to_world(actions)
        result=[]
        for k in range(actions.shape[1]):
            if budget is not None:
                budget.check()
            state_window = states[:,-self.history_size:]
            action_window = converted[:,:k+1][:,-self.history_size:]
            prediction = self.lewm.predict(state_window, self.lewm.action_encoder(action_window))[:,-1]
            self.calls += len(z)
            result.append(prediction[:,None])
            states = torch.cat((states,prediction[:,None]),dim=1)
        return torch.stack(result,dim=1)

    @torch.no_grad()
    def encode(self, rgb):
        # Same transform as the original cache and released model loader.
        from btm_jepa.data import image_transform
        device=next(self.lewm.parameters()).device
        x=torch.as_tensor(rgb,device=device)
        if x.ndim != 4 or x.shape[-1] != 3 or x.dtype != torch.uint8:
            raise ValueError('RGB must be uint8 [B,H,W,3]')
        x=image_transform()(x.permute(0,3,1,2))
        return self.lewm.encode({'pixels':x[:,None]})['emb'][:,-1,None].float()


def load_world(meta, device):
    from latent_planner import load_lewm
    from flow_jepa.common import file_hash
    if file_hash(meta['lewm_checkpoint']) != meta['checkpoint_sha256']:
        raise ValueError('Frozen world checksum mismatch')
    units=ActionUnits(meta['action_low'],meta['action_high'],meta['action_mean'],
                      meta['action_std'],meta['action_block'])
    world=LeWMWorld(load_lewm(meta['lewm_checkpoint']),units).to(device).eval()
    return world
