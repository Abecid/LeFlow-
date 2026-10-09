"""Combine original microbatches while retaining their RNG draws and examples.

The objective and global batch are unchanged. Larger FP32 matrix operations and
reduction order can differ in roundoff; this is not a bitwise-identity claim.
"""
import torch
from torch.nn import functional as F
from flow_jepa.models import System, prediction_loss


class FusedSystem(System):
    def forward(self, batch, world=None, *, consistency_weight=0.0,
                consistency_batch=2, consistency_steps=4, original_micro=4):
        if self.method != 'joint_flow_consistent':
            raise ValueError('Fused execution is only audited for joint_flow_consistent')
        z, a = batch['z'], batch['a']
        if len(z) % original_micro:
            raise ValueError('Fused batch must contain whole original microbatches')
        start, goal, target = z[:, 0], z[:, -1], z[:, 1:-1]
        ts, nzs, nas, gzs, gas, indices = [], [], [], [], [], []
        n = min(consistency_batch, original_micro)
        for lo in range(0, len(z), original_micro):
            hi = lo + original_micro
            # Same calls, shapes and ordering as each original System.forward.
            ts.append(torch.rand(original_micro, device=z.device))
            nzs.append(torch.randn_like(target[lo:hi]))
            nas.append(torch.randn_like(a[lo:hi]))
            if consistency_weight:
                gzs.append(torch.randn(n, *target.shape[1:], device=z.device))
                gas.append(torch.randn(n, *a.shape[1:], device=z.device))
                indices.extend(range(lo, lo + n))
        t, nz, na = torch.cat(ts), torch.cat(nzs), torch.cat(nas)
        xz = torch.lerp(nz, target, t[:, None, None, None])
        xa = torch.lerp(na, a, t[:, None, None, None])
        vz, va = self.planner(xz, xa, start, goal, t)
        endpoint = self.planner.state_parameterization == 'endpoint'
        zloss = F.mse_loss(vz, target if endpoint else target - nz)
        aloss = F.mse_loss(va, a - na)
        loss = zloss + aloss
        metrics = dict(state_objective=zloss.detach(), action_objective=aloss.detach())
        if consistency_weight:
            chosen = torch.tensor(indices, device=z.device)
            cs, cg = start[chosen], goal[chosen]
            gz, ga = self.planner.sample(cs, cg, steps=consistency_steps,
                                        noise=(torch.cat(gzs), torch.cat(gas)))
            path = torch.cat([cs[:, None], gz, cg[:, None]], 1)
            b, m, k, ad = ga.shape
            predicted = world.rollout(path[:, :-1].flatten(0, 1),
                                      ga.clamp(-1, 1).reshape(b*m, k, ad))[:, -1]
            reach = prediction_loss(predicted, path[:, 1:].flatten(0, 1))
            loss = loss + consistency_weight * reach
            metrics['generated_consistency'] = reach.detach()
        return loss, metrics
