"""Released-checkpoint gates; call before any benchmark evaluation."""
from __future__ import annotations
import torch
from .core import RefinementConfig, Work, sample_interiors
from .native import NativeAdapter, state_fingerprint


def validate_native_runtime(runtime, *, device, horizon=5, steps=16, samples=4):
    runtime=runtime.eval().requires_grad_(False)
    adapter=NativeAdapter(runtime)
    before=state_fingerprint(runtime)
    d=runtime.flow.latent_dim
    random=torch.Generator(device=device).manual_seed(7613)
    start=torch.randn(1,d,device=device,generator=random)
    goal=torch.randn(1,d,device=device,generator=random)
    a=torch.Generator(device=device).manual_seed(113)
    b=torch.Generator(device=device).manual_seed(113)
    with torch.no_grad():
        expected=runtime.sample_paths(start,goal,horizon=horizon,num_samples=samples,
                                      flow_steps=steps,generator=a)
        clean,_=sample_interiors(runtime.flow,start,goal,horizon=horizon,
              num_samples=samples,flow_steps=steps,config=RefinementConfig(),generator=b)
        zs=start[:,None].expand(1,samples,d).reshape(samples,d)
        zg=goal[:,None].expand(1,samples,d).reshape(samples,d)
        actual=adapter.join(clean.reshape(samples,horizon-1,d),zs,zg).reshape_as(expected)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actions=runtime.decode_actions(expected)
        adapter_actions=adapter.decode(actual.reshape(samples,horizon+1,d),Work())
        torch.testing.assert_close(adapter_actions,actions.reshape_as(adapter_actions),rtol=0,atol=0)
        pred_expected=runtime.rollout_paths(start,actions)
        pred_actual=adapter.rollout(zs,adapter_actions,Work()).reshape_as(pred_expected)
        torch.testing.assert_close(pred_actual,pred_expected,rtol=1e-5,atol=1e-6)
    with torch.inference_mode():
        # Exercise the exact outer context used by native policy evaluation.
        _,work=sample_interiors(runtime.flow,start,goal,horizon=horizon,
            num_samples=samples,flow_steps=steps,
            config=RefinementConfig(mode="guided_pnp",indices=(steps//2,),rounds=1),
            generator=torch.Generator(device=device).manual_seed(113),
            perturb_generator=torch.Generator(device=device).manual_seed(114),energy=adapter.energy)
    if work.backward_calls<1 or state_fingerprint(runtime)!=before:
        raise RuntimeError("guidance/weight-immutability gate failed")
    if any(p.grad is not None or p.requires_grad for p in runtime.parameters()):
        raise RuntimeError("frozen inference unexpectedly produced parameter gradients")
    return {"native_sampler_bitwise_parity":True,"native_action_bitwise_parity":True,
            "native_world_rollout_close":True,"nested_inference_mode_gradient":True,
            "weights_and_buffers_unchanged":True,"state_sha256":before,
            "note":"Synthetic latent-input ABI gate on real weights, not task-success evidence"}
