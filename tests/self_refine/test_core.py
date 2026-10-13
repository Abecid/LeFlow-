"""CPU correctness tests; synthetic dynamics are NOT benchmark evaluations."""
from dataclasses import replace
import math
import sys
from types import SimpleNamespace, ModuleType

import pytest
import torch
from torch import nn

from self_refine.core import RefinementConfig, Work, correct_clean, sample_interiors
from self_refine.native import NativeAdapter, PlannerEngine, state_fingerprint
from self_refine.solver import SelfRefiningLatentPathSolver


class TinyFlow(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x, t, start, goal):
        return self.linear(x).tanh() + .1 * (goal - start)[:, None] + t[:, None, None] * .03


class TinyInverse(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.net = nn.Linear(3 * d, a)

    def forward(self, start, goal):
        return self.net(torch.cat([start, goal, goal - start], -1))


class TinyWorld(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.action_encoder = nn.Linear(a, d)
        self.predictor = nn.Linear(d, d)

    def predict(self, emb, act):
        # Deliberately history-sensitive, exercising the native sliding window.
        return self.predictor(emb) + .2 * act + .05 * emb.mean(1, keepdim=True)


class TinyRuntime(nn.Module):
    def __init__(self, d=4, a=6):
        super().__init__()
        self.flow, self.inverse_dynamics = TinyFlow(d), TinyInverse(d, a)
        self.lewm = TinyWorld(d, a)
        self.action_block = 2
        self.register_buffer("persistent_buffer", torch.tensor([7.]))

    def encode_current_and_goal(self, info):
        return info["pixels"][:, -1], info["goal"][:, -1]


@pytest.fixture
def setup():
    torch.manual_seed(20)
    runtime = TinyRuntime()
    return runtime, torch.randn(2, 4), torch.randn(2, 4)


def gen(seed=31):
    return torch.Generator().manual_seed(seed)


def args(start, goal, config, **kw):
    return dict(start=start, goal=goal, horizon=5, num_samples=3, flow_steps=8,
                config=config, generator=gen(), perturb_generator=gen(71), **kw)


def reference_euler(flow, start, goal, n, horizon, steps, generator):
    """Independent native sample_paths arithmetic from inspected upstream."""
    b, d = start.shape
    zs = start[:, None].expand(b, n, d).reshape(b * n, d)
    zg = goal[:, None].expand(b, n, d).reshape(b * n, d)
    x = torch.randn(b * n, horizon - 1, d, generator=generator)
    dt = 1. / steps
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((b * n,), i * dt, dtype=start.dtype)
            x = x + flow(x, t, zs, zg) * dt
    return x.reshape(b, n, horizon - 1, d)


def reference_rollout(runtime, start, actions, history=3):
    z = start[:, None].clone()
    for i in range(actions.shape[1]):
        a = actions[:, max(0, i-history+1):i+1]
        pred = runtime.lewm.predict(z[:, -a.size(1):], runtime.lewm.action_encoder(a))[:, -1:]
        z = torch.cat([z, pred], 1)
    return z


@pytest.mark.parametrize("mode", ["native", "pnp", "guidance", "guided_pnp"])
def test_zero_refinement_bitwise_matches_native(setup, mode):
    r, start, goal = setup
    cfg = RefinementConfig(mode=mode, rounds=0)
    out, work = sample_interiors(r.flow, **args(start, goal, cfg))
    ref = reference_euler(r.flow, start, goal, 3, 5, 8, gen())
    assert torch.equal(out, ref)
    assert (work.flow_calls, work.flow_samples, work.world_calls) == (8, 48, 0)


@pytest.mark.parametrize("mode", ["pnp", "guided_pnp"])
def test_pnp_exact_extra_nfe(setup, mode):
    r, start, goal = setup
    adapter = NativeAdapter(r)
    cfg = RefinementConfig(mode=mode, indices=(2,4,6), rounds=2, guidance_batch_size=4)
    _, work = sample_interiors(r.flow, **args(start, goal, cfg, energy=adapter.energy))
    assert work.flow_calls == 8 + 3 * 2
    assert work.flow_samples == 6 * 14
    assert work.refinement_rounds == 6
    if mode == "guided_pnp":
        assert work.backward_calls == 6 * math.ceil(6 / 4)
        assert work.world_transitions == 6 * 6 * 5


def test_pnp_changes_path_but_not_candidate_rng_stream(setup):
    r, start, goal = setup
    a, b = gen(), gen()
    base, _ = sample_interiors(r.flow, **{**args(start, goal, RefinementConfig()), "generator": a})
    cfg = RefinementConfig(mode="pnp", indices=(2,4), rounds=2)
    refined, _ = sample_interiors(r.flow, **{**args(start, goal, cfg), "generator": b})
    assert not torch.equal(base, refined)
    assert torch.equal(a.get_state(), b.get_state())


def test_pnp_reproducible_and_perturb_seed_matters(setup):
    r, start, goal = setup
    cfg = RefinementConfig(mode="pnp", indices=(2,4))
    x, _ = sample_interiors(r.flow, **args(start, goal, cfg))
    y, _ = sample_interiors(r.flow, **args(start, goal, cfg))
    z, _ = sample_interiors(r.flow, **{**args(start, goal, cfg), "perturb_generator": gen(72)})
    assert torch.equal(x, y)
    assert not torch.equal(x, z)


def test_guidance_only_no_extra_flow_calls(setup):
    r, start, goal = setup
    adapter = NativeAdapter(r)
    cfg = RefinementConfig(mode="guidance", indices=(2,4), rounds=2)
    _, w = sample_interiors(r.flow, **args(start, goal, cfg, energy=adapter.energy))
    assert w.flow_calls == 8 and w.backward_calls == 4


def test_bounded_descent_and_microbatch_equivalence(setup):
    _, start, goal = setup
    clean = torch.full((2,4,4), 2.)
    def energy(y, *unused):
        return y.square().mean((-1,-2))
    cfg = RefinementConfig(radius=.1, guidance_batch_size=1)
    small = correct_clean(clean, start, goal, energy=energy, config=cfg, work=Work())
    big = correct_clean(clean, start, goal, energy=energy, config=replace(cfg, guidance_batch_size=8), work=Work())
    assert torch.equal(small, big)
    assert torch.all((small-clean).norm(dim=-1) <= .1000001)
    assert small.square().sum() < clean.square().sum()


def test_zero_radius_does_not_call_world_or_backpropagate(setup):
    r, start, goal = setup
    cfg = RefinementConfig(mode="guided_pnp", indices=(2,), radius=0.)
    _, w = sample_interiors(r.flow, **args(start, goal, cfg))
    assert w.backward_calls == w.world_calls == 0


def test_zero_gradient_is_finite(setup):
    _, start, goal = setup
    y = torch.randn(2,4,4)
    def energy(y, *unused):
        return (y * 0).sum((-1,-2))
    out = correct_clean(y, start, goal, energy=energy, config=RefinementConfig(), work=Work())
    assert torch.equal(out, y)


def test_guidance_survives_outer_inference_mode_and_freezes_weights(setup):
    r, start, goal = setup
    adapter = NativeAdapter(r)
    before = state_fingerprint(r)
    cfg = RefinementConfig(mode="guided_pnp", indices=(2,4), guidance_batch_size=2)
    with torch.inference_mode():
        xs, xg = start.clone(), goal.clone()
        result, w = sample_interiors(r.flow, **args(xs, xg, cfg, energy=adapter.energy))
    assert torch.isfinite(result).all() and w.backward_calls > 0
    assert state_fingerprint(r) == before
    assert not any(p.requires_grad or p.grad is not None for p in r.parameters())


@pytest.mark.parametrize("history", [1,2,3,7])
def test_native_rollout_value_and_gradient_parity(setup, history):
    r, start, goal = setup
    adapter = NativeAdapter(r, history)
    a = torch.randn(2,5,6, requires_grad=True)
    w = Work()
    actual = adapter.rollout(start, a, w)
    expected = reference_rollout(r, start, a, history)
    assert torch.equal(actual, expected)
    ga, = torch.autograd.grad(actual.square().sum(), a, retain_graph=True)
    ge, = torch.autograd.grad(expected.square().sum(), a)
    assert torch.equal(ga, ge)
    assert w.world_calls == 5 and w.world_transitions == 10


def test_native_decode_and_energy_formula(setup):
    r, start, goal = setup
    ad = NativeAdapter(r)
    clean = torch.randn(2,4,4, requires_grad=True)
    w = Work()
    paths = ad.join(clean, start, goal)
    assert torch.equal(paths[:,0], start) and torch.equal(paths[:,-1], goal)
    decoded = ad.decode(paths, w)
    ref = r.inverse_dynamics(paths[:,:-1].reshape(-1,4), paths[:,1:].reshape(-1,4)).reshape(2,5,6)
    assert torch.equal(decoded, ref)
    predicted = reference_rollout(r, start, decoded)
    expected = (predicted[:,-1]-goal).square().mean(-1) + .1*(predicted[:,1:-1]-clean).square().mean((-1,-2))
    actual = ad.energy(clean, start, goal, .1, Work())
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, torch.zeros_like(actual))  # no clamped-goal scoring
    grad, = torch.autograd.grad(actual.sum(), clean)
    assert grad.abs().sum() > 0


def test_engine_native_action_and_score_parity(setup):
    r, start, goal = setup
    engine = PlannerEngine(r, RefinementConfig())
    clean = reference_euler(r.flow, start, goal, 3, 5, 8, gen())
    zs = start[:,None].expand(2,3,4).reshape(6,4)
    zg = goal[:,None].expand(2,3,4).reshape(6,4)
    paths = torch.cat([zs[:,None], clean.reshape(6,4,4), zg[:,None]], 1)
    actions = r.inverse_dynamics(paths[:,:-1].reshape(-1,4), paths[:,1:].reshape(-1,4)).reshape(6,5,6)
    costs = (reference_rollout(r,zs,actions)[:,-1]-zg).square().mean(-1).reshape(2,3)
    best = costs.argmin(1)
    with torch.inference_mode():
        out = engine.plan({"pixels":start[:,None],"goal":goal[:,None]}, horizon=5,
                          num_samples=3,flow_steps=8,generator=gen(),perturb_generator=gen(71))
    assert torch.equal(out["actions"], actions.reshape(2,3,5,6)[torch.arange(2),best])
    assert torch.equal(out["all_costs"], costs)
    assert out["work"]["world_transitions"] == 30
    assert out["work"]["controller_ms"] > 0


@pytest.mark.parametrize("mode", ["native","pnp","guidance","guided_pnp"])
def test_engine_end_to_end_valid_shapes_and_accounting(setup, mode):
    r, start, goal = setup
    cfg = RefinementConfig(mode=mode, indices=(2,4), rounds=1, guidance_batch_size=2)
    engine = PlannerEngine(r,cfg)
    before = state_fingerprint(r)
    with torch.inference_mode():
        out = engine.plan({"pixels":start[:,None],"goal":goal[:,None]}, horizon=5,
                          num_samples=3,flow_steps=8,generator=gen(),perturb_generator=gen(71))
    assert out["actions"].shape == (2,5,6)
    assert out["all_costs"].shape == (2,3)
    assert not out["actions"].requires_grad
    assert before == state_fingerprint(r)
    if mode in {"guidance","guided_pnp"}:
        assert out["work"]["world_transitions"] == 30 + 2 * 30
        assert out["work"]["backward_calls"] == 6


@pytest.mark.parametrize("bad", [
    {"mode":"other"}, {"rounds":-1}, {"rounds":1.5}, {"radius":-1.},
    {"radius":float('nan')}, {"consistency_weight":float('inf')},
    {"guidance_batch_size":0}, {"mode":"pnp","indices":()},
    {"mode":"pnp","indices":(0,)}, {"mode":"pnp","indices":(8,)},
    {"mode":"pnp","indices":(2,2)}, {"mode":"pnp","indices":(1.5,)},
])
def test_invalid_config_fails(bad):
    with pytest.raises(ValueError):
        RefinementConfig(**bad).validate(8)


@pytest.mark.parametrize("bad_steps", [0,-1,1.5,True])
def test_bad_step_count_fails(bad_steps):
    with pytest.raises(ValueError):
        RefinementConfig().validate(bad_steps)


def test_separate_perturb_stream_required(setup):
    r,s,g = setup
    rng = gen()
    with pytest.raises(ValueError, match="separate"):
        sample_interiors(r.flow, **{**args(s,g,RefinementConfig(mode="pnp",indices=(2,))),
                                   "generator":rng,"perturb_generator":rng})


def test_guidance_cannot_silently_drop_missing_energy(setup):
    r,s,g = setup
    with pytest.raises(ValueError, match="energy"):
        sample_interiors(r.flow, **args(s,g,RefinementConfig(mode="guidance",indices=(2,))))


def test_nonfinite_velocity_fails(setup):
    _,s,g = setup
    def broken(x,*args): return x * float('nan')
    with pytest.raises(FloatingPointError):
        sample_interiors(broken,**args(s,g,RefinementConfig()))


def test_wrong_velocity_shape_fails(setup):
    _,s,g = setup
    def broken(x,*args): return x[...,0]
    with pytest.raises(ValueError, match="shape"):
        sample_interiors(broken,**args(s,g,RefinementConfig()))


def test_horizon_one_fixed_endpoints_no_flow(setup):
    r,s,g=setup
    out,w=sample_interiors(r.flow,**{**args(s,g,RefinementConfig()),"horizon":1})
    assert out.shape==(2,3,0,4) and w.flow_calls==0


def test_guidance_gradient_matches_finite_difference(setup):
    r,s,g=setup
    ad=NativeAdapter(r)
    r.double(); s=s.double(); g=g.double()
    x=torch.randn(2,4,4,dtype=torch.float64,requires_grad=True)
    gradient,=torch.autograd.grad(ad.energy(x,s,g,.1,Work()).sum(),x)
    step=1e-5
    plus=x.detach().clone(); minus=x.detach().clone()
    plus[0,2,1]+=step; minus[0,2,1]-=step
    numeric=(ad.energy(plus,s,g,.1,Work()).sum()-ad.energy(minus,s,g,.1,Work()).sum())/(2*step)
    assert torch.allclose(gradient[0,2,1],numeric,atol=1e-7,rtol=1e-5)


def test_disconnected_energy_raises_instead_of_fake_guidance(setup):
    _,s,g=setup
    def broken(y,*args): return y.new_zeros(len(y))
    with pytest.raises(RuntimeError):
        correct_clean(torch.randn(2,4,4),s,g,energy=broken,config=RefinementConfig(),work=Work())


def test_solver_telemetry_and_action_interface(setup,tmp_path,monkeypatch):
    r,s,g=setup
    class Box:
        shape=(2,3)
    spaces=ModuleType("gymnasium.spaces"); spaces.Box=Box
    monkeypatch.setitem(sys.modules,"gymnasium.spaces",spaces)
    solver=SelfRefiningLatentPathSolver("not-a-checkpoint",runtime=r,device="cpu",
        refinement={"mode":"guided_pnp","indices":[2],"rounds":1},
        num_samples=3,flow_steps=8,telemetry_path=str(tmp_path/"log.jsonl"))
    solver.configure(action_space=Box(),n_envs=2,
                     config=SimpleNamespace(horizon=5,action_block=2))
    with torch.inference_mode():
        out=solver.solve({"pixels":s[:,None],"goal":g[:,None]})
    assert out["actions"].shape==(2,5,6) and solver.action_dim==6
    assert len(out["costs"])==2 and len(solver.telemetry)==2
    assert len((tmp_path/"log.jsonl").read_text().splitlines())==2
    assert out["rollout_count"]==12


def test_fingerprint_tracks_buffer_changes(setup):
    r,_,_=setup
    a=state_fingerprint(r)
    r.persistent_buffer+=1
    assert state_fingerprint(r)!=a
