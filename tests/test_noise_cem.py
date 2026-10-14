"""CPU numerical/integration checks, NOT robot task-success benchmarks."""
from dataclasses import replace
import math
from unittest.mock import patch

import pytest
import torch

from leflow_refinement.core import PlannerOps, RefinementConfig, sample, select
from leflow_refinement.noise_search import (NoiseSearchConfig, fit_elites,
                                          noise_variants, plan, search, shell_project)


def fixture(ad=10, *, check=None):
    """Small frozen parameterized flow, bounded decoder and causal toy rollout."""
    gain = torch.nn.Parameter(torch.tensor(.03), requires_grad=False)
    def flow(x, t, start, goal):
        return -gain * x + .01 * (goal-start)[:, None]
    def decode(paths):
        indices = torch.arange(ad) % paths.shape[-1]
        return paths[:, 1:, indices].clamp(-1, 1)
    def rollout(start, actions):
        d = start.shape[-1]
        steps = start[:, None] + actions[..., :d].cumsum(1) / actions.shape[1]
        return torch.cat((start[:, None], steps), 1)
    return PlannerOps(flow, decode, rollout, check), gain


def endpoints(b=2, d=3):
    return torch.zeros(b, d), torch.linspace(-.4, .4, b*d).reshape(b, d)


def rng(seed=123):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize('ad', [10, 25])
@pytest.mark.parametrize('b', [1, 3])
def test_one_round_exact_native_euler(ad, b):
    s, g = endpoints(b)
    noise = torch.randn(b, 9, 4, 3, generator=rng())
    ops, _ = fixture(ad)
    cfg = NoiseSearchConfig(samples=9, elites=3, rounds=1, steps=4)
    actual = search(s, g, ops, cfg, initial_noise=noise)
    ref_ops, _ = fixture(ad)
    paths = sample(s, g, ref_ops, RefinementConfig(mode='euler', samples=9, steps=4), initial_noise=noise)
    expected = select(paths, ref_ops)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key
    assert ops.compute.as_dict() == ref_ops.compute.as_dict()
    assert actual['actions'].shape == (b, 5, ad)
    assert torch.equal(paths[:, :, 0], s[:, None].expand(-1, 9, -1))
    assert torch.equal(paths[:, :, -1], g[:, None].expand(-1, 9, -1))


@pytest.mark.parametrize('mode', ['noise_cem', 'noise_random'])
@pytest.mark.parametrize('projected', [False, True])
def test_counts_incumbent_and_frozen_parameters(mode, projected):
    s, g = endpoints()
    cfg = NoiseSearchConfig(mode=mode, samples=16, elites=4, steps=3, project_shell=projected)
    ops, gain = fixture()
    before = gain.detach().clone()
    with torch.inference_mode():
        out = search(s, g, ops, cfg, generator=rng())
    c = ops.compute
    assert c.flow_candidate_evaluations == 2 * 3 * 16 * 3
    assert c.inverse_candidate_transitions == c.world_candidate_transitions == 2 * 3 * 16 * 5
    assert c.flow_batches == 9 and c.world_batches == 15 and c.inverse_batches == 3
    assert c.backward_batches == 0 and c.correction_updates == 0
    assert torch.equal(before, gain) and gain.grad is None
    assert not out['actions'].requires_grad and not out['costs'].requires_grad
    assert out['all_costs'].shape == (2, 48)
    assert torch.equal(out['costs'], out['all_costs'].min(1).values)
    assert torch.equal(out['selected_indices'], out['all_costs'].argmin(1))
    history = torch.tensor(out['noise_search']['incumbent_costs'])
    assert (history[:, 1:] <= history[:, :-1]).all()
    assert out['actions'].abs().max() <= 1
    if projected:
        assert torch.allclose(out['selected_noise'].flatten(1).norm(dim=-1), torch.full((2,), math.sqrt(12)))


def test_elite_moments_are_batch_local_and_smoothed():
    raw = torch.tensor([[[[1., 3.]], [[5., 7.]], [[9., 11.]]],
                        [[[-10., -6.]], [[2., 4.]], [[6., 8.]]]])
    costs = torch.tensor([[2., 1., 3.], [0., 2., 1.]])
    mean, std = torch.zeros(2, 1, 1, 2), torch.ones(2, 1, 1, 2)
    cfg = NoiseSearchConfig(samples=3, elites=2, smoothing=.25)
    new_mean, new_std = fit_elites(raw, costs, mean, std, cfg)
    expected_mean = torch.tensor([[[[3., 5.]]], [[[-2., 1.]]]]) * .75
    expected_var = torch.tensor([[[[4., 4.]]], [[[64., 49.]]]]) * .75 + .25
    torch.testing.assert_close(new_mean, expected_mean)
    torch.testing.assert_close(new_std.square(), expected_var)


def test_population_variance_single_elite_has_positive_floor():
    raw = torch.ones(2, 3, 1, 2)
    cfg = NoiseSearchConfig(samples=3, elites=1, smoothing=0., min_std=.2)
    mean, std = fit_elites(raw, torch.zeros(2, 3), torch.zeros(2, 1, 1, 2),
                           torch.ones(2, 1, 1, 2), cfg)
    assert torch.equal(mean, torch.ones_like(mean))
    assert torch.equal(std, torch.full_like(std, .2))


def test_projected_fit_uses_raw_proposals():
    s, g = endpoints(1)
    initial = torch.arange(1., 49.).reshape(1, 4, 4, 3)
    cfg = NoiseSearchConfig(samples=4, elites=2, rounds=2, steps=2, project_shell=True)
    ops, _ = fixture()
    with patch('leflow_refinement.noise_search.fit_elites', wraps=fit_elites) as fit:
        search(s, g, ops, cfg, initial_noise=initial, generator=rng())
    assert fit.call_count == 1
    assert torch.equal(fit.call_args.args[0], initial)
    assert not torch.allclose(initial.flatten(2).norm(-1), torch.tensor(math.sqrt(12)))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_shell_projects_whole_interior_not_each_token(dtype):
    raw = torch.arange(1, 49, dtype=dtype).reshape(2, 2, 4, 3)
    projected, unprojected, count = shell_project(raw)
    assert count == 0 and torch.equal(unprojected, raw)
    torch.testing.assert_close(projected.flatten(2).norm(dim=-1), torch.full((2, 2), math.sqrt(12), dtype=dtype))
    assert projected[0, 0, 0].norm() != projected[0, 0, -1].norm()
    torch.testing.assert_close(projected / raw, (projected / raw)[..., :1, :1].expand_as(raw))


def test_shell_zero_resampling_and_no_input_mutation():
    raw = torch.ones(1, 3, 4, 3); raw[:, 1] = 0
    before = raw.clone()
    projected, updated, count = shell_project(raw, generator=rng())
    assert count == 1 and updated[0, 1].abs().sum() > 0
    assert torch.equal(before, raw)
    assert torch.isfinite(projected).all()
    torch.testing.assert_close(projected.flatten(2).norm(dim=-1), torch.full((1, 3), math.sqrt(12)))


def test_shell_persistent_zero_failure():
    with patch('torch.randn', side_effect=lambda shape, **kw: torch.zeros(shape, **{k:v for k,v in kw.items() if k!='generator'})):
        with pytest.raises(FloatingPointError, match='nonzero'):
            shell_project(torch.zeros(1, 2, 4, 3))


def test_shell_avoids_overflow_from_finite_large_input():
    projected, _, _ = shell_project(torch.full((1, 2, 4, 3), 1e30))
    assert torch.isfinite(projected).all()
    torch.testing.assert_close(projected.flatten(2).norm(dim=-1), torch.full((1, 2), math.sqrt(12)))


def test_deterministic_no_rng_or_input_mutation_with_explicit_generator():
    s, g = endpoints()
    before_s, before_g = s.clone(), g.clone()
    global_rng = torch.get_rng_state().clone()
    cfg = NoiseSearchConfig(samples=8, elites=2, steps=2, project_shell=True)
    a = search(s, g, fixture()[0], cfg, generator=rng())
    b = search(s, g, fixture()[0], cfg, generator=rng())
    for key in ('actions', 'all_costs', 'selected_indices', 'selected_noise'):
        assert torch.equal(a[key], b[key])
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(s, before_s) and torch.equal(g, before_g)


def test_exact_ties_keep_first_evaluated_candidate():
    s, g = endpoints(1)
    ops, _ = fixture()
    ops.rollout_fn = lambda start, actions: torch.cat((start[:, None], g.expand(len(start), 5, -1)), 1)
    initial = torch.randn(1, 8, 4, 3, generator=rng())
    out = search(s, g, ops, NoiseSearchConfig(samples=8, elites=2, steps=2), initial_noise=initial, generator=rng(44))
    assert out['selected_indices'].item() == 0
    assert torch.equal(out['selected_noise'], initial[:, 0])


def test_noise_search_resets_distribution_each_call():
    s, g = endpoints(1)
    cfg = NoiseSearchConfig(samples=8, elites=2, steps=2)
    ops, _ = fixture()
    a = search(s, g, ops, cfg, generator=rng())
    b = search(s, g, ops, cfg, generator=rng())
    assert torch.equal(a['all_costs'], b['all_costs'])


def test_select_uses_actual_predicted_terminal_not_clamped_goal():
    s, g = endpoints(1)
    ops, _ = fixture()
    out = search(s, g, ops, NoiseSearchConfig(samples=8, elites=2, rounds=1, steps=2), generator=rng())
    assert out['all_costs'].std() > 0 and out['all_costs'].max() > 0


def test_no_hidden_incumbent_rescore_or_action_average():
    s, g = endpoints(1)
    ops, _ = fixture()
    saved = []
    decoder = ops.decode_fn
    def record(paths):
        actions = decoder(paths); saved.append(actions.clone()); return actions
    ops.decode_fn = record
    out = search(s, g, ops, NoiseSearchConfig(samples=8, elites=2, steps=2), generator=rng())
    assert len(saved) == 3
    all_actions = torch.cat(saved)
    assert torch.equal(out['actions'][0], all_actions[out['selected_indices'].item()])


def test_independent_chunked_control_has_same_counts_without_fit():
    s, g = endpoints(1)
    cfg = NoiseSearchConfig(mode='noise_random', samples=8, elites=2, steps=2)
    with patch('leflow_refinement.noise_search.fit_elites', side_effect=AssertionError('should not fit')):
        out = search(s, g, fixture()[0], cfg, generator=rng())
    assert out['noise_search']['min_search_std'] == [[1., 1., 1.]]


def test_deadline_exception_is_not_converted_to_returning_incumbent():
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls > 15:
            raise TimeoutError('controller deadline')
    s, g = endpoints(1)
    with pytest.raises(TimeoutError, match='deadline'):
        search(s, g, fixture(check=check)[0], NoiseSearchConfig(samples=8, elites=2, steps=2), generator=rng())


@pytest.mark.parametrize('field,value', [('samples', 0), ('samples', True), ('steps', -1),
    ('horizon', 1), ('rounds', 0), ('elites', 65), ('elites', False), ('smoothing', 1.),
    ('smoothing', -1.), ('smoothing', float('nan')), ('min_std', 0), ('min_std', 2),
    ('min_std', True), ('project_shell', 1), ('mode', 'gradient')])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        NoiseSearchConfig(**{field: value})


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_noise_fails_before_any_model_call(bad):
    s, g = endpoints(1); ops, _ = fixture()
    x = torch.zeros(1, 8, 4, 3); x[0, 0, 0, 0] = bad
    with pytest.raises(FloatingPointError):
        search(s, g, ops, NoiseSearchConfig(samples=8, elites=2), initial_noise=x)
    assert ops.compute.flow_batches == 0


def test_nonfinite_score_not_reported_as_completed():
    s, g = endpoints(1); ops, _ = fixture()
    ops.rollout_fn = lambda start, actions: torch.full((len(start), 6, 3), 1e30)
    with pytest.raises(FloatingPointError, match='costs'):
        search(s, g, ops, NoiseSearchConfig(samples=8, elites=2, steps=1), generator=rng())


def test_bad_shape_and_precision():
    s, g = endpoints(1)
    cfg = NoiseSearchConfig(samples=8, elites=2)
    with pytest.raises(ValueError, match='initial_noise'):
        search(s, g, fixture()[0], cfg, initial_noise=torch.zeros(8, 4, 3))
    with pytest.raises(ValueError, match='FP32'):
        search(s.half(), g.half(), fixture()[0], cfg)


def test_dispatch_preserves_original_refinement_path():
    s, g = endpoints(1)
    cfg = RefinementConfig(mode='pnp', samples=8, steps=4, refine_at=(1, 2), repeats=1)
    actual = plan(s, g, fixture()[0], cfg, generator=rng(), perturb_generator=rng(44))
    ops, _ = fixture()
    expected = select(sample(s, g, ops, cfg, generator=rng(), perturb_generator=rng(44)), ops)
    for key in expected:
        assert torch.equal(actual[key], expected[key])
    with pytest.raises(TypeError):
        plan(s, g, fixture()[0], object())


def test_default_comparison_counts():
    configs = noise_variants()
    for name, cfg in configs.items():
        assert cfg.samples * cfg.rounds * cfg.steps == 3072, name
        assert cfg.samples * cfg.rounds * cfg.horizon == 960, name
