"""Synthetic numerical tests. These are not native benchmark measurements."""
from dataclasses import replace
import unittest

import torch
from torch import nn

from leflow_refinement.core import RefinementConfig, PlannerOps, sample, select, correct


torch.set_num_threads(1)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2), requires_grad=False)

    def forward(self, x, t, start, goal):
        return self.scale * (goal[:, None] - x) + t[:, None, None] * 0.01

    def decode(self, paths):
        return (paths[:, 1:] - paths[:, :-1]).tanh()

    def rollout(self, start, actions):
        return torch.cat((start[:, None], start[:, None] + (0.8 * actions).cumsum(1)), 1)


def setup(mode="world_pnp", samples=3):
    torch.manual_seed(7)
    model = Toy().eval()
    ops = PlannerOps(model, model.decode, model.rollout)
    cfg = RefinementConfig(mode=mode, samples=samples, steps=4, horizon=3,
                           refine_at=(1, 2), repeats=2)
    start, goal = torch.zeros(2, 4), torch.ones(2, 4)
    noise = torch.randn(2, samples, 2, 4)
    return model, ops, cfg, start, goal, noise


class RefinementTests(unittest.TestCase):
    def test_euler_matches_original_loop(self):
        model, ops, cfg, s, g, noise = setup("euler")
        result = sample(s, g, ops, cfg, initial_noise=noise)
        x = noise.reshape(-1, 2, 4).clone()
        ss, gg = s.repeat_interleave(3, 0), g.repeat_interleave(3, 0)
        for i in range(4):
            t = x.new_full((6,), i / 4)
            x = x + model(x, t, ss, gg) / 4
        torch.testing.assert_close(result[:, :, 1:-1], x.reshape(2, 3, 2, 4), rtol=0, atol=0)
        self.assertEqual(ops.compute.flow_batches, 4)
        self.assertEqual(ops.compute.flow_candidate_evaluations, 24)
        self.assertEqual(ops.compute.world_batches, 0)

    def test_fixed_endpoints_every_mode(self):
        for mode in ("euler", "pnp", "gradient", "world_pnp"):
            _, ops, cfg, s, g, noise = setup(mode)
            p = sample(s, g, ops, cfg, initial_noise=noise)
            torch.testing.assert_close(p[:, :, 0], s[:, None].expand(2, 3, 4))
            torch.testing.assert_close(p[:, :, -1], g[:, None].expand(2, 3, 4))
            self.assertFalse(p.requires_grad)

    def test_exact_compute_counts(self):
        for mode in ("pnp", "gradient", "world_pnp"):
            _, ops, cfg, s, g, noise = setup(mode)
            p = sample(s, g, ops, cfg, initial_noise=noise)
            result = select(p, ops)
            self.assertEqual(result["actions"].shape, (2, 3, 4))
            self.assertEqual(ops.compute.flow_batches, 8)
            self.assertEqual(ops.compute.flow_candidate_evaluations, 48)
            corrections = 0 if mode == "pnp" else 4
            self.assertEqual(ops.compute.backward_batches, corrections)
            self.assertEqual(ops.compute.world_batches, (corrections + 1) * 3)
            self.assertEqual(ops.compute.world_candidate_transitions, (corrections + 1) * 18)
            self.assertEqual(ops.compute.backward_candidate_transitions, corrections * 18)
            self.assertEqual(ops.compute.inverse_batches, corrections + 1)

    def test_gradient_reaches_path_not_weights(self):
        model, ops, cfg, s, g, noise = setup()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        p = sample(s, g, ops, cfg, initial_noise=noise)
        self.assertGreater(ops.compute.gradient_norm_sum, 0)
        self.assertLessEqual(ops.compute.max_node_displacement, cfg.radius + 1e-6)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(torch.isfinite(p).all())

    def test_nested_inference_mode(self):
        with torch.inference_mode():
            _, ops, cfg, s, g, noise = setup()
            p = sample(s, g, ops, cfg, initial_noise=noise)
            select(p, ops)
        self.assertGreater(ops.compute.gradient_norm_sum, 0)

    def test_zero_refinement_is_bitwise_euler(self):
        _, ops, cfg, s, g, noise = setup()
        p = sample(s, g, ops, replace(cfg, repeats=0), initial_noise=noise)
        _, ops2, _, _, _, _ = setup()
        q = sample(s, g, ops2, replace(cfg, mode="euler"), initial_noise=noise)
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_deterministic_independent_perturb_stream(self):
        outputs = []
        for _ in range(2):
            _, ops, cfg, s, g, noise = setup()
            outputs.append(sample(s, g, ops, cfg, initial_noise=noise,
                                  perturb_generator=torch.Generator().manual_seed(91)))
        torch.testing.assert_close(*outputs, rtol=0, atol=0)

    def test_initial_noise_is_not_mutated(self):
        _, ops, cfg, s, g, noise = setup()
        saved = noise.clone()
        sample(s, g, ops, cfg, initial_noise=noise)
        torch.testing.assert_close(noise, saved, rtol=0, atol=0)

    def test_pnp_does_not_query_world_during_sampling(self):
        _, ops, cfg, s, g, noise = setup("pnp")
        sample(s, g, ops, cfg, initial_noise=noise)
        self.assertEqual(ops.compute.world_candidate_transitions, 0)

    def test_gradient_control_radius_zero_is_euler(self):
        _, ops, cfg, s, g, noise = setup("gradient")
        p = sample(s, g, ops, replace(cfg, radius=0), initial_noise=noise)
        _, ops2, _, _, _, _ = setup()
        q = sample(s, g, ops2, replace(cfg, mode="euler"), initial_noise=noise)
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_consistency_and_goal_use_forward_rollout(self):
        _, ops, cfg, s, g, noise = setup("euler")
        p = sample(s, g, ops, cfg, initial_noise=noise)
        out = select(p, ops)
        # The generated endpoint equals the goal, but scored cost must not be zero.
        self.assertGreater(float(out["all_costs"].min()), 0)
        self.assertEqual(out["all_costs"].shape, (2, 3))

    def test_update_independent_of_candidate_batch_size(self):
        _, ops, cfg, s, g, noise = setup()
        one = correct(noise[0, :1], s[:1], g[:1], ops, cfg)
        _, ops2, _, _, _, _ = setup()
        many = correct(noise[0, :1].repeat(5, 1, 1), s[:1].repeat(5, 1),
                       g[:1].repeat(5, 1), ops2, cfg)
        torch.testing.assert_close(one.expand_as(many), many)

    def test_frozen_clean_update_improves_simple_objective(self):
        _, ops, cfg, s, g, noise = setup()
        y = noise[0, :1]
        def energy(x):
            path = torch.cat((s[:1, None], x, g[:1, None]), 1)
            _, pred = ops.consequences(s[:1], path)
            return ((pred[:, -1]-g[:1]).square().mean()
                    + cfg.consistency_weight*(pred[:, 1:-1]-x).square().mean())
        revised = correct(y, s[:1], g[:1], ops, replace(cfg, radius=1e-3))
        self.assertLess(float(energy(revised)), float(energy(y)))

    def test_deadline_propagates(self):
        _, ops, cfg, s, g, noise = setup()
        def stop():
            raise TimeoutError("deadline")
        ops.check = stop
        with self.assertRaises(TimeoutError):
            sample(s, g, ops, cfg, initial_noise=noise)

    def test_nan_input_fails(self):
        _, ops, cfg, s, g, noise = setup()
        noise[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            sample(s, g, ops, cfg, initial_noise=noise)

    def test_nan_velocity_fails(self):
        model, _, cfg, s, g, noise = setup()
        ops = PlannerOps(lambda x, *a: x*float("nan"), model.decode, model.rollout)
        with self.assertRaises(FloatingPointError):
            sample(s, g, ops, cfg, initial_noise=noise)

    def test_invalid_configurations(self):
        for kw in ({"mode":"bad"}, {"samples":0}, {"steps":0}, {"horizon":1},
                   {"repeats":-1}, {"radius":-1}, {"radius":float("nan")},
                   {"refine_at":(0,)}, {"refine_at":(1,1)}, {"samples":2.5}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                RefinementConfig(**kw)

    def test_bad_shapes_fail(self):
        _, ops, cfg, s, g, noise = setup()
        with self.assertRaises(ValueError):
            sample(s, g[:1], ops, cfg)
        with self.assertRaises(ValueError):
            sample(s, g, ops, cfg, initial_noise=noise[:, :1])

    def test_bad_rollout_contract_fails(self):
        model, _, cfg, s, g, noise = setup()
        ops = PlannerOps(model, model.decode, lambda s, a: s[:, None])
        with self.assertRaises(ValueError):
            sample(s, g, ops, cfg, initial_noise=noise)

    def test_no_diffusion_or_endpoint_variable_confusion(self):
        target = torch.randn(2, 3, 4)
        noise = torch.randn_like(target)
        for t in (0., 0.2, 0.7, 1.):
            x = (1-t)*noise+t*target
            clean = x+(1-t)*(target-noise)
            torch.testing.assert_close(clean, target)


if __name__ == "__main__":
    unittest.main()
