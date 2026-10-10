"""Action-gradient, causal trust and deployment-budget checks for guided TTT."""
import copy
import unittest

import torch

from flow_jepa.execution.guided_ttt import GuidedTTTPolicy, GuidedTTTController
from test_execution import World, FakeBank
import test_progress_ttt as progress_tests


class GuidedTTTTests(unittest.TestCase):
    def setUp(self):
        self.base = progress_tests.ProgressTTTTests(); self.base.setUp()
        self.c = self.base.c
        self.c['guided_ttt'] = dict(width=16, first_guided_step=4, strength=1.,
            gradient_rms_cap=.25, outer_step=.5, trust_floor=.01,
            surrogate_weight=.1, guided_bc_weight=.25, meta_weight=.25, anchor_weight=.01)

    def context(self, model):
        batch = self.base.base.batch(); z = batch['z']
        ctx = model.prepare(z[:, 0], z[:, 2], z[:, 3], model.factual_history(batch, World()))
        return ctx, batch

    def test_gradient_matches_finite_differences_and_no_solver_jacobian(self):
        model = GuidedTTTPolicy(self.c).eval(); ctx, _ = self.context(model)
        action = torch.rand(3, 2, 8)*.4-.2
        gradient = model.energy_gradient(ctx, action)
        direction = torch.randn_like(action); eps = .003
        def energy(a):
            base, correction = model.energy_maps(ctx, a)
            return (base+correction).mean(-1).sum()/model.error_scale
        numerical = (energy(action+eps*direction)-energy(action-eps*direction))/(2*eps)
        self.assertTrue(torch.allclose(numerical, (gradient*direction).sum(), atol=.003, rtol=.02))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        source = action.clone().requires_grad_(True)
        model.energy_gradient(ctx, source, create_graph=True).square().sum().backward()
        self.assertIsNone(source.grad)

    def test_clean_endpoint_guidance_sign_prefix_mask_and_full_gaussian(self):
        model = GuidedTTTPolicy(self.c).eval(); ctx, _ = self.context(model)
        noise = torch.full((3, 4, 5, 8), 3.)
        endpoint = torch.full_like(noise, .5); seen = []
        model.velocity = lambda ctx, a, t: endpoint-noise
        def gradient(ctx, prefix, **kw):
            seen.append(prefix.clone()); return torch.ones_like(prefix)
        model.energy_gradient = gradient
        out = model.propose_context(ctx, 4, noise=noise)
        self.assertEqual(len(seen), 4)
        self.assertTrue(torch.allclose(seen[0], torch.full_like(seen[0], .5)))
        self.assertTrue(torch.allclose(out[:, 1:3, 0], torch.full_like(out[:, 1:3, 0], .375)))
        self.assertTrue(torch.allclose(out[:, [0,3]], endpoint[:, [0,3]]))
        self.assertTrue(torch.allclose(out[:, :, 1:], endpoint[:, :, 1:]))
        self.assertTrue(torch.equal(noise, torch.full_like(noise, 3.)))

    def test_trust_requires_held_out_support_and_rejects_worse_prediction(self):
        model = GuidedTTTPolicy(self.c).eval(); z = torch.randn(1, 2, 8)
        history = model.empty_history(z)
        for key in ('start','next','predicted','action'): history[key].fill_(float('nan'))
        ctx = model.prepare(z, z+.2, z+.4, history)
        self.assertTrue(torch.isfinite(ctx['thought']).all())
        self.assertEqual(float(ctx['fast_trust']), 0)
        self.assertTrue(torch.equal(ctx['fast_weights'], model.fast_prior[None]))
        history = model.empty_history(z)
        history['mask'][:, -1] = True
        history['start'][:, -1] = z; history['predicted'][:, -1] = z
        history['next'][:, -1] = -z
        ctx = model.prepare(z, z+.2, z+.4, history)
        self.assertEqual(float(ctx['fast_trust']), 0)
        self.assertTrue(torch.equal(ctx['fast_weights'], model.fast_prior[None]))

    def test_trust_can_accept_predictive_adaptation(self):
        model = GuidedTTTPolicy(self.c).eval(); z = torch.randn(1, 2, 8)
        h = model.empty_history(z)
        h['mask'][:] = True
        h['start'][:] = z[:, None]; h['predicted'][:] = z[:, None]
        h['next'][:] = -z[:, None]
        ctx = model.prepare(z, z, z, h)
        self.assertGreater(float(ctx['fast_trust']), .5)
        self.assertLess(float(ctx['trust_adapted_mse']), float(ctx['trust_prior_mse']))
        # If the newest residual has the opposite sign, prior must win.
        h['predicted'][:, -1] = -z; h['next'][:, -1] = z
        changed = model.prepare(z, z, z, h)
        self.assertEqual(float(changed['fast_trust']), 0)

    def test_meta_gradients_reach_critic_memory_and_flow_world_unchanged(self):
        model = GuidedTTTPolicy(self.c); world = World(); batch = self.base.base.batch()
        before = copy.deepcopy(world.state_dict())
        for depth in (1, 3):
            model.zero_grad(set_to_none=True)
            loss, metrics = model(batch, world, depth=depth); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for p in (model.energy_state.weight, model.energy_action.weight, model.energy_keys.weight,
                      model.energy_base.weight, model.fast_prior, model.flow.out[-1].weight):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(float(p.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in world.parameters()))
            self.assertEqual(float(metrics['reasoning_depth']), depth)
        self.assertTrue(all(torch.equal(v, before[k]) for k,v in world.state_dict().items()))
        with self.assertRaisesRegex(ValueError, 'inherited'): model(batch, world, weight=.1)

    def test_future_actions_and_outcomes_do_not_enter_generation(self):
        model = GuidedTTTPolicy(self.c).eval(); world = World(); batch = self.base.base.batch()
        seen = []
        original = model.propose_context
        def propose(*args, **kw):
            result = original(*args, **kw); seen.append(result.clone()); return result
        model.propose_context = propose
        torch.manual_seed(24); model(batch, world, depth=3); before = seen.copy(); seen.clear()
        changed = {k:v.clone() for k,v in batch.items()}
        changed['a'].neg_(); changed['z'][:, 1].mul_(-3); changed['z4'].mul_(-2)
        torch.manual_seed(24); model(changed, world, depth=3)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before, seen)))

    def test_guidance_checks_budget_and_controller_preserves_world_allowance(self):
        model = GuidedTTTPolicy(self.c).eval(); world = World()
        controller = GuidedTTTController(model, world, self.c, FakeBank())
        start, goal = torch.randn(1,2,8), torch.randn(1,2,8)
        before = copy.deepcopy(model.state_dict())
        action, _, _, _ = controller.plan(start, goal)
        self.assertEqual(world.calls, 480)
        self.assertEqual(controller.trace[-1]['guidance_gradient_evaluations'], 12)
        self.assertTrue(torch.isfinite(action).all()); self.assertLessEqual(float(action.abs().max()), 1)
        controller.observe(controller.pending[0]+.01)
        self.assertTrue(torch.equal(controller.action_history[-1], action))
        controller.begin_episode(); self.assertFalse(controller.history); self.assertFalse(controller.action_history)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        ctx, _ = self.context(model)
        class Stop:
            def check(self): raise RuntimeError('budget exhausted')
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            model.propose_context(ctx,4,budget=Stop())


if __name__ == '__main__': unittest.main()
