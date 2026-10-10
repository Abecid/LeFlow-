"""Causal adaptation, analytic fit, useful gradients and unchanged search cost."""
import copy
import unittest

import torch

from flow_jepa.execution.progress_ttt import ProgressTTTPolicy, ProgressTTTController, ridge_update
from test_execution import World, FakeBank
import test_flow_reasoning as flow_tests


class ProgressTTTTests(unittest.TestCase):
    def setUp(self):
        self.base = flow_tests.FlowReasoningTests(); self.base.setUp()
        self.c = self.base.c
        self.c['progress_ttt'] = dict(rank=8, ridge=.125, key_width=16)

    def test_ridge_optimum_gradient_and_empty_prior(self):
        x = torch.randn(2, 6, 3, requires_grad=True)
        y = torch.randn(2, 6); mask = torch.ones(2, 6, dtype=torch.bool)
        prior = torch.randn(3, requires_grad=True)
        w = ridge_update(x, y, mask, prior, .125, 2)
        normal = x.transpose(-1, -2) @ ((x*w[:, None]).sum(-1)-y)[..., None]/2
        normal = normal.squeeze(-1)+.125*(w-prior)
        self.assertLess(float(normal.detach().abs().max()), 2e-6)
        loss = w.square().sum(); dx, dp = torch.autograd.grad(loss, (x, prior))
        direction = torch.randn_like(x); eps = 1e-3
        plus = ridge_update(x.detach()+eps*direction, y, mask, prior.detach(), .125, 2).square().sum()
        minus = ridge_update(x.detach()-eps*direction, y, mask, prior.detach(), .125, 2).square().sum()
        self.assertTrue(torch.allclose((plus-minus)/(2*eps), (dx*direction).sum(), atol=.003, rtol=.005))
        self.assertGreater(float(dp.abs().sum()), 0)
        empty = ridge_update(torch.full_like(x, float('nan')), torch.full_like(y, float('nan')),
                             ~mask, prior, .125, 2)
        self.assertTrue(torch.equal(empty, prior.expand(2, -1)))

    def test_padding_batch_isolation_and_no_parameter_mutation(self):
        model = ProgressTTTPolicy(self.c).eval(); z = torch.randn(2, 2, 8)
        before = copy.deepcopy(model.state_dict()); history = model.empty_history(z)
        a = model.prepare(z, z+.2, z+.4, history)
        for k in ('start','next','predicted','action'): history[k].fill_(float('nan'))
        b = model.prepare(z, z+.2, z+.4, history)
        self.assertTrue(torch.equal(a['thought'], b['thought']))
        self.assertTrue(torch.equal(b['fast_weights'], model.fast_prior.expand(2, -1)))
        history = model.factual_history(self.base.batch(), World()); q = torch.randn(3, 2, 8)
        w = model.fit_memory(history, q)
        history['next'][0].mul_(-1)
        changed = model.fit_memory(history, q)
        self.assertFalse(torch.equal(w[0], changed[0]))
        self.assertTrue(torch.equal(w[1:], changed[1:]))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_prefix_correction_ignores_action_suffix_and_search_revision(self):
        model = ProgressTTTPolicy(self.c).eval(); world = World(); batch = self.base.batch()
        z = batch['z'][:, 0]; q = batch['z'][:, 2]
        context = model.prepare(z, q, batch['z'][:, 3], model.factual_history(batch, world))
        actions = torch.randn(3, 4, 5, 8); predicted = torch.randn(3, 4, 5, 2, 8)
        scores = model.score_candidates(context, actions, predicted, q)
        other = actions.clone(); other[:, :, 1:].mul_(5)
        changed = model.score_candidates(context, other, predicted, q)
        self.assertTrue(torch.equal(scores['correction'][:, :, 0], changed['correction'][:, :, 0]))
        revised = model.revise(context, actions, scores)
        rescored = model.score_candidates(revised, actions, predicted, q)
        self.assertTrue(torch.equal(scores['cost'], rescored['cost']))
        self.assertIs(context['fast_weights'], revised['fast_weights'])

    def test_teacher_future_never_enters_fit_or_candidate_evidence(self):
        model = ProgressTTTPolicy(self.c).eval(); world = World(); batch = self.base.batch()
        seen = []; weights = []
        handle = model.evidence.register_forward_pre_hook(lambda m, xs:seen.append(xs[0].detach().clone()))
        original_fit = model.fit_memory
        def fit(history, target):
            result = original_fit(history, target); weights.append(result.detach().clone()); return result
        model.fit_memory = fit
        torch.manual_seed(99); first, _ = model(batch, world)
        original, original_weights = seen.copy(), weights.copy(); seen.clear(); weights.clear()
        changed = {k:v.clone() for k, v in batch.items()}
        changed['a'].neg_(); changed['z'][:, 1].mul_(-3); changed['z4'].mul_(-3)
        torch.manual_seed(99); second, _ = model(changed, world)
        handle.remove()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, seen)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original_weights, weights)))
        self.assertGreater(abs(float((first-second).detach())), 1e-5)

    def test_outer_gradients_reach_fast_features_prior_flow_and_readout(self):
        model = ProgressTTTPolicy(self.c); world = World(); batch = self.base.batch()
        before = copy.deepcopy(world.state_dict())
        for depth in (1, 2, 3):
            model.zero_grad(set_to_none=True)
            loss, metrics = model(batch, world, depth=depth); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for p in (model.fast_keys[0].weight, model.fast_prior,
                      model.fast_condition.weight, model.flow.out[-1].weight):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(float(p.grad.abs().sum()), 0)
            if depth > 1:self.assertGreater(float(model.evidence[0].weight.grad.abs().sum()), 0)
            self.assertEqual(float(metrics['reasoning_depth']), depth)
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in world.state_dict().items()))
        with self.assertRaisesRegex(ValueError, 'inherited'):model(batch, world, weight=.1)

    def test_fast_readout_can_change_flow_without_changing_physical_states(self):
        model = ProgressTTTPolicy(self.c).eval(); world = World(); batch = self.base.batch()
        z = batch['z'][:, 0]; q = batch['z'][:, 2]; h = model.factual_history(batch, world)
        context = model.prepare(z, q, batch['z'][:, 3], h)
        noise = torch.randn(3, 4, 5, 8)
        first = model.propose_context(context, 4, noise=noise)
        with torch.no_grad():model.fast_condition.weight.zero_()
        no_readout = model.prepare(z, q, batch['z'][:, 3], h)
        second = model.propose_context(no_readout, 4, noise=noise)
        self.assertGreater(float((first-second).abs().max()), 1e-7)
        self.assertTrue(torch.equal(context['fast_weights'], no_readout['fast_weights']))
        self.assertTrue(torch.equal(context['start'], no_readout['start']))

    def test_fast_correction_changes_action_selection_with_identical_world_costs(self):
        model = ProgressTTTPolicy(self.c).eval(); z = torch.randn(1, 2, 8)
        context = model.prepare(z, z, z, model.empty_history(z))
        actions = torch.zeros(1, 2, 5, 8)
        actions[0, 0, 0, 0] = 1.; actions[0, 1, 0, 0] = -1.
        predicted = z[:, None, None].expand(1, 2, 5, 2, 8)
        def keys(start, action, prefix, target):
            result = start.new_zeros(*start.shape[:-1], model.rank)
            result[..., 0] = action[..., 0, None]
            return result
        model.transition_keys = keys
        weights = torch.zeros(1, model.rank); weights[:, 0] = 1.
        a = model.score_candidates({**context, 'fast_weights':weights}, actions, predicted, z)
        b = model.score_candidates({**context, 'fast_weights':-weights}, actions, predicted, z)
        self.assertTrue(torch.equal(a['raw'], b['raw']))
        self.assertEqual(int(a['cost'].argmin()), 1)
        self.assertEqual(int(b['cost'].argmin()), 0)

    def test_controller_stores_only_observed_prefixes_resets_and_fits_once(self):
        model = ProgressTTTPolicy(self.c).eval(); world = World()
        controller = ProgressTTTController(model, world, self.c, FakeBank())
        start, goal = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
        before = copy.deepcopy(model.state_dict()); calls = []
        original = model.fit_memory
        def fit(history, target):calls.append(history['mask'].clone()); return original(history, target)
        model.fit_memory = fit
        for i in range(6):
            prior_calls = world.calls
            action, _, _, _ = controller.plan(start, goal)
            self.assertEqual(world.calls-prior_calls, 480)
            self.assertEqual(len(calls), i+1)
            self.assertEqual(len(controller.action_history), min(i, 4))
            self.assertEqual(controller.trace[-1]['fast_support_steps'], min(i, 4))
            predicted = world.rollout(start, action[None, None])[:, 0]
            self.assertTrue(torch.allclose(predicted, controller.pending[0]))
            start = predicted+.03
            controller.observe(start)
            self.assertTrue(torch.equal(controller.action_history[-1], action))
            controller.observe(start)  # Duplicate observation cannot duplicate support.
            self.assertEqual(len(controller.action_history), min(i+1, 4))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        controller.begin_episode()
        self.assertEqual(controller.history, []); self.assertEqual(controller.action_history, [])
        self.assertFalse(controller.model_history(start, 8)['mask'].any())


if __name__ == '__main__':unittest.main()
