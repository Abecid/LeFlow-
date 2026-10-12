"""Causal nonlinear adaptation, exact inner derivatives and decision effects."""
import copy
import unittest

import torch

from flow_jepa.execution.nonlinear_repair import NonlinearRepairPolicy, NonlinearRepairController
from flow_jepa.leflow_ttt.model import read_memory, memory_gradients
from test_execution import World, FakeBank
import test_retrieval_repair as fixtures


class NonlinearRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(3072)
        self.fixture = fixtures.RetrievalRepairTests(); self.fixture.setUp()
        self.c = copy.deepcopy(self.fixture.c)
        self.c['nonlinear_repair_ttt'] = dict(hidden=8, inner_steps=4, gradient_cap=5.,
            elastic=.01, initial_learning_rate=.05, max_learning_rate=.25)

    def test_manual_inner_gradients_and_outer_derivatives_match_autograd(self):
        weights = tuple(torch.randn(*shape, dtype=torch.float64, requires_grad=True)*.1
                        for shape in ((2, 3, 4), (2, 4, 1), (2, 3, 4)))
        keys = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        values = torch.randn(2, 5, 1, dtype=torch.float64)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        loss = (.5*(read_memory(weights, keys)-values).square().sum(-1)*mask /
                mask.sum(1)[:, None]).sum()
        exact = torch.autograd.grad(loss, weights, create_graph=True)
        manual = memory_gradients(weights, keys, values, mask)
        for a, b in zip(exact, manual):
            self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-10))
        reference = torch.autograd.grad(sum(g.square().sum() for g in exact), (*weights, keys), retain_graph=True)
        actual = torch.autograd.grad(sum(g.square().sum() for g in manual), (*weights, keys))
        for a, b in zip(reference, actual):
            self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-10))

    def test_empty_padding_episode_isolation_and_support_fit(self):
        m = NonlinearRepairPolicy(self.c); batch = self.fixture.batch(); w = World()
        history = m.factual_history(batch, w); q = batch['z'][:, 2]
        empty = m.empty_history(q)
        for k in ('start', 'next', 'predicted', 'action'): empty[k].fill_(float('nan'))
        fitted, diag = m.fit_memory(empty, q)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(fitted, m.memory_prior(3))))
        self.assertTrue(torch.equal(diag['delta'], torch.zeros(3)))
        fitted, diag = m.fit_memory(history, q)
        self.assertLess(float(diag['after'].mean().detach()), float(diag['before'].mean().detach()))
        changed = copy.deepcopy(history); changed['next'][0].neg_()
        other, _ = m.fit_memory(changed, q)
        self.assertTrue(all(torch.equal(a[1:], b[1:]) for a, b in zip(fitted, other)))
        self.assertTrue(any(not torch.equal(a[0], b[0]) for a, b in zip(fitted, other)))

    def test_all_fast_matrices_meta_gradients_and_world_frozen(self):
        m = NonlinearRepairPolicy(self.c); w = World(); batch = self.fixture.batch()
        original = copy.deepcopy(w.state_dict())
        for depth in (1, 2):
            m.zero_grad(set_to_none=True); loss, parts = m(batch, w, depth=depth); loss.backward()
            self.assertTrue(torch.isfinite(loss)); self.assertEqual(float(parts['memory_inner_steps']), 4)
            required = [m.memory_w0, m.memory_w1, m.memory_w2, m.inner_lr_logit,
                        m.fast_keys[0].weight, m.fast_condition.weight, m.flow.out[-1].weight]
            if depth == 2: required += [m.repair_flow.out[-1].weight, m.evidence[0].weight]
            for parameter in required:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in w.parameters()))
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in w.state_dict().items()))

    def test_future_outcomes_and_teacher_actions_cannot_enter_planning(self):
        m = NonlinearRepairPolicy(self.c).eval(); batch = self.fixture.batch(); captured = []
        revise = m.revise
        def record(context, actions, scores):
            result = revise(context, actions, scores)
            captured.append((actions.clone(), result['thought'].clone(), result['repair_anchor'].clone()))
            return result
        m.revise = record
        torch.manual_seed(10); first, _ = m(batch, World(), depth=2); before = captured.pop()
        changed = {k:v.clone() for k, v in batch.items()}
        changed['a'].neg_(); changed['z'][:, 1].neg_(); changed['z4'].neg_()
        torch.manual_seed(10); second, _ = m(changed, World(), depth=2); after = captured.pop()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))
        self.assertNotEqual(float(first.detach()), float(second.detach()))

    def test_memory_can_change_flow_repairs_and_selection(self):
        m = NonlinearRepairPolicy(self.c).eval(); batch = self.fixture.batch(); w = World()
        z, q = batch['z'][:, 0], batch['z'][:, 2]
        h = m.factual_history(batch, w); altered = copy.deepcopy(h); altered['next'].neg_()
        a = m.prepare(z, q, q, h); b = m.prepare(z, q, q, altered)
        noise = torch.randn(3, 4, 5, 8)
        self.assertFalse(torch.equal(m.propose_context(a, 4, noise=noise), m.propose_context(b, 4, noise=noise)))
        anchor = torch.zeros(3, 5, 8)
        self.assertFalse(torch.equal(m.propose_context({**a, 'repair_anchor':anchor}, 4, noise=noise),
                                    m.propose_context({**b, 'repair_anchor':anchor}, 4, noise=noise)))
        # Same world predictions: flipping the fitted function reverses selection.
        z = z[:1]; ctx = m.prepare(z, z, z, m.empty_history(z))
        actions = torch.zeros(1, 2, 5, 8); actions[0, 0, 0, 0] = 1; actions[0, 1, 0, 0] = -1
        predicted = z[:, None, None].expand(1, 2, 5, *z.shape[1:])
        def keys(start, action, prefix, target):
            key = start.new_zeros(*start.shape[:-1], m.rank)
            key[..., 0] = action[..., 0, None]; key[..., 1] = 1
            return key
        m.memory_keys = keys
        w0, w1, w2 = (torch.zeros_like(t) for t in m.memory_prior(1))
        w0[:, 0, 0] = 1; w2[:, 1, 0] = 1; w1[:, 0, 0] = 1
        for parameter in m.terminal_error.parameters(): parameter.data.zero_()
        one = m.score_candidates({**ctx, 'fast_weights':(w0, w1, w2)}, actions, predicted, z)
        two = m.score_candidates({**ctx, 'fast_weights':(w0, -w1, w2)}, actions, predicted, z)
        self.assertTrue(torch.equal(one['raw'], two['raw']))
        self.assertEqual(int(one['cost'].argmin()), 1); self.assertEqual(int(two['cost'].argmin()), 0)
        # Unexecuted suffix is absent from the prefix correction query.
        suffix = actions.clone(); suffix[:, :, 1:] = torch.randn_like(suffix[:, :, 1:])
        self.assertTrue(torch.equal(m.prefix_maps(ctx, actions, predicted[:, :, 0], z),
                                    m.prefix_maps(ctx, suffix, predicted[:, :, 0], z)))

    def test_controller_only_observed_prefixes_one_fit_reset_and_budget(self):
        m = NonlinearRepairPolicy(self.c).eval(); w = World()
        ctrl = NonlinearRepairController(m, w, self.c, FakeBank())
        original = copy.deepcopy(m.state_dict()); start, goal = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
        calls = []; fit = m.fit_memory
        def record(history, target): calls.append(history['mask'].clone()); return fit(history, target)
        m.fit_memory = record
        for i in range(6):
            before = w.calls; action, _, _, _ = ctrl.plan(start, goal); trace = ctrl.trace[-1]
            self.assertEqual(w.calls-before, 320); self.assertEqual(len(calls), i+1)
            self.assertEqual(trace['fast_support_steps'], min(i, 4))
            self.assertEqual(trace['memory_inner_steps'], 4)
            self.assertNotIn('fast_weight_ridge', trace)
            self.assertLessEqual(trace['selected_distance_from_incumbent_linf'], .250001)
            self.assertLessEqual(trace['round_best_scores'][1], trace['round_best_scores'][0]+1e-7)
            predicted = w.rollout(start, action[None, None])[:, 0]
            self.assertTrue(torch.allclose(predicted, ctrl.pending[0])); start = predicted+.01
            ctrl.observe(start); ctrl.observe(start)
            self.assertEqual(len(ctrl.action_history), min(i+1, 4))
            self.assertTrue(torch.equal(ctrl.action_history[-1], action))
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in m.state_dict().items()))
        ctrl.begin_episode(); self.assertFalse(ctrl.history); self.assertFalse(ctrl.action_history)
        self.assertIsNone(ctrl.previous_plan)


if __name__ == '__main__': unittest.main()
