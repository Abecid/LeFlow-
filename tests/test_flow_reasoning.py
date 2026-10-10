"""Transport correctness, causal deliberation and bounded execution behavior."""
import copy
import unittest

import torch

from flow_jepa.execution.flow_reasoning import FlowReasoningPolicy
from flow_jepa.execution.flow_controller import FlowReasoningController
from test_execution import World, FakeBank
import test_execution_revision as alignment_tests


class FlowReasoningTests(unittest.TestCase):
    def setUp(self):
        base = alignment_tests.AlignedTests(); base.setUp()
        self.c = base.c
        self.c['flow_reasoning'] = dict(rounds=3, solver_steps=8, candidates=4,
                                       flow_depth=3, usefulness_weight=.25)

    def batch(self):
        return dict(z=torch.randn(3, 4, 2, 8), z4=torch.randn(3, 2, 8),
            a=torch.rand(3, 5, 8)*2-1, history_start=torch.randn(3, 4, 2, 8),
            history_next=torch.randn(3, 4, 2, 8), history_action=torch.rand(3, 4, 8)*2-1,
            history_mask=torch.ones(3, 4, dtype=torch.bool))

    def context(self, model):
        z = torch.randn(2, 2, 8)
        return model.prepare(z, z+.2, z+.4, model.empty_history(z)), z

    def test_complete_flow_transport_uses_unrestricted_source_and_final_clamp(self):
        model = FlowReasoningPolicy(self.c).eval()
        context, _ = self.context(model)
        noise = torch.full((2, 4, 5, 8), 3.)
        endpoint = torch.full_like(noise, .25); seen = []
        def constant_velocity(ctx, actions, time):
            seen.append(actions.clone())
            return endpoint-noise
        model.velocity = constant_velocity
        actual = model.propose_context(context, 4, noise=noise)
        self.assertTrue(torch.allclose(actual, endpoint))
        self.assertTrue(torch.equal(seen[0], noise))
        self.assertEqual(len(seen), 8)
        self.assertTrue(torch.equal(noise, torch.full_like(noise, 3.)))

    def test_imagined_evidence_changes_generation_but_not_scoring_context(self):
        model = FlowReasoningPolicy(self.c).eval(); world = World()
        context, z = self.context(model)
        noise = torch.randn(2, 4, 5, 8)
        actions = model.propose_context(context, 4, noise=noise)
        predicted = world.rollout(z[:, None].expand(-1, 4, -1, -1).flatten(0, 1),
                                  actions.flatten(0, 1)).reshape(2, 4, 5, 2, 8)
        scores = model.score_candidates(context, actions, predicted, z+.2)
        revised = model.revise(context, actions, scores)
        alternate = {k:v.clone() for k, v in scores.items()}
        alternate['maps'][:, 0, 0] += .8
        changed = model.revise(context, actions, alternate)
        self.assertGreater(float((revised['thought']-changed['thought']).detach().abs().max()), 1e-5)
        a = model.propose_context(revised, 4, noise=noise)
        b = model.propose_context(changed, 4, noise=noise)
        self.assertGreater(float((a-b).abs().max()), 1e-7)
        after = model.score_candidates(changed, actions, predicted, z+.2)
        self.assertTrue(torch.equal(scores['cost'], after['cost']))
        self.assertTrue(torch.equal(context['score_thought'], revised['score_thought']))

    def test_candidate_pairing_permutation_and_detached_world_evidence(self):
        model = FlowReasoningPolicy(self.c).eval(); context, z = self.context(model)
        actions = torch.randn(2, 4, 5, 8, requires_grad=True)
        predicted = torch.randn(2, 4, 5, 2, 8, requires_grad=True)
        scores = model.score_candidates(context, actions, predicted, z)
        revised = model.revise(context, actions, scores)
        order = torch.tensor([2, 0, 3, 1])
        permuted = model.revise(context, actions[:, order], {k:v[:, order] for k, v in scores.items()})
        mismatched = model.revise(context, actions[:, order], scores)
        self.assertTrue(torch.allclose(revised['thought'], permuted['thought'], atol=1e-6))
        self.assertGreater(float((revised['thought']-mismatched['thought']).detach().abs().max()), 1e-6)
        revised['thought'].square().sum().backward()
        self.assertIsNone(actions.grad); self.assertIsNone(predicted.grad)

    def test_teacher_actions_and_future_labels_never_enter_deliberation(self):
        model = FlowReasoningPolicy(self.c).eval(); world = World(); batch = self.batch()
        seen = []
        handle = model.evidence.register_forward_pre_hook(lambda module, inputs:seen.append(inputs[0].detach().clone()))
        torch.manual_seed(99); first, _ = model(batch, world)
        original = seen.copy(); seen.clear()
        changed = {k:v.clone() for k, v in batch.items()}
        changed['a'].neg_(); changed['z'][:, 1].mul_(3); changed['z4'].mul_(3)
        torch.manual_seed(99); second, _ = model(changed, world)
        handle.remove()
        self.assertEqual(len(original), 2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, seen)))
        self.assertGreater(abs(float((first-second).detach())), 1e-5)

    def test_all_depths_train_flow_and_deep_evidence_with_world_frozen(self):
        model = FlowReasoningPolicy(self.c); world = World(); batch = self.batch()
        original = copy.deepcopy(world.state_dict())
        self.assertFalse(hasattr(model, 'output'))
        for depth in (1, 2, 3):
            model.zero_grad(set_to_none=True)
            loss, metrics = model(batch, world, depth=depth); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.flow.out[-1].weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.prefix_error[-1].weight.grad.abs().sum()), 0)
            if depth > 1:
                self.assertGreater(float(model.evidence[0].weight.grad.abs().sum()), 0)
            self.assertEqual(float(metrics['reasoning_depth']), depth)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in world.state_dict().items()))
        with self.assertRaisesRegex(ValueError, 'inherited'):
            model(batch, world, weight=.1)

    def test_masked_history_is_ignored_and_observed_error_affects_context(self):
        model = FlowReasoningPolicy(self.c).eval(); z = torch.randn(2, 2, 8)
        history = model.empty_history(z)
        initial = model.prepare(z, z+.2, z+.4, history)['thought']
        for key in ('start', 'next', 'predicted'):
            history[key].normal_(100, 10)
        ignored = model.prepare(z, z+.2, z+.4, history)['thought']
        self.assertTrue(torch.equal(initial, ignored))
        history['mask'][:, -1] = True
        observed = model.prepare(z, z+.2, z+.4, history)['thought']
        self.assertGreater(float((initial-observed).detach().abs().max()), 1e-5)

    def test_controller_uses_evidence_between_rounds_and_exact_execution_budget(self):
        model = FlowReasoningPolicy(self.c).eval(); world = World()
        controller = FlowReasoningController(model, world, self.c, FakeBank())
        seen = []
        handle = model.evidence.register_forward_pre_hook(lambda module, inputs:seen.append(world.calls))
        start, goal = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
        action, _, _, _ = controller.plan(start, goal)
        handle.remove()
        self.assertEqual(world.calls, 480); self.assertEqual(seen, [160, 320])
        trace = controller.trace[-1]
        self.assertEqual(trace['reasoning_depth'], 3)
        self.assertTrue(all(a >= b for a, b in zip(trace['round_best_scores'], trace['round_best_scores'][1:])))
        self.assertGreater(trace['round_action_changes'][-1], 0)
        self.assertIsNone(controller.previous_plan)
        predicted = world.rollout(start, action[None, None])[:, 0]
        self.assertTrue(torch.allclose(predicted, controller.pending[0]))
        controller.observe(predicted+.01)
        self.assertTrue(torch.equal(controller.previous_plan, controller.proposed_plan))
        controller.begin_episode()
        self.assertIsNone(controller.previous_plan); self.assertEqual(controller.history, [])


if __name__ == '__main__':
    unittest.main()
