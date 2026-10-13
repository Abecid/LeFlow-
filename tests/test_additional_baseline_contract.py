"""Config, dispatcher and no-test-leak comparison contract for five new runs."""
import unittest
from pathlib import Path
from flow_jepa.common import config
from flow_jepa.execution.train import component_types


class TestAdditionalBaselineContract(unittest.TestCase):
    def test_shared_data_and_budget(self):
        source=config("config/flow_metaworld_flow_reasoning.json")
        names=("vanilla_action_flow","action_flow_search","sage_port",
               "diffusion_policy_port","feedback_wm_port")
        for method in names:
            with self.subTest(method=method):
                p=Path(f"config/flow_metaworld_{method}.json")
                self.assertTrue(p.exists())
                c=config(p)
                for field in ("schema","training_tasks","heldout_tasks","seeds",
                              "data","encoder","evaluation","training"):
                    self.assertEqual(c[field],source[field],field)
                self.assertEqual(c["controller_grounded"]["world_sha256"],
                                 source["controller_grounded"]["world_sha256"])
                self.assertEqual(c["controller_grounded"]["optimization_gpu_seconds"],28800)
                self.assertEqual(c["training"]["planner_steps"],20000)
                self.assertEqual(c["training"]["global_batch"],64)
                self.assertEqual(c["evaluation"]["budget_primitive"],200)
                self.assertEqual(c["evaluation"]["controller_seconds_per_episode"],10)
                self.assertFalse(c["execution"]["test_enabled"])
                self.assertEqual(c["methods"],[method])
                method_name,policy,dataset,controller=component_types(c)
                self.assertEqual(method_name,method)
                self.assertTrue(callable(policy) and callable(controller) and callable(dataset))


if __name__=="__main__":
    unittest.main()
