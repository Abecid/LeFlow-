"""Fixture-only validation for case selection and comparison claim boundaries."""
from dataclasses import asdict, replace
import copy
import unittest
import numpy as np
from leflow_refinement.native_eval import case_plan, jsonable
from leflow_refinement.compare import paired
from crossbench.spec import Spec


class ProtocolTests(unittest.TestCase):
    def test_native_cases_are_validation_only(self):
        src=dict(ep_idx=np.repeat(np.arange(4),40),step_idx=np.tile(np.arange(40),4))
        plan=case_plan(src,10,42,25,[1,2])
        self.assertEqual(len(plan),10)
        self.assertTrue(all(r['episode'] in (1,2) and r['step']<=14 for r in plan))
        self.assertEqual(plan,case_plan(src,10,42,25,[1,2]))
        self.assertEqual(len({r['id'] for r in plan}),10)

    def test_no_implicit_all_source_evaluation(self):
        src=dict(ep_idx=np.zeros(40),step_idx=np.arange(40))
        with self.assertRaises(ValueError):case_plan(src,1,42,25,[])
        with self.assertRaises(ValueError):case_plan(src,100,42,25,[0])

    def reports(self):
        s=Spec('pusht','leflow',eval_episodes=2)
        records=[dict(id=str(i),episode=i,source_offset=40*i,start=0,goal_offset=25,
                      seed=i,success=(i==0)) for i in range(2)]
        a=dict(identity=dict(spec=s.identity(),cache='cache',cases='cases'),records=records)
        b=copy.deepcopy(a)
        b['identity']=dict(specification=asdict(s),cache='cache',case_hash='cases')
        return a,b

    def test_paired_wins_and_regressions(self):
        a,b=self.reports();b['records'][1]['success']=True
        result=paired(a,b)
        self.assertEqual(result['delta_percentage_points'],50)
        self.assertEqual(result['wins'],['1'])
        self.assertEqual(result['regressions'],[])
        self.assertFalse(result['latency_hardware_verified'])

    def test_missing_duplicate_or_changed_resets_rejected(self):
        for change in ('missing','duplicate','reset'):
            a,b=self.reports()
            if change=='missing':b['records'].pop()
            elif change=='duplicate':b['records'][1]=copy.deepcopy(b['records'][0])
            else:b['records'][1]['seed']=999
            with self.assertRaises(ValueError):paired(a,b)

    def test_different_compute_or_action_budgets_rejected(self):
        a,b=self.reports();b['identity']['specification']['controller_seconds']=20
        with self.assertRaises(ValueError):paired(a,b)
        a,b=self.reports();b['identity']['specification']['action_budget']=100
        with self.assertRaises(ValueError):paired(a,b)

    def test_native_crossbench_mix_rejected(self):
        a,b=self.reports();a['identity']['protocol']='native-swm-validation-refinement-v1'
        with self.assertRaises(ValueError):paired(a,b)

    def test_cache_or_cases_mismatch_rejected(self):
        for key in ('cache','case_hash'):
            a,b=self.reports();b['identity'][key]='different'
            with self.assertRaises(ValueError):paired(a,b)

    def test_primitive_serialization(self):
        self.assertEqual(jsonable(dict(a=np.int64(2),b=np.array([True]))),dict(a=2,b=[True]))


if __name__=='__main__':unittest.main()
