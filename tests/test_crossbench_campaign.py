import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from crossbench.spec import matrix, METHODS
from crossbench.campaign import main
from crossbench.report import collect
from flow_jepa.common import save_json

ROOT=Path(__file__).resolve().parents[1]


class TestCampaign(unittest.TestCase):
    def test_default_plan_reuses_metaworld_and_has_sixteen_jobs(self):
        plan=matrix()
        self.assertEqual(len(plan['jobs']),16)
        self.assertFalse(plan['metaworld']['retrain'])
        self.assertFalse(plan['sealed_final_tests_enabled'])
        self.assertEqual({j['primitive_dim'] for j in plan['jobs'] if j['dataset']=='cube'},{5})
        self.assertEqual({j['method'] for j in plan['jobs']},set(METHODS))
    def test_dry_run_creates_no_training(self):
        with tempfile.TemporaryDirectory() as root:
            subprocess.run([sys.executable,'-m','crossbench.campaign','--workspace',root,
                            '--data-root',root+'/data'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            self.assertTrue((Path(root)/'plan.json').exists())
            self.assertFalse((Path(root)/'runs').exists())
    def test_execute_without_gpu_records_blocker(self):
        with tempfile.TemporaryDirectory() as root,patch('crossbench.campaign.shutil.which',return_value=None),patch.object(sys,'argv',[
              'campaign','--workspace',root,'--data-root',root+'/data','--execute']):
            with self.assertRaisesRegex(RuntimeError,'No NVIDIA'):main()
            status=json.loads((Path(root)/'status.json').read_text())
            self.assertFalse(status['training_started'])
    def test_missing_results_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as root:
            before=(ROOT/'docs/results-ledger.json').read_bytes()
            result=collect(root,ROOT)
            native=[r for r in result['rows'] if r['dataset']!='metaworld']
            self.assertEqual(len(native),16)
            self.assertTrue(all(r['status']=='not_started' and r['successes'] is None for r in native))
            mw=next(r for r in result['rows'] if r['dataset']=='metaworld' and r['method']=='vanilla_action_flow')
            self.assertEqual(mw['successes'],98)
            self.assertEqual(before,(ROOT/'docs/results-ledger.json').read_bytes())
    def test_fixture_result_refused(self):
        with tempfile.TemporaryDirectory() as root:
            run=Path(root)/'runs'/'pusht'/'vanilla_action_flow';run.mkdir(parents=True)
            save_json(run/'complete.json',{'identity':{'fixture':True}})
            save_json(run/'best.json',{})
            with self.assertRaisesRegex(ValueError,'fixture'):collect(root,ROOT)


if __name__=='__main__':unittest.main()
