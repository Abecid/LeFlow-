import tempfile
from pathlib import Path
import unittest
import torch
from crossbench.spec import Spec, METHODS
from crossbench.train import run
from crossbench.world import LeWMWorld, ActionUnits
from flow_jepa.common import save_json
from test_crossbench_core import TinyLeWM
from test_crossbench_data_eval import make_cache


def fake_eval(model,world,bank,s,root,out,sha,device,rank,size):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=dict(successes=1,episodes=1,success_rate=1.,timeouts=0)
    save_json(out/'report.json',dict(metrics=m,records=[],fixture=True))
    return m


def world():return LeWMWorld(TinyLeWM(),ActionUnits([-1]*2,[1]*2,[0]*2,[1]*2))


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_all_eight_methods_train_checkpoint_and_validate(self):
        for method in METHODS:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as root:
                make_cache(root);s=Spec('pusht',method,dim=16,width=32,updates=4,global_batch=2,eval_episodes=1)
                dest=Path(root)/'run'
                result=run(s,root,dest,'cpu',world=world(),eval_fn=fake_eval,allow_fixture=True,online=False)
                self.assertEqual(result['validation_steps'],[1,2,3,4])
                self.assertEqual(result['step'],4);self.assertTrue((dest/'best.pt').exists())
                self.assertTrue((dest/'complete.json').exists())
    def test_interrupted_eval_resume_preserves_parameters(self):
        with tempfile.TemporaryDirectory() as root:
            make_cache(root);s=Spec('pusht','vanilla_action_flow',dim=16,width=32,updates=4,global_batch=2,eval_episodes=1)
            complete=Path(root)/'continuous';resume=Path(root)/'resume'
            run(s,root,complete,'cpu',world=world(),eval_fn=fake_eval,allow_fixture=True,online=False)
            def broken(*args):
                if 'step_0000002' in str(args[5]):raise RuntimeError('injected interruption')
                return fake_eval(*args)
            with self.assertRaisesRegex(RuntimeError,'injected'):
                run(s,root,resume,'cpu',world=world(),eval_fn=broken,allow_fixture=True,online=False)
            run(s,root,resume,'cpu',world=world(),eval_fn=fake_eval,allow_fixture=True,online=False)
            a=torch.load(complete/'last.pt',weights_only=False)['model']
            b=torch.load(resume/'last.pt',weights_only=False)['model']
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))


if __name__=='__main__':unittest.main()
