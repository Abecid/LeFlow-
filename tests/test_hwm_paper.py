import hashlib
import json
from pathlib import Path
import unittest

import torch

from flow_jepa.baselines.hwm import HWM, ActionEncoder


class HWMPaperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(3072)
        self.c=json.loads(Path('config/flow_metaworld_hwm_paper.json').read_text())

    def test_only_device_placement_differs_from_dino_predictor_source(self):
        root=Path('flow_jepa/baselines/vendor')
        p=json.loads((root/'dino-provenance.json').read_text())
        s=(root/'dino_vit.py').read_text()
        self.assertEqual(hashlib.sha256(s.encode()).hexdigest(),p['vendored_sha256'])
        original=s.replace(p['only_change']['after'],p['only_change']['before'])
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),p['source_sha256'])

    def test_action_encoder_ignores_padding(self):
        enc=ActionEncoder(width=32,heads=4).eval()
        x=torch.randn(2,70,4); lengths=torch.tensor([8,20])
        y=x.clone();y[0,8:]=100.;y[1,20:]=-100.
        with torch.no_grad(): a,b=enc(x,lengths),enc(y,lengths)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_predictor_is_causal_and_uses_history(self):
        model=HWM(self.c).eval()
        z=torch.randn(1,3,32,1024);m=torch.randn(1,3,4)
        with torch.no_grad():
            base=model.predict(z,m)
            z2,m2=z.clone(),m.clone();z2[:,2]=50.;m2[:,2]=-50.
            future=model.predict(z2,m2)
            torch.testing.assert_close(base[:,:2],future[:,:2],rtol=0,atol=0)
            z2=z.clone();z2[:,0]=-30.
            past=model.predict(z2,m)
            self.assertFalse(torch.allclose(base[:,-1],past[:,-1]))

    def test_teacher_forcing_is_only_l1_on_actual_waypoints(self):
        model=HWM(self.c).eval()
        z=torch.randn(1,5,32,1024);a=torch.randn(1,4,70,4)
        lengths=torch.tensor([[4,6,12,20]])
        with torch.no_grad():
            loss,parts=model(dict(z=z,a=a,lengths=lengths))
            macro=model.action_encoder(a.flatten(0,1),lengths.flatten()).reshape(1,4,4)
            expected=(model.predict(z[:,:-1],macro)-z[:,1:]).abs().mean()
        self.assertTrue(torch.equal(loss,expected))
        self.assertEqual(float(loss),float(parts['teacher_forcing_l1']))
        self.assertGreater(sum(p.numel() for p in model.parameters()),70_000_000)


    def test_paper_cem_zero_momentum_is_native_solver(self):
        from flow_jepa.baselines.hwm import PaperCEM,CEMSolver
        from types import SimpleNamespace
        from gymnasium.spaces import Box
        import numpy as np
        class Cost:
            def get_cost(self,info,actions): return (actions-.4).square().mean((2,3))
        kwargs=dict(num_samples=19,topk=5,n_steps=4,seed=91)
        native=CEMSolver(Cost(),**kwargs);paper=PaperCEM(Cost(),momentum=0.,**kwargs)
        for solver in [native,paper]:
            solver.configure(action_space=Box(-1.,1.,shape=(1,4),dtype=np.float32),
                n_envs=1,config=SimpleNamespace(horizon=2,action_block=1))
        a,b=native(dict(start=torch.zeros(1,4))),paper(dict(start=torch.zeros(1,4)))
        self.assertTrue(torch.equal(a['actions'],b['actions']))
        self.assertEqual(a['costs'],b['costs'])

    def test_momentum_smooths_standard_deviation_only(self):
        from flow_jepa.baselines.hwm import PaperCEM
        from types import SimpleNamespace
        from gymnasium.spaces import Box
        import numpy as np
        class Cost:
            def get_cost(self,info,actions): return (actions-.4).square().mean((2,3))
        solver=PaperCEM(Cost(),num_samples=19,topk=5,n_steps=4,seed=91,momentum=.4)
        solver.configure(action_space=Box(-1.,1.,shape=(1,4),dtype=np.float32),n_envs=1,
            config=SimpleNamespace(horizon=2,action_block=1))
        actual=solver(dict(start=torch.zeros(1,4)))['actions']
        mean,std=torch.zeros(1,2,4),torch.ones(1,2,4)
        generator=torch.Generator().manual_seed(91)
        for _ in range(4):
            samples=torch.randn(1,19,2,4,generator=generator)*std[:,None]+mean[:,None]
            samples[:,0]=mean
            indices=(samples-.4).square().mean((2,3)).topk(5,dim=1,largest=False).indices
            elites=samples[torch.arange(1)[:,None],indices]
            mean=elites.mean(1);std=.4*std+.6*elites.std(1)
        self.assertTrue(torch.equal(actual,mean))

if __name__=='__main__': unittest.main()
