import copy
import unittest
import torch
from flow_jepa.leflow_ttt.model import FastMemory, memory_gradients, read_memory, AdaptiveLeFlow


def cfg():
    return dict(encoder=dict(dim=16),baseline=dict(leflow=dict(horizon=5,
        flow=dict(hidden_dim=512,depth=1,max_horizon=20,time_dim=64,dropout=0.),
        inverse=dict(hidden_dim=512,depth=1,dropout=0.))),leflow_ttt=dict(memory_dim=8,memory_hidden=12,
        residual_scale=.1,residual_bound=.5,elastic=.1,inner_gradient_cap=10.,inner_steps=4))


def history(b=2):
    return dict(start=torch.randn(b,4,3,16),predicted=torch.randn(b,4,3,16),
        next=torch.randn(b,4,3,16),actions=torch.randn(b,4,5,8),mask=torch.ones(b,4,dtype=torch.bool))


class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)

    def test_inner_gradient_is_exact(self):
        weights=(torch.randn(2,3,5,dtype=torch.float64,requires_grad=True),
                 torch.randn(2,5,4,dtype=torch.float64,requires_grad=True),
                 torch.randn(2,3,5,dtype=torch.float64,requires_grad=True))
        keys=torch.randn(2,7,3,dtype=torch.float64);values=torch.randn(2,7,4,dtype=torch.float64)
        mask=torch.tensor([[1,1,1,0,0,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
        loss=.5*(((read_memory(weights,keys)-values).square()*mask[...,None]).sum((1,2))/mask.sum(1)).sum()
        expected=torch.autograd.grad(loss,weights)
        for a,b in zip(memory_gradients(weights,keys,values,mask),expected):torch.testing.assert_close(a,b)

    def test_empty_history_and_nonfinite_padding(self):
        m=FastMemory(cfg());h=history();h['mask'].zero_()
        for k in ('start','predicted','next','actions'):h[k].fill_(float('nan'))
        weights,d=m.fit(h,4)
        for w,p in zip(weights,m.prior(2)):self.assertTrue(torch.equal(w,p))
        self.assertEqual(float(d['fast_delta']),0.)
        self.assertTrue(all(torch.isfinite(w).all() for w in weights))

    def test_empty_support_meta_gradient_is_finite(self):
        m=FastMemory(cfg());h=history();h['mask'][0].zero_()
        weights,_=m.fit(h,4)
        m.correction(weights,torch.randn(2,3,16),torch.randn(2,5,8),torch.randn(2,3,16)).square().mean().backward()
        for p in m.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())

    def test_no_cross_example_or_parameter_mutation(self):
        m=FastMemory(cfg());h=history();before={k:v.clone() for k,v in m.state_dict().items()}
        a,_=m.fit(h,4);other=copy.deepcopy(h)
        for k in ('start','predicted','next','actions'):other[k][1].add_(40)
        b,_=m.fit(other,4)
        for x,y in zip(a,b):torch.testing.assert_close(x[0],y[0],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(x[1],y[1]) for x,y in zip(a,b)))
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_outer_gradient_through_inner_update(self):
        m=FastMemory(cfg());h=history();weights,_=m.fit(h,4)
        output=m.correction(weights,torch.randn(2,3,16),torch.randn(2,5,8),torch.randn(2,3,16))
        output.square().mean().backward()
        for parameter in (m.w0,m.w1,m.w2,m.value.weight,m.key[0].weight,m.lr_logit):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_zero_conditioning_preserves_native_modules(self):
        m=AdaptiveLeFlow(cfg()).eval();x=torch.randn(2,4,512);t=torch.rand(2);z0=torch.randn(2,512);zg=torch.randn(2,512)
        ctx=torch.randn(2,8);paths=torch.cat((z0[:,None],x,zg[:,None]),1)
        with torch.no_grad():
            torch.testing.assert_close(m.velocity(x,t,z0,zg,ctx),m.flow(x,t,z0,zg),rtol=0,atol=0)
            native=m.inverse(paths[:,:-1].reshape(-1,512),paths[:,1:].reshape(-1,512)).reshape(2,5,8)
            torch.testing.assert_close(m.decode(paths,ctx),native)

    def test_memory_affects_generation_and_decoding(self):
        m=AdaptiveLeFlow(cfg()).eval()
        torch.nn.init.normal_(m.flow_context.weight,std=.02);torch.nn.init.normal_(m.inverse_context.weight,std=.02)
        h=history();w,_=m.memory.fit(h,4)
        z0=torch.randn(2,512);zg=torch.randn(2,512)
        c=m.memory.context(w,z0,zg);p=m.memory.context(m.memory.prior(2),z0,zg)
        x=torch.randn(2,4,512);t=torch.rand(2);paths=torch.cat((z0[:,None],x,zg[:,None]),1)
        self.assertGreater(float((m.velocity(x,t,z0,zg,c)-m.velocity(x,t,z0,zg,p)).abs().max()),1e-6)
        self.assertGreater(float((m.decode(paths,c)-m.decode(paths,p)).abs().max()),1e-6)


if __name__=='__main__':unittest.main()
