"""HWM paper-based spatial port. NOT released HWM robotics source.

Uses the paper's PushT high-level sizes, variable five-waypoint teacher forcing,
four-dimensional action CLS encoder, and Appendix C d=50 compute-sweep settings.
The pinned DINO-WM predictor is unchanged except device placement of its mask.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .vendor.dino_vit import ViTPredictor
from .cem import CEMSolver, verify_solver
from types import SimpleNamespace
from gymnasium.spaces import Box
import numpy as np


class ActionEncoder(nn.Module):
    def __init__(self, width=256, depth=2, heads=4, maximum=70):
        super().__init__()
        self.input = nn.Linear(4,width)
        self.cls = nn.Parameter(torch.randn(1,1,width)*.02)
        self.position = nn.Parameter(torch.randn(1,maximum+1,width)*.02)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,4*width,
            dropout=.1,activation='gelu',batch_first=True,norm_first=True),depth,
            norm=nn.LayerNorm(width),enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,4))

    def forward(self, actions, lengths):
        x = torch.cat((self.cls.expand(len(actions),-1,-1),self.input(actions)),1)
        mask = torch.arange(x.shape[1],device=x.device)[None] > lengths[:,None]
        return self.head(self.transformer(x+self.position[:,:x.shape[1]],src_key_padding_mask=mask)[:,0])


class HWM(nn.Module):
    def __init__(self,c):
        super().__init__()
        cfg=c['baseline']['hwm']; d=c['encoder']['dim']; self.context=cfg['waypoints']-1
        self.action_encoder=ActionEncoder(**cfg['action_encoder'])
        # Input/output projections adapt the shared1024-D features to the paper's
        #768-D predictor; ten action channels follow DINO-WM's released setting.
        self.visual=nn.Linear(d,758)
        self.macro=nn.Linear(4,10)
        self.predictor=ViTPredictor(num_patches=32,num_frames=self.context,dim=768,
            depth=10,heads=12,mlp_dim=3072,dropout=.1,emb_dropout=0.,pool='mean')
        self.output=nn.Linear(768,d)
        self.register_buffer('action_mean',torch.zeros(8))
        self.register_buffer('action_std',torch.ones(8))
        self.calls=0

    def predict(self,z,macro):
        b,t,p,_=z.shape
        assert t<=self.context and macro.shape==(b,t,4)
        m=self.macro(macro)[:,:,None].expand(-1,-1,p,-1)
        x=torch.cat((self.visual(z),m),-1).flatten(1,2)
        return self.output(self.predictor(x)).reshape(b,t,p,-1)

    def forward(self,batch,world=None):
        z,actions,lengths=batch['z'],batch['a'],batch['lengths']
        b,t,l,d=actions.shape
        normalized=(actions.reshape(b,t,l//2,8)-self.action_mean)/self.action_std
        macro=self.action_encoder(normalized.reshape(b*t,l,4),lengths.flatten()).reshape(b,t,4)
        pred=self.predict(z[:,:-1],macro)
        loss=F.l1_loss(pred,z[:,1:])
        return loss,dict(teacher_forcing_l1=loss.detach(),
            macro_std=macro.flatten(0,1).std(0,unbiased=False).mean().detach(),
            prediction_std=pred.flatten(0,2).std(0,unbiased=False).mean().detach())

    def rollout(self,start,macro,budget=None):
        states=[start]; predictions=[]
        for i in range(macro.shape[1]):
            if budget is not None: budget.check()
            left=max(0,i+1-self.context)
            z=torch.stack(states[left:],1)
            pred=self.predict(z,macro[:,left:i+1])[:,-1]
            self.calls+=len(start)
            states.append(pred); predictions.append(pred)
        return torch.stack(predictions,1)


class HWMCost:
    def __init__(self,model,world,high):
        self.model,self.world,self.high=model,world,high
        self.budget=None

    def get_cost(self,info,actions):
        b,n,h,d=actions.shape
        start,goal=info['start'].flatten(0,1),info['goal'].flatten(0,1)
        actions=actions.reshape(b*n,h,d)
        if self.high: pred=self.model.rollout(start,actions,self.budget)[:,-1]
        else:
            actions=actions*self.model.action_std+self.model.action_mean
            pred=self.world.rollout(start,actions,budget=self.budget)[:,-1]
        return (pred-goal).abs().mean((1,2)).reshape(b,n)


class PaperCEM(CEMSolver):
    """Native CEM equations plus HWM Appendix C's standard-deviation momentum.

    This is confined to HWM. Momentum zero is exactly the released CEM update.
    No candidate clipping, variance floor, best-sample replacement or warm start.
    """
    def __init__(self,*args,momentum=0.,**kwargs):
        super().__init__(*args,**kwargs);self.momentum=momentum

    @torch.inference_mode()
    def solve(self,info_dict,init_action=None):
        mean,std=self.init_action_distrib(init_action)
        mean,std=mean.to(self.device),std.to(self.device)
        expanded={k:v[:,None].expand(-1,self.num_samples,*v.shape[1:]) for k,v in info_dict.items()}
        for _ in range(self.n_steps):
            samples=torch.randn(self.n_envs,self.num_samples,self.horizon,self.action_dim,
                generator=self.torch_gen,device=self.device)*std[:,None]+mean[:,None]
            samples[:,0]=mean
            costs=self.model.get_cost(expanded,samples)
            values,indices=costs.topk(self.topk,dim=1,largest=False)
            elites=samples[torch.arange(self.n_envs,device=self.device)[:,None],indices]
            mean=elites.mean(1)
            std=self.momentum*std+(1-self.momentum)*elites.std(1)
        return dict(actions=mean.cpu(),costs=values.mean(1).cpu().tolist())


class HWMController:
    def __init__(self,model,world,c):
        verify_solver()
        self.model,self.world,self.c=model,world,c
        cfg=c['baseline']['hwm']; self.execution_blocks=cfg['low']['receding_horizon']
        self.costs=[]; self.solvers=[]
        for high,key in [(True,'high'),(False,'low')]:
            f=cfg[key]
            cost=HWMCost(model,world,high)
            solver=PaperCEM(cost,momentum=f['var_ema'],batch_size=1,num_samples=f['candidates'],n_steps=f['iterations'],
                topk=f['elites'],var_scale=1.,device=next(model.parameters()).device,seed=3072)
            solver.configure(action_space=Box(-1.,1.,shape=(1,4),dtype=np.float32),n_envs=1,
                config=SimpleNamespace(horizon=f['horizon'],action_block=1 if high else 2))
            self.costs.append(cost);self.solvers.append(solver)
        self.initial_calls=0

    def begin_episode(self):
        for solver in self.solvers: solver.torch_gen.manual_seed(torch.initial_seed())
        self.initial_calls=self.model.calls

    @torch.no_grad()
    def plan(self,start,goal,*,budget=None):
        for cost in self.costs: cost.budget=budget
        high=self.solvers[0](dict(start=start,goal=goal))
        macro=high['actions'].to(start.device)
        # The first predicted state under the OPTIMIZED MEAN plan is the subgoal.
        # Do not replace it with retrieval or feed low-level scores into routing.
        subgoal=self.model.rollout(start,macro[:,:1],budget)[:,0]
        low=self.solvers[1](dict(start=start,goal=subgoal))
        actions=low['actions'][0].to(start.device)*self.model.action_std+self.model.action_mean
        return actions[:self.execution_blocks].flatten(),subgoal[0],float(low['costs'][0]),self.c['baseline']['hwm']['low']['horizon']

    def diagnostics(self):
        return dict(coarse_predictions=self.model.calls-self.initial_calls)
