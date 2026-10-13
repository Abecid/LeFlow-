"""Image-goal DDPM / deterministic DDIM action-chunk policy.

Adaptation of Diffusion Policy (Chi et al.). Frozen V-JEPA2.1 observations,
ten primitive actions grouped as [5,8], conditional temporal 1-D UNet,
100-step DDPM epsilon training and 16-step DDIM inference.
Not the authors' pretrained visual backbone or task protocol.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class FiLMResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8,channels),nn.SiLU(),
            nn.Conv1d(channels,channels,3,padding=1),
            nn.GroupNorm(8,channels),nn.SiLU(),
            nn.Conv1d(channels,channels,3,padding=1))
        self.cond = nn.Linear(cond_dim,2*channels)

    def forward(self,x,condition):
        scale,bias=self.cond(condition).chunk(2,dim=-1)
        return x+self.net(x)*(1+scale[:,:,None])+bias[:,:,None]


def cosine_alpha_bar(steps, device=None):
    s=.008
    ticks=torch.linspace(0,steps,steps+1,dtype=torch.float64,device=device)
    cum=(torch.cos(((ticks/steps+s)/(1+s))*math.pi/2)**2)
    cum=cum/cum[0]
    betas=(1-cum[1:]/cum[:-1]).clamp(.0001,.999)
    return torch.cumprod(1-betas,dim=0).float()


class DenoiseUNet(nn.Module):
    def __init__(self,dim=256,channels=128):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.head=nn.Conv1d(8,channels,3,padding=1)
        self.down_a=FiLMResBlock(channels,dim)
        self.down=nn.Conv1d(channels,2*channels,3,stride=2,padding=1)
        self.mid_a=FiLMResBlock(2*channels,dim)
        self.mid_b=FiLMResBlock(2*channels,dim)
        self.up=nn.Conv1d(2*channels,channels,3,padding=1)
        self.up_a=FiLMResBlock(channels,dim)
        self.out=nn.Sequential(nn.GroupNorm(8,channels),nn.SiLU(),
                               nn.Conv1d(channels,8,1))

    def forward(self,noisy,t,context):
        # noisy:[B,5,8], t:[B] normalized [0,1], context:[B,256].
        cond=context+self.time(t[:,None].float())
        x=self.down_a(self.head(noisy.transpose(1,2)),cond)
        middle=self.mid_b(self.mid_a(self.down(x),cond),cond)
        up=self.up(F.interpolate(middle,size=x.shape[-1],mode="nearest"))
        return self.out(self.up_a(up+x,cond)).transpose(1,2)


class DiffusionActionPolicy(nn.Module):
    def __init__(self,c):
        super().__init__()
        cfg=c["diffusion_policy_port"]
        width=int(cfg["width"])
        d=c["encoder"]["dim"]
        self.visual=nn.Linear(d,width)
        self.context=nn.Sequential(nn.Linear(2*width,width),nn.LayerNorm(width),nn.SiLU())
        self.net=DenoiseUNet(width,int(cfg["channels"]))
        self.steps=int(cfg["train_steps"])
        self.inference_steps=int(cfg["inference_steps"])
        if self.steps<2 or self.inference_steps<2 or self.inference_steps>self.steps:
            raise ValueError("Invalid diffusion schedule")
        self.register_buffer("alpha_bar",cosine_alpha_bar(self.steps))

    def condition(self,start,goal):
        a=self.visual(start).mean(dim=1)
        g=self.visual(goal).mean(dim=1)
        return self.context(torch.cat((a,g),dim=-1))

    def forward(self,batch,world=None,weight=0.,depth=None):
        z=batch["z"];clean=batch["a"]
        n=len(clean);noise=torch.randn_like(clean)
        t=torch.randint(0,self.steps,(n,),device=clean.device)
        ab=self.alpha_bar[t].to(clean.dtype)[:,None,None]
        noisy=ab.sqrt()*clean+(1-ab).sqrt()*noise
        pred=self.net(noisy,t.float()/(self.steps-1),self.condition(z[:,0],z[:,3]))
        loss=(pred-noise).square().mean()
        return loss,{"noise_mse":loss.detach()}

    @torch.no_grad()
    def sample(self,start,goal,count=1,budget=None,guidance=None):
        n=len(start)
        cond=self.condition(start,goal).repeat_interleave(count,dim=0)
        x=torch.randn(n*count,5,8,device=start.device,dtype=start.dtype)
        timesteps=torch.linspace(self.steps-1,0,self.inference_steps,device=start.device).long()
        for i,t in enumerate(timesteps.tolist()):
            if budget is not None: budget.check()
            tick=x.new_full((len(x),),t/(self.steps-1))
            epsilon=self.net(x,tick,cond)
            a=self.alpha_bar[t].to(x.dtype)
            clean=((x-(1-a).sqrt()*epsilon)/a.sqrt()).clamp(-1,1)
            if guidance is not None:
                clean=guidance(clean,i,self.inference_steps)
                if clean.shape!=x.shape:
                    raise ValueError("Guidance changed denoised action tensor shape")
                clean=clean.clamp(-1,1)
            if i+1<len(timesteps):
                a_next=self.alpha_bar[timesteps[i+1]].to(x.dtype)
                x=a_next.sqrt()*clean+(1-a_next).sqrt()*epsilon
            else:
                x=clean
        if budget is not None: budget.check()
        return x.reshape(n,count,5,8).clamp(-1,1)


class DiffusionPolicyController:
    execution_blocks=1

    def __init__(self,model,world,c,bank):
        self.model=model

    def begin_episode(self):
        pass

    @torch.no_grad()
    def plan(self,start,goal,*,budget=None):
        action=self.model.sample(start,goal,count=1,budget=budget)[0,0]
        return action[0],goal[0],0.,5
