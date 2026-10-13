"""Feedback World Model-inspired correction + action-aware diffusion guidance.

Paper: An et al. (2026), arXiv:2605.15705.
The authors' public GitHub is a project site, not a MetaWorld implementation.
This independently specified port keeps the same trained conditional DDPM as
DiffusionActionPolicy; it adds inference-only factual residual feedback and
counterfactual action-controllability-weighted gradient guidance.

No RL labels or hypothetical physical outcomes are used. This is NOT a
verbatim release of Feedback-WM's robotics implementation.
"""
from __future__ import annotations
import torch
from ..models import distance
from .diffusion_port import DiffusionActionPolicy


class FeedbackWorldPolicy(DiffusionActionPolicy):
    """Same learned diffusion parameters and training loss as DiffusionActionPolicy."""
    pass


class FeedbackWorldController:
    execution_blocks=1

    def __init__(self,model,world,c,bank):
        self.model,self.world=model,world
        self.cfg=c["feedback_wm_port"]
        self.begin_episode()

    def begin_episode(self):
        self._predicted_next=None
        self._feedback=None
        self._weights=None
        self.feedback_steps=0

    @torch.no_grad()
    def observe(self,actual):
        # Called only AFTER the controller's previously commanded prefix executes.
        if self._predicted_next is None:
            return
        residual=(actual-self._predicted_next).detach()
        decay=float(self.cfg["feedback_decay"])
        self._feedback=(residual if self._feedback is None
                        else decay*self._feedback+(1-decay)*residual)
        self._predicted_next=None
        self.feedback_steps+=1

    @torch.no_grad()
    def controllability(self,start,action,budget=None):
        """Estimate world-feature action sensitivity from local counterfactual probes.

        Each probe differs only in primitive actions; this uses NO observed
        outcome labels for the unexecuted probes. Weights are stop-gradient.
        """
        if budget is not None:budget.check()
        probes=int(self.cfg["probes"])
        offsets=torch.randn(probes,8,device=start.device,dtype=start.dtype)
        local=(action[None]+float(self.cfg["probe_sigma"])*offsets).clamp(-1,1)
        predicted=self.world(start.expand(probes,-1,-1),local)
        var=predicted.float().var(dim=0,unbiased=False)
        mean=var.mean()
        if float(mean)<1.e-12:
            return torch.ones_like(var)
        exponent=float(self.cfg["controllability_power"])
        weights=(var/(mean+1.e-12)+1.e-6).pow(exponent)
        weights=weights/(weights.mean()+1.e-12)
        return weights.clamp(max=10.).detach()

    @torch.no_grad()
    def plan(self,start,goal,*,budget=None):
        # Per-decision controllability weights are computed once, from the
        # current policy's denoised prefix, then reused for guided DDIM steps.
        self._weights=None
        last=int(self.cfg["guidance_last_steps"])
        feedback_gain=float(self.cfg["feedback_gain"])
        strength=float(self.cfg["guidance_strength"])
        max_rms=float(self.cfg["max_gradient_rms"])

        def guidance(clean,step,total):
            if step<total-last:
                return clean
            if budget is not None:budget.check()
            if self._weights is None:
                self._weights=self.controllability(start,clean[0,0].detach(),budget)
            # Stop-grad diffusion update and differentiate only through
            # the one-block frozen action-conditioned spatial world.
            with torch.enable_grad():
                plan=clean.detach().requires_grad_(True)
                raw=self.world(start.expand(len(plan),-1,-1),
                               plan[:,0].clamp(-1,1))
                corrected=raw
                if self._feedback is not None:
                    corrected=corrected+feedback_gain*self._feedback
                # Original spatial features, not task success labels.
                energy=(self._weights[None]*
                        (corrected-goal.expand_as(corrected)).square()).mean()
                gradient=torch.autograd.grad(energy,plan)[0].detach()
            rms=gradient.square().mean().sqrt().clamp_min(1.e-8)
            gradient=gradient*(max_rms/rms).clamp(max=1.)
            return (clean-strength*gradient).clamp(-1,1)

        plan=self.model.sample(start,goal,count=1,budget=budget,guidance=guidance)[0,0]
        first=plan[0]
        if budget is not None:budget.check()
        # This becomes factual feedback only when observe(actual) is called.
        self._predicted_next=self.world(start,first[None]).detach()
        predicted_goal_distance=distance(self._predicted_next,goal)
        return first,goal[0],float(predicted_goal_distance[0]),5
