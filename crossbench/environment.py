"""Pinned native PushT/Cube reset and success adapters (no learned success proxy)."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from PIL import Image

PIN='06b1f4c24ccb237a9b2d6bb65bf010a6a5b41e83'


def configure_renderer_device():
    """Call before importing SWM/MuJoCo, which can initialize EGL on import."""
    if os.getenv('LIBGL_ALWAYS_SOFTWARE')=='1':
        os.environ['MUJOCO_EGL_DEVICE_ID']='0'
        return
    visible=os.getenv('CUDA_VISIBLE_DEVICES','').split(',')
    local=int(os.getenv('LOCAL_RANK','0'))
    if os.getenv('MUJOCO_GL','egl')=='egl' and local<len(visible) and visible[local].isdigit():
        os.environ['MUJOCO_EGL_DEVICE_ID']=visible[local]


def verify_install():
    import importlib.metadata
    dist=importlib.metadata.distribution('stable-worldmodel')
    raw=dist.read_text('direct_url.json')
    if not raw or json.loads(raw).get('vcs_info',{}).get('commit_id') != PIN:
        raise RuntimeError(f'Install the pinned stable-worldmodel commit {PIN}; refusing an unverified simulator API')


class NativeEnv:
    def __init__(self, dataset, action_budget=50, verify_version=True):
        if verify_version:verify_install()
        # EGL enumerates physical GPUs independently of CUDA's logical indices.
        # Keep each simulator renderer on its own evaluation rank's GPU.
        configure_renderer_device()
        import gymnasium as gym
        import stable_worldmodel.envs  # registers the pinned tasks
        from omegaconf import OmegaConf
        path=Path(__file__).resolve().parents[1]/'config'/'eval'/f'{dataset}.yaml'
        cfg=OmegaConf.to_container(OmegaConf.load(path),resolve=False)
        wc=cfg['world'];name=wc['env_name']
        kwargs={k:v for k,v in wc.items() if k not in
                ('env_name','num_envs','history_size','frame_skip','max_episode_steps')}
        if dataset=='pusht':kwargs['render_mode']='rgb_array'
        if dataset not in ('pusht','cube'):raise ValueError(dataset)
        self.env=gym.make(name,max_episode_steps=max(1000,action_budget+1),**kwargs)
        self.dataset=dataset;self.callables=cfg['eval']['callables']
        self.action_space=self.env.action_space
        self.reset_audit={}

    def render(self):
        render=getattr(self.env.unwrapped,'render_multiview',None)
        img=render() if callable(render) else self.env.render()
        if not isinstance(img,np.ndarray) or img.ndim!=3 or img.shape[-1]!=3:
            raise ValueError('Expected one RGB camera, not multiview output')
        return np.asarray(Image.fromarray(img).resize((224,224),Image.Resampling.BILINEAR)).copy()

    def reset_from_source(self, source, row):
        start=row['source_offset']+row['start'];end=start+row['goal_offset']
        seed=int(np.asarray(source['seed'][start]).item()) if 'seed' in source else row['seed']
        variation={key.removeprefix('variation.'):source[key][start].copy()
                   for key in source if key.startswith('variation.')}
        options={'variation':list(variation),'variation_values':variation} if variation else {}
        self.env.action_space.seed(seed)
        self.env.reset(seed=seed,options=options)
        for entry in self.callables:
            fn=getattr(self.env.unwrapped,entry['method'],None)
            if fn is None:raise ValueError('Native reset callable missing: '+entry['method'])
            args={}
            for name,spec in entry['args'].items():
                value=spec['value']
                if not spec.get('in_dataset',True):args[name]=value;continue
                goal=value.startswith('goal_');key=value[5:] if goal else value
                if key not in source:raise ValueError('Missing native reset dataset column: '+key)
                args[name]=np.asarray(source[key][end if goal else start]).copy()
            fn(**args)
        actual=self.render()
        initial=np.asarray(source['pixels'][start]).copy();goal=np.asarray(source['pixels'][end]).copy()
        self.reset_audit=dict(render_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            recorded_initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),seed=seed,
            first_observation='recorded_initial_pixels_as_in_native_evaluate_from_dataset')
        # The native SWM dataset evaluator also uses the recorded first image;
        # every subsequent observation is rendered after actual physics execution.
        return initial,goal

    def step(self, action):
        if not np.isfinite(action).all():raise FloatingPointError('Nonfinite physical action')
        action=np.asarray(action,dtype=self.action_space.dtype)
        if not self.action_space.contains(action):raise ValueError('Physical action outside native bounds')
        _,reward,terminated,truncated,info=self.env.step(action)
        if self.dataset=='pusht':
            # Pinned PushT.step returns eval_state(goal_state,state)[0] as terminated.
            success=bool(terminated)
        else:
            if 'success' not in info:
                raise ValueError('Pinned Cube environment omitted its native success flag')
            success=bool(info['success'])
        return self.render(),float(reward),success,bool(terminated or truncated)

    def close(self):
        self.env.close()
