from __future__ import annotations

import pickle
import numpy as np


def make_env(task, seed, c):
    from metaworld import Task
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS
    name = task + '-v3'
    cls = ALL_V3_ENVIRONMENTS[name]
    env = cls(render_mode='rgb_array', camera_name=c['camera'],
              height=c['render_size'], width=c['render_size'])
    rng = np.random.default_rng(seed)
    # Same public Task payload used by MetaWorld's benchmark generators.
    rand_vec = rng.uniform(env._random_reset_space.low, env._random_reset_space.high)
    payload = dict(rand_vec=rand_vec, env_cls=cls, partially_observable=False)
    env.set_task(Task(name, pickle.dumps(payload)))
    env.seed(seed)
    return env


def collect_episode(task, seed, c, mode='expert'):
    from metaworld.policies import ENV_POLICY_MAP
    env = make_env(task, seed, c)
    policy = ENV_POLICY_MAP[task + '-v3']()
    rng = np.random.default_rng(seed)
    frames, actions, rewards, successes = [], [], [], []
    try:
        obs, _ = env.reset()
        frames.append(env.render().copy())
        for _ in range(c['primitive_budget']):
            action = policy.get_action(obs) if mode == 'expert' else rng.uniform(-1, 1, 4)
            action = np.clip(action, -1, 1).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            successes.append(bool(info['success']))
            frames.append(env.render().copy())
            if terminated or truncated:
                break
        # The collection remains intact even when the expert fails.
        good = np.flatnonzero(successes)
        goal_index = int(good[-1] + 1) if len(good) else len(frames)-1
        return dict(frames=np.stack(frames), actions=np.stack(actions),
                    rewards=np.asarray(rewards), successes=np.asarray(successes),
                    goal_index=goal_index, expert_success=bool(len(good)))
    finally:
        env.close()
