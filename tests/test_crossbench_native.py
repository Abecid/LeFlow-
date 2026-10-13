"""Native physics checks, enabled explicitly in the dependency-complete CI job."""
import os
import unittest
import numpy as np


@unittest.skipUnless(os.getenv('CROSSBENCH_NATIVE_TEST')=='1','requires pinned native simulator dependencies')
class TestNative(unittest.TestCase):
    def test_actual_bounds_reset_render_and_success(self):
        from crossbench.environment import NativeEnv
        for task,pd in [('pusht',2),('cube',5)]:
            with self.subTest(task=task):
                env=NativeEnv(task)
                try:
                    env.env.action_space.seed(3072);env.env.reset(seed=3072)
                    base=env.env.unwrapped
                    self.assertEqual(env.action_space.shape,(pd,))
                    image=env.render()
                    source={'pixels':np.stack([image,image]),'seed':np.array([3072,3072])}
                    if task=='pusht':
                        state=base._get_obs().copy()
                        source['state']=np.stack([state,state])
                        # Pinned SWM PushT uses normalized relative deltas, not pixel coordinates.
                        action=np.zeros(pd,dtype=env.action_space.dtype)
                    else:
                        qpos=base._data.qpos.copy();qvel=base._data.qvel.copy()
                        obj=base._data.joint('object_joint_0').qpos.copy()
                        source.update(qpos=np.stack([qpos,qpos]),qvel=np.stack([qvel,qvel]),
                            privileged_block_0_pos=np.stack([obj[:3],obj[:3]]),
                            privileged_block_0_quat=np.stack([obj[3:],obj[3:]]))
                        action=np.zeros(pd,dtype=env.action_space.dtype)
                    row=dict(source_offset=0,start=0,goal_offset=1,seed=3072)
                    env.reset_from_source(source,row);h1=env.reset_audit['render_sha256']
                    env.reset_from_source(source,row);h2=env.reset_audit['render_sha256']
                    self.assertEqual(h1,h2)
                    rgb,reward,success,done=env.step(action)
                    self.assertEqual(rgb.shape,(224,224,3));self.assertTrue(np.isfinite(reward))
                    self.assertTrue(success,'Self-goal should satisfy the native task predicate')
                    print('NATIVE_VERIFIED',task,pd,'reset/render/success',flush=True)
                finally:env.close()


if __name__=='__main__':unittest.main()
