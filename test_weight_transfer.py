import copy
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env
from g1_stair_env import G1StairEnv
from gait_metrics import GaitMetrics


class WeightTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=G1StairEnv(mode='flat')

    def setUp(self):
        self.e.reset()

    def test_api(self):
        check_env(self.e)
        self.assertEqual(self.e.observation_space.shape,(73,))
        self.assertEqual(self.e.action_space.shape,(29,))

    def test_reference_continuity_bounds_and_lateral_alternation(self):
        e=self.e
        refs=np.array([e.reference_pose(p) for p in np.linspace(0,1,1001)])
        self.assertLess(np.max(np.abs(np.diff(refs,axis=0))),.04)
        np.testing.assert_allclose(refs[0],refs[-1],atol=1e-9)
        self.assertTrue(np.all(refs>=e.joint_ranges[:,0]))
        self.assertTrue(np.all(refs<=e.joint_ranges[:,1]))
        self.assertAlmostEqual(e._trajectory(.25)[1],-.06)
        self.assertAlmostEqual(e._trajectory(.75)[1],.06)

    def test_force_gate_both_halves_and_wrap(self):
        e=self.e
        for phase in [.159,.659]:
            e.phase=phase
            e.data.time=1
            e.foot_state['forces'][:]=e.body_weight/2
            with patch.object(mujoco,'mj_step'),patch.object(mujoco,'mj_forward'):
                e.step(np.zeros(29))
            self.assertAlmostEqual(e.phase,.16 if phase<.5 else .66)
            self.assertTrue(e.phase_waiting)
            sw=0 if phase<.5 else 1
            e.foot_state['forces'][sw]=0
            e.foot_state['forces'][1-sw]=e.body_weight
            with patch.object(mujoco,'mj_step'),patch.object(mujoco,'mj_forward'):
                e.step(np.zeros(29))
            self.assertGreater(e.phase,.16 if phase<.5 else .66)
        e.phase=.999
        with patch.object(mujoco,'mj_step'),patch.object(mujoco,'mj_forward'):
            e.step(np.zeros(29))
        self.assertLess(e.phase,.01)

    def test_residual_and_clipping(self):
        e=self.e
        e._apply_action(np.zeros(29)); base=e.target_q.copy()
        e._apply_action(np.ones(29))
        np.testing.assert_allclose(e.target_q-base,.05,atol=1e-8)
        e.data.qvel[e.joint_qvel]=1e4
        e._apply_action(np.zeros(29))
        np.testing.assert_allclose(e.data.ctrl,e.torque_limits[:,0])

    def test_crouch_excludes_swing_knee(self):
        self.assertLess(self.e.crouch_penalty([.8,.9],[True,True]),0)
        self.assertEqual(self.e.crouch_penalty([.8,.3],[True,True]),0)
        self.assertEqual(self.e.crouch_penalty([.8,.9],[True,False]),0)

    def test_force_units_and_slip(self):
        e=self.e
        for _ in range(50):e.step(np.zeros(29))
        self.assertLess(abs(sum(e.foot_state['forces'])-e.body_weight),.2*e.body_weight)
        e.data.qvel[:]=0
        e.data.qvel[0]=1
        mujoco.mj_forward(e.model,e.data)
        f=e._read_feet()
        np.testing.assert_allclose(f['speeds'],[1,1],atol=1e-6)
        np.testing.assert_allclose(f['slip'],[1,1],atol=1e-6)

    def test_valid_steps_require_replant_and_advance(self):
        i=self.e._info()
        i.update(upright=1,torso_upright=1)
        for s in ['left','right']:
            i.update({f'{s}_foot_contact':True,f'{s}_foot_height':0,f'{s}_foot_x':0,
                      f'{s}_contact_force':i['body_weight']/2})
        m=GaitMetrics(i); m.update(i)
        for s,o in [('left','right'),('right','left')]:
            i['swing_leg']=s; i['stance_leg']=o
            i.update({f'{s}_foot_contact':False,f'{s}_foot_height':.03,f'{s}_foot_x':.04,
                      f'{s}_contact_force':0,f'{o}_contact_force':i['body_weight']})
            m.update(i)
            before=len(m.events)
            self.assertEqual(before,0 if s=='left' else 1)
            i.update({f'{s}_foot_contact':True,f'{s}_foot_height':0,f'{s}_contact_force':i['body_weight']/2})
            for _ in range(3):m.update(i)
        self.assertEqual(m.summary()['alternating_step_pairs'],1)
        self.assertEqual(m.summary()['valid_left_steps'],1)
        self.assertEqual(m.summary()['valid_right_steps'],1)
        # A lifted foot that returns to its original x does not make a valid step.
        m=GaitMetrics(i); m.update(i)
        i.update(swing_leg='left',stance_leg='right',left_foot_contact=False,left_foot_height=.03,
                 left_contact_force=0,right_contact_force=i['body_weight'])
        m.update(i)
        i.update(left_foot_contact=True,left_foot_height=0,left_contact_force=i['body_weight']/2)
        for _ in range(3):m.update(i)
        self.assertEqual(len(m.events),0)

if __name__=='__main__':unittest.main(verbosity=2)
