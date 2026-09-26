import unittest
import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env
from g1_gait_env_v3 import G1StairEnv


class GaitTests(unittest.TestCase):
    def setUp(self):
        self.e = G1StairEnv(mode="flat")
        self.e.reset(seed=42)

    def tearDown(self):
        self.e.close()

    def test_api(self):
        check_env(self.e)
        self.assertEqual(self.e.action_space.shape, (29,))
        self.assertEqual(self.e.observation_space.shape, (73,))
        o, i = self.e.reset()
        np.testing.assert_allclose(o[-2:], [0, 1])
        o, r, _, _, i = self.e.step(self.e.action_space.sample())
        self.assertTrue(np.isfinite(o).all() and np.isfinite(r))
        self.assertGreater(i['phase'], 0)
        self.assertEqual(len(i['left_foot_position']), 3)

    def test_phase_wrap_and_alternation(self):
        self.e.phase = 0.499
        self.assertEqual(self.e.step(np.zeros(29))[-1]['swing_leg'], 'right')
        self.e.phase = 0.999
        i = self.e.step(np.zeros(29))[-1]
        self.assertLess(i['phase'], 0.01)
        self.assertEqual(i['swing_leg'], 'left')

    def test_reference_symmetry_continuity_and_bounds(self):
        e = self.e
        for p in np.linspace(0, 1, 1001):
            q = e.reference_pose(p)
            self.assertTrue(np.all(q - .10 >= e.joint_ranges[:, 0]))
            self.assertTrue(np.all(q + .10 <= e.joint_ranges[:, 1]))
        np.testing.assert_allclose(e.reference_pose(0), e.reference_pose(1), atol=1e-10)
        for part in ['hip_pitch', 'knee', 'ankle_pitch']:
            l, r = [e.joint_indices[f'{s}_{part}_joint'] for s in ['left', 'right']]
            self.assertAlmostEqual(e.reference_pose(.25)[l], e.reference_pose(.75)[r])
        self.assertGreater(e.reference_pose(.25)[e.joint_indices['left_knee_joint']], .3)

    def test_pd_clipping_and_substeps(self):
        from unittest.mock import patch
        e = self.e
        e.data.qvel[e.joint_qvel] = 1e4
        e._apply_action(np.ones(29))
        np.testing.assert_allclose(e.data.ctrl, e.torque_limits[:, 0])
        e.reset()
        with patch.object(e, '_apply_action', wraps=e._apply_action) as call:
            e.step(np.zeros(29))
            self.assertEqual(call.call_count, 5)
        self.assertTrue(np.isfinite(e.data.ctrl).all())

    def test_feet_contact_and_slide(self):
        e = self.e
        # Move the soles to floor level, then refresh collision detection.
        e.data.qpos[2] -= min(e.foot_state['heights']) + .001
        mujoco.mj_forward(e.model, e.data)
        f = e._read_feet()
        self.assertTrue(f['contact'].all())
        e.data.qvel[:2] = [1, 0]
        mujoco.mj_forward(e.model, e.data)
        f = e._read_feet()
        np.testing.assert_allclose(f['slip'], [1, 1], atol=1e-6)

    def test_stairs_unchanged(self):
        e = G1StairEnv(mode='stairs')
        try:
            self.assertEqual(e.reset()[0].shape, (71,))
            self.assertEqual(e.action_scale, .25)
            self.assertIn('height', e.step(np.zeros(29))[-1]['reward_terms'])
        finally:
            e.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)

