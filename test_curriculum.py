"""Checks for the shared environment before committing to a long training run."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

import g1_pd_env_v2 as g1_stair_env
from g1_pd_env_v2 import G1StairEnv


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.env = G1StairEnv(mode="flat")
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def test_flat_scene_and_api(self):
        # A flat environment must never invoke staircase generation.
        with patch.object(g1_stair_env, "create_stair_scene", side_effect=AssertionError):
            flat = G1StairEnv(mode="flat")
            try:
                self.assertEqual(flat.scene_path, g1_stair_env.BASE_SCENE)
                self.assertEqual(flat.observation_space.shape, (71,))
                self.assertEqual(flat.action_space.shape, (29,))
                self.assertEqual(mujoco.mj_name2id(flat.model, mujoco.mjtObj.mjOBJ_GEOM, "hackathon_step_1"), -1)
                check_env(flat, warn=True)
            finally:
                flat.close()

    def test_repeated_resets_and_random_step_metrics(self):
        self.env.action_space.seed(7)
        for _ in range(3):
            obs, info = self.env.reset()
            self.assertEqual(info["episode_step"], 0)
            self.assertEqual(info["forward_distance"], 0.0)
            np.testing.assert_array_equal(self.env.previous_action, np.zeros(29))
            obs, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(info["episode_step"], 1)
            self.assertEqual(info["forward_distance"], info["x"] - self.env.initial_x)
            self.assertTrue({"x", "y", "height", "forward_velocity", "upright", "fallen", "forward_distance", "episode_step"}.issubset(info))

    def test_pd_targets_damping_and_limits(self):
        action = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
        self.env._apply_action(action)
        target = np.clip(self.env.nominal_joint_positions + 0.25 * action,
                         self.env.joint_ranges[:, 0], self.env.joint_ranges[:, 1])
        np.testing.assert_allclose(self.env.target_q, target)
        np.testing.assert_allclose(self.env.data.ctrl, self.env.kp * (target - self.env.nominal_joint_positions))
        self.env.data.qvel[self.env.joint_qvel] = 10000
        self.env._apply_action(np.zeros(29))
        np.testing.assert_allclose(self.env.data.ctrl, self.env.torque_limits[:, 0])
        self.env.data.qvel[self.env.joint_qvel] = -10000
        self.env._apply_action(np.zeros(29))
        np.testing.assert_allclose(self.env.data.ctrl, self.env.torque_limits[:, 1])

    def test_pd_updated_every_physics_step(self):
        with patch.object(self.env, '_apply_action', wraps=self.env._apply_action) as apply:
            self.env.step(np.zeros(29))
            self.assertEqual(apply.call_count, self.env.frame_skip)

    def test_flat_reward_favors_motion_without_height_reward(self):
        zero = np.zeros(29, dtype=np.float32)
        dt = self.env.model.opt.timestep * self.env.frame_skip
        with patch.object(self.env, "_terrain_height", side_effect=AssertionError):
            resting = self.env._reward_terms(0.0, 0.0, 0.0, 0.79, 1.0, zero, False)
            walking = self.env._reward_terms(0.4, 0.4 * dt, 0.0, 0.79, 1.0, zero, False)
            different_height = self.env._reward_terms(0.4, 0.4 * dt, 0.0, 0.6, 1.0, zero, False)
        self.assertEqual(walking, different_height)
        self.assertNotIn("height", walking)
        self.assertGreater(sum(walking.values()), 6 * sum(resting.values()))
        self.assertAlmostEqual(walking["velocity"], 4.0)
        self.assertAlmostEqual(resting["velocity"], 4.0 * np.exp(-3.2))

    def test_falls_and_time_limit(self):
        # Disable physics advancement to isolate the two termination conditions.
        zero = np.zeros(29, dtype=np.float32)
        with patch.object(mujoco, "mj_step"):
            self.env.data.qpos[2] = self.env.fall_height - 0.01
            _, _, terminated, _, info = self.env.step(zero)
            self.assertTrue(terminated)
            self.assertEqual(info["reward_terms"]["fall"], -30.0)
            self.env.reset()
            self.env.data.qpos[3:7] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
            _, _, terminated, _, info = self.env.step(zero)
            self.assertTrue(terminated)
            self.assertLess(info["upright"], 0.45)
            self.env.reset()
            self.env.current_step = self.env.max_episode_steps - 1
            _, _, terminated, truncated, info = self.env.step(zero)
            self.assertFalse(terminated)
            self.assertTrue(truncated)

    def test_stairs_preserved_and_invalid_mode_rejected(self):
        stairs = G1StairEnv(mode="stairs")
        try:
            self.assertEqual(stairs.observation_space.shape, (71,))
            self.assertEqual(stairs.action_space.shape, (29,))
            for index in range(1, 6):
                self.assertGreaterEqual(mujoco.mj_name2id(stairs.model, mujoco.mjtObj.mjOBJ_GEOM, f"hackathon_step_{index}"), 0)
            stairs.reset()
            _, reward, _, _, info = stairs.step(np.zeros(29, dtype=np.float32))
            self.assertTrue(np.isfinite(reward))
            self.assertIn("height", info["reward_terms"])
        finally:
            stairs.close()
        with self.assertRaises(ValueError):
            G1StairEnv(mode="invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)

