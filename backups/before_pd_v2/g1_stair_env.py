"""Shared torque-controlled G1 environment for flat walking and stairs."""

from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
G1_FOLDER = PROJECT_DIR / "unitree_mujoco" / "unitree_robots" / "g1"
BASE_SCENE = G1_FOLDER / "scene_29dof.xml"
STAIR_SCENE = G1_FOLDER / "scene_hackathon_stairs.xml"

# Retain the original staircase for later curriculum stages.
number_of_steps = 5
step_length = 0.35
step_height = 0.10
step_width = 1.20
stair_start_x = 1.2


def create_stair_scene():
    """Build from the base scene only when stairs are explicitly needed."""
    tree = ET.parse(BASE_SCENE)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise RuntimeError("The G1 scene has no worldbody element.")
    for i in range(number_of_steps):
        full_height = step_height * (i + 1)
        ET.SubElement(worldbody, "geom", {
            "name": f"hackathon_step_{i + 1}",
            "type": "box",
            "size": f"{step_length / 2} {step_width / 2} {full_height / 2}",
            "pos": f"{stair_start_x + i * step_length} 0 {full_height / 2}",
            "friction": "1.0 0.005 0.0001",
            "rgba": "0.55 0.55 0.55 1",
        })
    tree.write(STAIR_SCENE)
    return STAIR_SCENE


class G1StairEnv(gym.Env):
    # Default preserves compatibility with existing staircase scripts/models.
    def __init__(self, mode="stairs"):
        super().__init__()
        if mode not in ("flat", "stairs"):
            raise ValueError("mode must be 'flat' or 'stairs'")
        self.mode = mode
        self.scene_path = BASE_SCENE if mode == "flat" else create_stair_scene()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.model.nq + self.model.nv,), np.float32
        )
        self.frame_skip = 5
        self.max_episode_steps = 1000
        self.current_step = 0
        self.initial_height = None
        self.fall_height = None
        self.initial_x = 0.0
        self.previous_action = np.zeros(self.model.nu, dtype=np.float32)
        self.nominal_qpos = None
        print(f"G1 mode={mode}: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")

    def _get_obs(self):
        # Preserve the exact 71-value observation layout used by older models.
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _get_position(self):
        return tuple(float(value) for value in self.data.qpos[:3])

    def _upright_score(self):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self.data.qpos[3:7])
        return float(mat.reshape(3, 3)[2, 2])

    def _terrain_height(self, x):
        if self.mode == "flat" or x < stair_start_x:
            return 0.0
        step_index = min(int((x - stair_start_x) / step_length) + 1, number_of_steps)
        return step_index * step_height

    def _apply_action(self, action):
        # Keep the original torque control: no position targets or PD controller.
        ranges = self.model.actuator_ctrlrange
        max_abs = np.maximum(np.abs(ranges[:, 0]), np.abs(ranges[:, 1]))
        self.data.ctrl[:] = action * max_abs * 0.20

    def _info(self, forward_velocity=0.0, fallen=False):
        x, y, z = self._get_position()
        return {
            "x": x,
            "y": y,
            "height": z,
            "forward_velocity": float(forward_velocity),
            "upright": self._upright_score(),
            "fallen": bool(fallen),
            "forward_distance": x - self.initial_x,
            "episode_step": self.current_step,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.previous_action[:] = 0.0
        self.initial_x, _, z = self._get_position()
        if self.nominal_qpos is None:
            self.nominal_qpos = self.data.qpos.copy()
        if self.initial_height is None:
            self.initial_height = z
            self.fall_height = z * 0.55
        return self._get_obs(), self._info()

    def _reward_terms(self, forward_velocity, delta_x, y, z, upright, action, fallen):
        # Preserve penalties and their scales across both modes.
        terms = {
            "lateral": -0.5 * abs(y),
            "effort": -0.00005 * float(np.sum(np.square(self.data.ctrl))),
            "action_change": -0.05 * float(np.mean(np.square(action - self.previous_action))),
        }
        if self.mode == "flat":
            # At rest the velocity term is only 0.163; at 0.4 m/s it is 4.0.
            # Upright contributes at most 0.5, so posture alone is not the goal.
            # Signed progress rewards forward motion and penalizes moving back.
            # No alive bonus, body-height reward, or terrain reward in flat mode.
            terms.update(
                velocity=4.0 * float(np.exp(-20.0 * (forward_velocity - 0.4) ** 2)),
                upright=0.5 * upright,
                progress=10.0 * delta_x,
                fall=-30.0 if fallen else 0.0,
            )
        else:
            # Preserve the previous staircase reward; no stair training yet.
            target_height = self.initial_height + self._terrain_height(self.data.qpos[0])
            terms.update(
                velocity=1.5 * float(np.exp(-2.0 * (forward_velocity - 0.4) ** 2)),
                upright=2.0 * upright,
                height=1.5 * float(np.exp(-8.0 * abs(z - target_height))),
                progress=2.0 * delta_x,
                fall=-20.0 if fallen else 0.0,
            )
        return terms

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Expected a finite action with shape (29,).")
        if self.initial_height is None:
            raise RuntimeError("Call reset() before step().")
        self.current_step += 1
        x_before, _, _ = self._get_position()
        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x, y, z = self._get_position()
        delta_x = x - x_before
        forward_velocity = delta_x / (self.model.opt.timestep * self.frame_skip)
        upright = self._upright_score()
        fallen = z < self.fall_height or upright < 0.45
        terms = self._reward_terms(forward_velocity, delta_x, y, z, upright, action, fallen)
        self.previous_action = action.copy()
        info = self._info(forward_velocity, fallen)
        info["reward_terms"] = terms
        truncated = self.current_step >= self.max_episode_steps
        return self._get_obs(), float(sum(terms.values())), bool(fallen), truncated, info
