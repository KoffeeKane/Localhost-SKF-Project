"""Flat residual gait learning; retain the v2 environment for staircase mode."""
import mujoco
import numpy as np
from gymnasium import spaces
from g1_pd_env_v2 import G1StairEnv as PDEnv


class G1StairEnv(PDEnv):
    def __init__(self, mode="stairs"):
        super().__init__(mode)
        self.gait_period = 1.2  # seconds per left/right cycle: 100 steps/minute
        self.phase = 0.0
        self.gait_enabled = mode == "flat"
        if not self.gait_enabled:
            return
        self.action_scale = 0.10
        self.observation_space = spaces.Box(-np.inf, np.inf, (73,), np.float32)
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                            for j in self.model.actuator_trnid[:, 0]]
        self.joint_indices = {name: i for i, name in enumerate(self.joint_names)}
        for side in ("left", "right"):
            for part in ("hip_pitch", "knee", "ankle_pitch"):
                if f"{side}_{part}_joint" not in self.joint_indices:
                    raise ValueError(f"Missing required {side} {part} joint")
        self.foot_bodies = [self._body_id(f"{side}_ankle_roll_link") for side in ("left", "right")]
        self.torso_body = self._body_id("torso_link")
        self.pelvis_body = self._body_id("pelvis")
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_geom < 0:
            raise ValueError("Missing floor geom")
        self.foot_geoms = []
        for body in self.foot_bodies:
            ids = np.flatnonzero((self.model.geom_bodyid == body) &
                                ((self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)))
            if len(ids) != 4 or not np.all(self.model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_SPHERE):
                raise ValueError("Expected four inspected sole collision spheres per foot")
            self.foot_geoms.append(ids)
        self.geom_to_foot = {int(g): i for i, ids in enumerate(self.foot_geoms) for g in ids}
        self.foot_state = None

    def _body_id(self, name):
        result = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if result < 0:
            raise ValueError(f"Missing inspected body {name}")
        return result

    def _get_obs(self):
        obs = super()._get_obs()
        if not self.gait_enabled:
            return obs
        angle = 2 * np.pi * self.phase
        return np.concatenate((obs, [np.sin(angle), np.cos(angle)])).astype(np.float32)

    def reference_pose(self, phase=None):
        q = self.nominal_joint_positions.copy()
        if not self.gait_enabled:
            return q
        p = self.phase if phase is None else phase
        for side, shift in (("left", 0.0), ("right", 0.5)):
            wave = np.sin(2 * np.pi * (p + shift))
            lift = max(0.0, wave) ** 2  # C1 at lift-off and touchdown
            # More flexion on swing; keep stance close to the standing pose.
            hip = -0.08 * wave - 0.27 * lift
            knee = 0.40 * lift
            q[self.joint_indices[f"{side}_hip_pitch_joint"]] += hip
            q[self.joint_indices[f"{side}_knee_joint"]] += knee
            q[self.joint_indices[f"{side}_ankle_pitch_joint"]] -= hip + knee
        return np.clip(q, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def _apply_action(self, action):
        if not self.gait_enabled:
            return super()._apply_action(action)
        self.target_q = np.clip(self.reference_pose() + self.action_scale * np.clip(action, -1, 1),
                                self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        torque = self.kp * (self.target_q - self.data.qpos[self.joint_qpos])
        torque -= self.kd * self.data.qvel[self.joint_qvel]
        self.data.ctrl[:] = np.clip(torque, self.torque_limits[:, 0], self.torque_limits[:, 1])

    def _read_feet(self):
        positions = self.data.xpos[self.foot_bodies].copy()
        heights = np.array([np.min(self.data.geom_xpos[ids, 2] - self.model.geom_size[ids, 0])
                            for ids in self.foot_geoms])
        contact = np.zeros(2, dtype=bool)
        slip = np.zeros(2)
        jac = np.zeros((3, self.model.nv))
        for c in self.data.contact:
            if c.dist > 0.001:
                continue
            other = int(c.geom2) if c.geom1 == self.floor_geom else int(c.geom1) if c.geom2 == self.floor_geom else -1
            if other in self.geom_to_foot:
                i = self.geom_to_foot[other]
                contact[i] = True
                mujoco.mj_jac(self.model, self.data, jac, None, c.pos, self.foot_bodies[i])
                point_v = jac @ self.data.qvel
                slip[i] = max(slip[i], float(np.dot(point_v[:2], point_v[:2])))
        mat = self.data.xmat[self.torso_body].reshape(3, 3)
        root_mat = self.data.xmat[self.pelvis_body].reshape(3, 3)
        yaw = np.arctan2(root_mat[1, 0], root_mat[0, 0])
        rel = positions - self.data.qpos[:3]
        rel_x = rel[:, 0] * np.cos(yaw) + rel[:, 1] * np.sin(yaw)
        return dict(positions=positions, heights=heights, contact=contact, slip=slip,
                    relative_x=rel_x, torso_upright=float(mat[2, 2]),
                    torso_pitch=float(np.arctan2(-mat[2, 0], np.hypot(mat[0, 0], mat[1, 0]))))

    def _info(self, forward_velocity=0.0, fallen=False):
        info = super()._info(forward_velocity, fallen)
        if not self.gait_enabled or self.foot_state is None:
            return info
        f = self.foot_state
        info.update(phase=float(self.phase), swing_leg="left" if self.phase < 0.5 else "right",
                    torso_upright=f["torso_upright"], torso_pitch=f["torso_pitch"])
        for i, side in enumerate(("left", "right")):
            info.update({f"{side}_foot_height": float(f["heights"][i]),
                         f"{side}_foot_x": float(f["positions"][i, 0]),
                         f"{side}_foot_position": f["positions"][i].tolist(),
                         f"{side}_foot_relative_x": float(f["relative_x"][i]),
                         f"{side}_foot_contact": bool(f["contact"][i]),
                         f"{side}_foot_slip_speed": float(np.sqrt(f["slip"][i]))})
        return info

    def reset(self, seed=None, options=None):
        self.phase = 0.0
        super().reset(seed=seed, options=options)
        if self.gait_enabled:
            self.foot_state = self._read_feet()
        return self._get_obs(), self._info()

    def _reward_terms(self, forward_velocity, delta_x, y, z, upright, action, fallen):
        if not self.gait_enabled:
            return super()._reward_terms(forward_velocity, delta_x, y, z, upright, action, fallen)
        f = self.foot_state
        swing = 0 if self.phase < 0.5 else 1
        stance = 1 - swing
        gate = np.sin(2 * np.pi * self.phase) ** 2
        clearance = np.clip(f["heights"][swing] / 0.05, 0, 1)
        support = float(f["contact"][stance]) * np.exp(-((f["heights"][stance] / 0.025) ** 2))
        forward = np.clip(f["relative_x"][swing] / 0.12, 0, 1)
        pose_error = np.mean((self.data.qpos[self.joint_qpos] - self.reference_pose()) ** 2)
        return dict(
            velocity=2.0 * float(np.exp(-20 * (forward_velocity - 0.4) ** 2)),
            progress=3.0 * float(np.clip(delta_x, -0.02, 0.02)),
            upright=0.5 * upright,
            torso=0.75 * f["torso_upright"],
            torso_pitch=-0.75 * f["torso_pitch"] ** 2,
            imitation=0.5 * float(np.exp(-10 * pose_error)),
            swing_lift=1.5 * gate * float(clearance) * support,
            swing_forward=1.0 * gate * float(forward * clearance) * support,
            stance=0.5 * gate * support,
            sliding=-0.5 * float(np.sum(np.minimum(f["slip"], 4.0))),
            lateral=-0.5 * abs(y),
            effort=-0.00005 * float(np.sum(self.data.ctrl ** 2)),
            action_change=-0.05 * float(np.mean((action - self.previous_action) ** 2)),
            fall=-30.0 if fallen else 0.0,
        )

    def step(self, action):
        if not self.gait_enabled:
            return super().step(action)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Expected 29 finite actions")
        if self.initial_height is None:
            raise RuntimeError("Call reset before step")
        action = np.clip(action, -1, 1)
        x_before = float(self.data.qpos[0])
        self.current_step += 1
        for _ in range(self.frame_skip):
            self._apply_action(action)
            mujoco.mj_step(self.model, self.data)
            self.phase = (self.phase + self.model.opt.timestep / self.gait_period) % 1.0
        # Refresh kinematics and contacts at the integrated state (not the prior substep).
        mujoco.mj_forward(self.model, self.data)
        self.foot_state = self._read_feet()
        x, y, z = self._get_position()
        dx = x - x_before
        velocity = dx / (self.model.opt.timestep * self.frame_skip)
        upright = self._upright_score()
        fallen = bool(z < self.fall_height or upright < 0.45)
        terms = self._reward_terms(velocity, dx, y, z, upright, action, fallen)
        self.previous_action = action.copy()
        info = self._info(velocity, fallen)
        info["reward_terms"] = terms
        return self._get_obs(), float(sum(terms.values())), fallen, self.current_step >= self.max_episode_steps, info
