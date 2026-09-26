import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import gymnasium as gym

from gymnasium import spaces

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

UNITREE_MUJOCO = os.path.join(
    PROJECT_DIR,
    "unitree_mujoco"
)

G1_FOLDER = os.path.join(
    UNITREE_MUJOCO,
    "unitree_robots",
    "g1"
)

BASE_SCENE = os.path.join(
    G1_FOLDER,
    "scene_29dof.xml"
)

STAIR_SCENE = os.path.join(
    G1_FOLDER,
    "scene_hackathon_stairs.xml"
)

#Staircase settings
number_of_steps = 5
step_length = 0.35
step_height = 0.10
step_width = 1.20
stair_start_x = 1.2

#The scene that creates the scene_hackathon_stairs.xml
shutil.copy2(
    BASE_SCENE,
    STAIR_SCENE
)

tree = ET.parse(STAIR_SCENE)
root = tree.getroot()

worldbody = root.find("worldbody")

for i in range(number_of_steps):

    full_height = step_height * (i + 1)

    x = stair_start_x + i * step_length
    z = full_height / 2

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": f"hackathon_step_{i + 1}",
            "type": "box",
            "size": (
                f"{step_length / 2} "
                f"{step_width / 2} "
                f"{full_height / 2}"
            ),
            "pos": f"{x} 0 {z}",
            "friction": "1.0 0.005 0.0001",
            "rgba": "0.55 0.55 0.55 1"
        }
    )

tree.write(STAIR_SCENE)

#The entire G1StairEnv class:
class G1StairEnv(gym.Env):

    def __init__(self):

        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            STAIR_SCENE
        )

        self.data = mujoco.MjData(
            self.model
        )

        print("\nMuJoCo model:")
        print("nq =", self.model.nq)
        print("nv =", self.model.nv)
        print("nu =", self.model.nu)


        # ----------------------------------------------------
        # ACTION SPACE
        # ----------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )


        # ----------------------------------------------------
        # OBSERVATION SPACE
        #
        # Same structure used during training:
        #
        # qpos + qvel
        # ----------------------------------------------------

        obs_size = (
            self.model.nq
            + self.model.nv
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )


        self.frame_skip = 5

        self.max_episode_steps = 1000

        self.current_step = 0

        self.initial_height = None
        self.fall_height = None

        self.previous_action = np.zeros(
            self.model.nu,
            dtype=np.float32
        )

        self.nominal_qpos = None


    # ========================================================
    # GET OBSERVATION
    # ========================================================

    def _get_obs(self):

        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy()
            ]
        ).astype(np.float32)


    # ========================================================
    # GET ROOT POSITION
    # ========================================================

    def _get_position(self):

        return (
            float(self.data.qpos[0]),
            float(self.data.qpos[1]),
            float(self.data.qpos[2])
        )

    def _upright_score(self):

        # Root quaternion is qpos[3:7]
        quat = self.data.qpos[3:7]

        mat = np.zeros(9)

        mujoco.mju_quat2Mat(
            mat,
            quat
        )

        mat = mat.reshape(3, 3)

        # 1.0 = upright
        # lower values = tilted
        return float(mat[2, 2])


    def _terrain_height(self, x):

        if x < stair_start_x:
            return 0.0

        step_index = int(
            (x - stair_start_x)
            / step_length
        ) + 1

        step_index = min(
            step_index,
            number_of_steps
        )

        return (
            step_index
            * step_height
        )

    # ========================================================
    # PPO ACTION -> MUJOCO CONTROLS
    # ========================================================

    def _apply_action(self, action):

        # PPO outputs -1..1
        # Scale to a fraction of each motor's allowed torque
        torque_scale = 0.20

        ranges = self.model.actuator_ctrlrange

        max_abs = np.maximum(
            np.abs(ranges[:, 0]),
            np.abs(ranges[:, 1])
        )

        ctrl = (
            action
            * max_abs
            * torque_scale
        )

        self.data.ctrl[:] = ctrl

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0

        # Reset robot to Unitree's initial pose
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(
                self.model,
                self.data,
                0
            )
        else:
            mujoco.mj_resetData(
                self.model,
                self.data
            )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        # Save initial standing pose once
        if self.nominal_qpos is None:
            self.nominal_qpos = self.data.qpos.copy()

        # Reset previous action
        self.previous_action[:] = 0.0

        # Get root height
        _, _, z = self._get_position()

        # Set initial height and fall threshold once
        if self.initial_height is None:

            self.initial_height = z

            self.fall_height = (
                self.initial_height
                * 0.55
            )

            print(
                f"\nInitial robot height: "
                f"{self.initial_height:.3f} m"
            )

            print(
                f"Fall threshold: "
                f"{self.fall_height:.3f} m"
            )

        # IMPORTANT:
        # This must be OUTSIDE the if statement above
        return self._get_obs(), {}

    # ========================================================
    # SIMULATION STEP
    # ========================================================

    def step(
        self,
        action
    ):

        self.current_step += 1


        x_before, _, z_before = (
            self._get_position()
        )


        self._apply_action(
            action
        )


        # Advance physics
        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data
            )


        x_after, y_after, z_after = (
            self._get_position()
        )


        dt = (
            self.model.opt.timestep
            * self.frame_skip
        )


        forward_velocity = (
            x_after - x_before
        ) / dt


        upward_velocity = (
            z_after - z_before
        ) / dt


        # ====================================================
        # IMPROVED REWARD
        # ====================================================

        target_velocity = 0.4

        # Reward forward velocity near target
        velocity_reward = np.exp(
            -2.0
            * (forward_velocity - target_velocity) ** 2
        )

        # Reward upright posture
        upright = self._upright_score()

        upright_reward = (
            2.0 * upright
        )

        # Expected terrain height
        terrain_z = self._terrain_height(
            x_after
        )

        target_body_height = (
            self.initial_height
            + terrain_z
        )

        height_error = abs(
            z_after
            - target_body_height
        )

        height_reward = np.exp(
            -8.0
            * height_error
        )

        # Direct progress reward
        progress_reward = (
            2.0
            * (x_after - x_before)
        )

        # Penalize drifting sideways
        lateral_penalty = (
            0.5
            * abs(y_after)
        )

        # Penalize motor effort
        effort_penalty = (
            0.00005
            * float(
                np.sum(
                    np.square(
                        self.data.ctrl
                    )
                )
            )
        )

        # Penalize sudden changes in action
        action_change_penalty = (
            0.05
            * float(
                np.mean(
                    np.square(
                        action
                        - self.previous_action
                    )
                )
            )
        )

        self.previous_action = action.copy()

        # Fall detection
        fallen = (
            z_after < self.fall_height
            or upright < 0.45
        )

        fall_penalty = (
            20.0
            if fallen
            else 0.0
        )

        reward = (
            1.5 * velocity_reward
            + upright_reward
            + 1.5 * height_reward
            + progress_reward
            - lateral_penalty
            - effort_penalty
            - action_change_penalty
            - fall_penalty
        )


        terminated = fallen


        truncated = (
            self.current_step
            >= self.max_episode_steps
        )


        info = {

            "x": x_after,

            "y": y_after,

            "height": z_after,

            "forward_velocity":
                forward_velocity,

            "fallen":
                fallen
        }


        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info
        )