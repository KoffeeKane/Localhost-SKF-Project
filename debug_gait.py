"""Print reference-only gait telemetry across resets; optional static kinematic scan."""
import json
import mujoco
import numpy as np
from g1_stair_env import G1StairEnv

e = G1StairEnv(mode='flat')
e.reset()
print('Joint order:', e.joint_names)
print('Foot bodies:', [mujoco.mj_id2name(e.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in e.foot_bodies])
for n in range(360):
    _, _, t, tr, i = e.step(np.zeros(29))
    if n % 15 == 0 or t or tr:
        print(json.dumps({k: i[k] for k in ['episode_step', 'phase', 'swing_leg', 'left_foot_height',
              'right_foot_height', 'left_foot_x', 'right_foot_x', 'forward_velocity', 'upright']}))
    if t or tr:
        print('RESET: reference-only fall=', t)
        e.reset()
e.close()
