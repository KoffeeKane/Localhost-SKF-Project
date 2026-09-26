"""Save visual evidence and per-step gait measurements for one deterministic episode."""
import argparse
import json
import struct
import zlib
from pathlib import Path
import mujoco
import numpy as np
from stable_baselines3 import PPO
from g1_stair_env import G1StairEnv


def save_png(path, pixels):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    h, w, _ = pixels.shape
    raw = b''.join(b'\x00' + row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w,h,8,2,0,0,0))
                     + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--reference', action='store_true')
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
env = G1StairEnv(mode='flat')
policy = None if args.reference else PPO.load(str(Path(__file__).parent / 'g1_walk_ppo_v4.zip'), device='cpu')
obs, info = env.reset()
renderer = mujoco.Renderer(env.model, height=360, width=480)
camera = mujoco.MjvCamera()
camera.azimuth = 120
camera.elevation = -12
camera.distance = 2.4
frames = []
trace = []
try:
    for n in range(1000):
        a = np.zeros(29) if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, reward, t, tr, info = env.step(a)
        row = dict(info)
        row['residual_saturation_fraction'] = float(np.mean(np.abs(a) > .95))
        row['joint_tracking_rmse'] = float(np.sqrt(np.mean((env.data.qpos[env.joint_qpos] - env.target_q) ** 2)))
        row['reference_tracking_rmse'] = float(np.sqrt(np.mean((env.data.qpos[env.joint_qpos] - env.reference_pose()) ** 2)))
        trace.append(row)
        if n % 10 == 0 or t or tr:
            camera.lookat[:] = [env.data.qpos[0], 0, .55]
            renderer.update_scene(env.data, camera=camera)
            frame = renderer.render().copy()
            frames.append(frame)
            save_png(args.output_dir / f'frame_{n+1:04}.png', frame)
        if t or tr:
            break
finally:
    renderer.close()
    env.close()
chosen = np.unique(np.linspace(0, len(frames)-1, min(16, len(frames))).astype(int))
montage = np.full((360*((len(chosen)+3)//4), 480*4, 3), 255, dtype=np.uint8)
for j, idx in enumerate(chosen):
    montage[(j//4)*360:(j//4+1)*360, (j%4)*480:(j%4+1)*480] = frames[idx]
save_png(args.output_dir/'v4_rollout.png', montage)
(args.output_dir/'v4_trace.json').write_text(json.dumps(trace, indent=2))
print('Saved rollout evidence at', args.output_dir)
