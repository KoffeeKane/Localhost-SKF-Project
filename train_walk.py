"""Train flat-ground walking from scratch; no staircase transfer in this stage."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from g1_stair_env import G1StairEnv

PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_STEPS = 500_000
MODEL_NAME = "g1_walk_ppo_v4.zip"

PPO_SETTINGS = dict(
    # v2 reported KL ~0.04-0.05 and clip_fraction ~0.35-0.42: reduce update size.
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    # v2's mean action std reached 1.98. Reduce the incentive for more noise,
    # while retaining stochastic exploration; leave initial std/architecture alone.
    ent_coef=0.001,
    # SB3 stops an update epoch early above roughly 1.5 * target_kl.
    target_kl=0.02,
    device="cpu",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / MODEL_NAME)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.timesteps >= 500000:
        gate = json.loads((PROJECT_DIR / 'logs' / 'v4_reference' / 'validation.json').read_text())
        if not gate['passed']:
            raise RuntimeError('Reference-only unloading/lifting gate did not pass')
        for name, expected in gate['environment_hashes'].items():
            if hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Reference environment changed after validation: ' + name)
    output = args.output.resolve()
    if output.suffix != ".zip":
        parser.error("--output must end in .zip")
    # Never silently replace any previous model, including the staircase models.
    if output.exists():
        raise FileExistsError(f"Model already exists: {output}. Choose a new --output filename.")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or PROJECT_DIR / "logs" / ("walk_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir.mkdir(parents=True, exist_ok=True)

    env = G1StairEnv(mode="flat")
    env = Monitor(env, str(log_dir / "episodes"), info_keywords=(
        "forward_distance", "x", "height", "upright", "fallen", "episode_step",
    ))
    try:
        # Deliberately create a NEW policy; do not load either staircase model.
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, **PPO_SETTINGS)
        model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
        metadata = {
            "control": "force_gated_weight_transfer_residual", "action_scale": env.unwrapped.action_scale,
            "kp": env.unwrapped.kp.tolist(), "kd": env.unwrapped.kd.tolist(),
            "nominal_joint_positions": env.unwrapped.nominal_joint_positions.tolist(),
            "reference_config": env.unwrapped.cfg,
            "gait_period": env.unwrapped.gait_period, "observation_shape": list(env.observation_space.shape),
            "mode": "flat", "policy": "MlpPolicy", "from_scratch": True,
            "requested_timesteps": args.timesteps, "seed": args.seed,
            "ppo": PPO_SETTINGS, "stable_baselines3": stable_baselines3.__version__,
            "model": str(output),
            "environment_sha256": hashlib.sha256((PROJECT_DIR / "g1_stair_env.py").read_bytes()).hexdigest(),
        }
        (log_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Training flat walking for {args.timesteps:,} requested timesteps.")
        print("PPO completes whole 2048-step rollouts: 500,000 requests produce 501,760 steps.")
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        model.save(str(output))
        metadata["actual_timesteps"] = model.num_timesteps
        metadata["final_action_std_mean"] = float(model.policy.log_std.detach().exp().mean())
        metadata["completed"] = True
        (log_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Training finished at {model.num_timesteps:,} steps. Saved: {output}")
        print(f"Logs: {log_dir.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
