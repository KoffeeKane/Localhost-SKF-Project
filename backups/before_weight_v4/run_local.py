"""Deterministic G1 evaluation, with an optional MuJoCo viewer."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO

from g1_stair_env import G1StairEnv
from gait_metrics import GaitMetrics

# Change these together to replay a different stage/model.
MODE = "flat"
MODEL_NAME = "g1_walk_ppo_v3.zip"
PROJECT_DIR = Path(__file__).resolve().parent


def evaluate(mode=MODE, model_name=MODEL_NAME, episodes=None, headless=False):
    """Evaluate complete episodes; headless mode defaults to three episodes."""
    if headless and episodes is None:
        episodes = 3
    if episodes is not None and episodes < 1:
        raise ValueError("episodes must be positive")
    model_path = PROJECT_DIR / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Model missing: {model_path}. Train it first with train_walk.py.")
    env = G1StairEnv(mode=mode)
    results = []
    best_forward_distance = 0.0
    try:
        ppo = PPO.load(str(model_path), env=env, device="cpu")
        obs, info = env.reset()
        print(f"Model: {model_path}\nMode: {mode}; deterministic=True", flush=True)
        if headless:
            context = nullcontext(None)
        else:
            import mujoco.viewer
            print("Close the MuJoCo window to stop.", flush=True)
            context = mujoco.viewer.launch_passive(env.model, env.data)
        with context as viewer:
            episode_number = 1
            max_x, max_height = info["x"], info["height"]
            max_forward_distance = 0.0
            episode_reward = 0.0
            gait = GaitMetrics(info)
            while episodes is None or episode_number <= episodes:
                if viewer is not None and not viewer.is_running():
                    break
                loop_start = time.perf_counter()
                action, _ = ppo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("Non-finite simulation output during evaluation.")
                max_x = max(max_x, info["x"])
                max_height = max(max_height, info["height"])
                max_forward_distance = max(max_forward_distance, info["forward_distance"])
                best_forward_distance = max(best_forward_distance, max_forward_distance)
                episode_reward += reward
                gait.update(info)
                if viewer is not None:
                    viewer.sync()
                if terminated or truncated:
                    result = {
                        "episode": episode_number,
                        "steps_survived": info["episode_step"],
                        "max_x": max_x,
                        "final_x": info["x"],
                        "max_height": max_height,
                        "fallen": info["fallen"],
                        "truncated": bool(truncated),
                        "forward_distance": info["forward_distance"],
                        "max_forward_distance": max_forward_distance,
                        "final_upright": info["upright"],
                        "episode_reward": episode_reward,
                        # Conservative milestone: complete the episode without a fall
                        # and finish at least 2 m ahead, rather than lunge then collapse.
                        "milestone_reached": bool(not terminated and info["forward_distance"] >= 2.0),
                    }
                    result.update(gait.summary())
                    results.append(result)
                    print(
                        f"Episode {episode_number}: steps={info['episode_step']}, "
                        f"max x={max_x:.3f} m, final x={info['x']:.3f} m, "
                        f"max height={max_height:.3f} m, fell={info['fallen']}; "
                        f"best forward distance={best_forward_distance:.3f} m",
                        flush=True,
                    )
                    episode_number += 1
                    if episodes is not None and episode_number > episodes:
                        break
                    obs, info = env.reset()
                    max_x, max_height = info["x"], info["height"]
                    max_forward_distance = 0.0
                    episode_reward = 0.0
                    gait = GaitMetrics(info)
                if viewer is not None:
                    dt = env.model.opt.timestep * env.frame_skip
                    time.sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        print("Evaluation interrupted; only completed episodes are reported.")
    finally:
        env.close()
    return {
        "mode": mode, "model": str(model_path), "deterministic": True,
        "episodes": results, "best_forward_distance": best_forward_distance,
        "walking_milestone_reached": bool(len(results) >= 3 and all(r["milestone_reached"] for r in results)),
        "note": "Resets use the same initial pose; repeated deterministic episodes test repeatability, not robustness to perturbations.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("flat", "stairs"), default=MODE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--headless", action="store_true", help="Evaluate without opening a viewer")
    parser.add_argument("--report", type=Path, help="Save evaluation metrics as JSON")
    args = parser.parse_args()
    report = evaluate(args.mode, args.model, args.episodes, args.headless)
    print(f"Walking milestone reached: {report['walking_milestone_reached']}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Evaluation report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
