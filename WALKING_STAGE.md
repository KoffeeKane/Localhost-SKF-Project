# Flat-ground walking stage

The shared `G1StairEnv` accepts `mode="flat"` or `mode="stairs"`.
Flat mode loads Unitree's original `scene_29dof.xml` without generating or adding stairs.
The default environment mode remains `stairs` for compatibility, while `train_walk.py`
and `run_local.py` explicitly select flat mode.

## Run evaluation

In Command Prompt:

```cmd
cd /d C:\Projects\SKF_Project
.venv\Scripts\activate.bat
python run_local.py
```

The defaults near the top of `run_local.py` are:

```python
MODE = "flat"
MODEL_NAME = "g1_walk_ppo_v1.zip"
```

For three deterministic episodes without a viewer:

```cmd
python run_local.py --headless --episodes 3 --report logs\walk_v1\evaluation.json
```

To replay the older model on stairs, without training it:

```cmd
python run_local.py --mode stairs --model g1_staircase_ppo_v2.zip --episodes 3
```

Each completed episode reports its length, maximum and final x, maximum height,
and whether it fell. The evaluator also tracks the best forward distance from the
episode's initial x. JSON reports retain full precision and final upright score.
Closing the viewer ends evaluation; an incomplete episode is not scored as a success.

## Flat reward

Let `v` be forward root velocity, `dx` the change in root x during this control step,
`u` the upright score, `y` the lateral root position, `tau` the applied motor torques,
and `a` and `a_prev` the current and previous actions:

```text
reward = 4 exp(-20 (v - 0.4)^2)
       + 0.5 u
       + 10 dx
       - 0.5 abs(y)
       - 0.00005 sum(tau^2)
       - 0.05 mean((a - a_prev)^2)
       - 30 if fallen
```

The final line is an additional penalty only on the terminating fall step.
There is no constant alive bonus or height/terrain reward in flat mode.
Velocity contributes about 0.163 while stationary versus 4.0 at the target 0.4 m/s;
upright posture contributes at most 0.5. Signed progress discourages backward motion.
The old staircase reward is retained in stairs mode.

Actions still apply 20% of each actuator's maximum absolute control range as torque.
Observations remain `qpos + qvel` (71 values); actions remain 29 values. Each action
advances five physics steps, and the episode limit remains 1,000 control steps.
A fall is root height below 55% of initial height or upright score below 0.45.

## Training configuration

`train_walk.py` always constructs a fresh PPO MlpPolicy in flat mode; it never loads
the staircase policies. Compared with `train_stairs.py`, learning rate changes from
0.0003 to 0.0002, entropy coefficient from 0.01 to 0.001, and `target_kl` from None to
0.02. These modest changes address the reported high late-training KL/clipping and
the saved v2 mean action standard deviation of 1.977. The KL setting triggers early
stopping within an update when the estimated divergence becomes too large; it is
not a hard bound on every logged KL value.

The remaining requested settings are unchanged: rollout length 2048, batch size 64,
gamma 0.99, GAE lambda 0.95, and clip range 0.2. Network architecture and initial action
standard deviation use PPO's defaults. Training uses seed 42 and CPU.

The default request is 500,000 timesteps. PPO finishes full rollouts, resulting in
501,760 actual steps. The model is saved beside the scripts as `g1_walk_ppo_v1.zip`.
Existing model files are protected: the script refuses to overwrite its output.
To deliberately start another independent run, choose a new filename:

```cmd
python train_walk.py --output g1_walk_ppo_v2.zip
```

Training saves `progress.csv`, per-episode monitor metrics, and `run_config.json`
under its log directory. The first full run uses `logs\walk_v1`; subsequent runs
default to a timestamped log directory.

## Validation and milestone

```cmd
python test_curriculum.py
```

Checks cover flat-scene selection, the Gymnasium API, repeated resets, observations,
actions, metrics, torque scaling, reward priorities, fall conditions, time limits,
and retained staircase geometry.

The first milestone requires at least three completed deterministic episodes,
each ending at least 2 m ahead without falling. Identical resets and a deterministic
policy normally produce identical episodes. This verifies repeatability under the
fixed start state; it does not establish robustness to disturbances.

There is no automatic transfer to stairs in this stage.
