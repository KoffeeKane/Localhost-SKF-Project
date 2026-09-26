# G1 v3: procedural-reference residual RL

## Local execution and changed files
All simulation, training, and evaluation use C:\Projects\SKF_Project\.venv\Scripts\python.exe in the existing project. No separate training environment was created.

- g1_stair_env.py: flat-ground reference oscillator, residual PD control, phase observations, foot measurements, posture and gait rewards.
- g1_pd_env_v2.py: preserved v2 implementation, used for staircase mode.
- train_walk.py: defaults to fresh v3 model and records residual control settings.
- run_local.py: defaults to v3 and reports measured foot-lift events and reward components.
- gait_metrics.py: measured lift-event classification and episode gait statistics.
- test_gait.py: new sanity checks.
- test_curriculum.py: existing v2 checks now explicitly target preserved v2 implementation.
- debug_gait.py: reference-only phase, foot, velocity, and upright telemetry.
- inspect_gait_rollout.py: deterministic image sequence and detailed per-step diagnostics.

Existing model archives are not overwritten. Pre-v3 scripts are backed up in C:\Projects\SKF_Project\backups\before_gait_v3. The new flat observation is incompatible with v1/v2 policies; use their original environment when replaying them. Staircase mode retains the v2 71-value observation and PD behavior; no stair training is run.

## Observation and controller
Flat observation has 73 float32 values: qpos[0:36] (root xyz, root quaternion wxyz, 29 joint angles), qvel[0:35] (root linear/angular velocity, 29 joint velocities), then sin(2*pi*phase), cos(2*pi*phase). Both training and evaluation import the same environment. PPO actions remain 29 values in [-1,1], ordered by loaded MuJoCo actuators and mapped through actuator joint IDs to qpos/qvel addresses.

Baseline pose: hip pitch -0.10 rad, knee +0.30 rad, ankle pitch -0.20 rad on both sides; all other joints zero.

Phase advances every 0.002-second physics step: phase = (phase + dt/1.2) mod 1. One full left/right cycle lasts 1.2 seconds (100 individual steps/minute). Policy acts every five physics steps (0.01 seconds).

For left leg shift=0 and right leg shift=0.5:

    s = sin(2*pi*(phase + shift))
    b = max(0, s)^2
    hip_offset = -0.08*s - 0.27*b
    knee_offset = 0.40*b
    ankle_offset = -(hip_offset + knee_offset)
    reference = nominal + offsets
    target = clip(reference + 0.10*clip(action,-1,1), joint_limits)
    torque = clip(kp*(target-q) - kd*qvel, torque_limits)

The positive-sine-squared term is continuous with a continuous first derivative at swing boundaries. Other reference joints remain at nominal. There is no scripted lateral weight shift or arm swing. Left is commanded swing in phase [0,0.5), right in [0.5,1). PD torques are recomputed every physics step. Joint/actuator mappings, hinge types, unit motor gears, actuator control limits, and joint force limits retain the existing safety checks.

PD gains kp/kd: hips 100/2, knees 150/4, ankles 40/2, waist 100/2, shoulders/elbows 40/1, wrists 10/0.5. These are unchanged from v2. Residuals are +/-0.10 rad on every joint before joint-limit clipping.

## Foot measurements and rewards
Foot bodies are the inspected left_ankle_roll_link and right_ankle_roll_link. Each has four sole collision spheres. Foot height means minimum sphere-bottom clearance above the z=0 floor, not ankle-body height. Small negative values reflect contact penetration. Floor contact is detected from MuJoCo contacts within 1 mm. Sliding uses contact-point Jacobians times qvel, avoiding finite-difference noise. Torso orientation comes from torso_link; relative foot x is expressed along pelvis yaw heading.

Let g=sin(2*pi*phase)^2, c=clip(swing_sole_height/0.05,0,1), f=clip(swing_foot_relative_x/0.12,0,1), and S=stance_contact*exp(-(stance_sole_height/0.025)^2). Let E be mean squared joint error against reference, u root upright score, ut torso upright score, p torso pitch, v forward velocity, and dx forward displacement per policy step.

Final per-step reward is the sum of:

| Term | Formula |
|---|---|
| Velocity | 2*exp(-20*(v-0.4)^2) |
| Progress | 3*clip(dx,-0.02,0.02) |
| Root upright | 0.5*u |
| Torso upright | 0.75*ut |
| Torso pitch | -0.75*p^2 |
| Reference imitation | 0.5*exp(-10*E) |
| Swing lift | 1.5*g*c*S |
| Swing forward | g*f*c*S |
| Stance support | 0.5*g*S |
| Sliding | -0.5*sum(min(slip_speed_squared_per_foot,4)) |
| Lateral drift | -0.5*abs(y) |
| Effort | -0.00005*sum(torque^2) |
| Action changes | -0.05*mean((action-previous_action)^2) |
| Fall | -30 on fall |

Forward progress is capped at 0.06 reward per policy step. Swing rewards require clearance and stance support; mere dragging cannot earn them. Fall conditions remain root height below 55% initial height or root upright below 0.45; time limit is 1,000 policy steps.

## PPO and verification
PPO settings: learning_rate=0.0002, ent_coef=0.001, target_kl=0.02, n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_range=0.2, CPU, seed=42, MlpPolicy, default 10 optimization epochs with KL early stopping. Fresh policy; no v2 weight transfer.

All 13 environment tests passed. Checks cover reset/API, 73 observations and 29 actions, finite random steps, phase advance/wrap/alternation, reference continuity/symmetry, reference-plus-residual joint bounds, PD saturation/substeps, measured contact and sliding, and preserved staircase behavior. Final short PPO run completed 2,048 steps. Debug logs show phase alternation but reference-only dynamics still keep feet largely planted and fall after 113 steps. The reference is a gait prior, not a validated dynamically balanced walking controller.

A qualified measured step requires: active swing foot clearance >2 cm, no swing contact, opposite-foot floor contact, >2 cm forward travel relative to pelvis within that swing interval, and root/torso upright scores >0.75. Alternation requires adjacent qualified events from opposite legs. Sustained alternation requires at least four alternating qualified events. High foot positions during a fall do not count as walking.

## Completed training and evaluation
Saved C:\Projects\SKF_Project\g1_walk_ppo_v3.zip after 501,760 actual timesteps (500,000 requested; whole rollouts). Training took approximately 652 seconds (10 minutes 52 seconds), averaging 768 steps/second. Final logged rolling mean reward was 331 and episode length 126 steps. Final logged approximate KL 0.021164, clip fraction 0.237, explained variance rounded to 1.0. Saved action standard deviation mean 0.844516. Reward is not directly comparable with v2 because its formula changed.

Three deterministic episodes ran locally through run_local.py with the MuJoCo viewer:

| Episode | Steps survived | Max x | Final x | Fell | Max left sole height | Max right sole height | Alternating lift | 2 m milestone |
|---|---:|---:|---:|---|---:|---:|---|---|
| 1 | 126 | 0.747116 m | 0.747116 m | Yes | 0.008798 m | 0.009105 m | No | No |
| 2 | 126 | 0.747116 m | 0.747116 m | Yes | 0.008798 m | 0.009105 m | No | No |
| 3 | 126 | 0.747116 m | 0.747116 m | Yes | 0.008798 m | 0.009105 m | No | No |

The foot-height maxima include initial spawn clearance and do not represent successful swing lifts. Zero qualified step events occurred. Identical initial conditions produce identical episodes; this is repeatability, not a robustness assessment.

A separate deterministic local rollout was rendered and visually inspected. Its chronological image sequence shows nearly planted feet, body movement over them, toe pivoting, and eventual forward collapse. It does not resemble natural alternating steps. V3 moves farther than v2 (0.192 m) and the reported v1 result (0.649 m), but forward displacement is still not walking. A continuous desktop replay was launched after evaluation; close the MuJoCo window to stop it.

## Failure evidence and interpretation
The oscillator and reference alternate correctly, but the robot does not unload either foot enough to swing. At phase 0.25 and 0.75, both feet are still contacting the floor. Mean per-policy-step reward contributions in each deterministic episode:

| Term | Mean contribution |
|---|---:|
| Velocity | 1.280552 |
| Progress | 0.017788 |
| Root upright | 0.465917 |
| Torso upright | 0.690256 |
| Torso pitch penalty | -0.091792 |
| Imitation | 0.444949 |
| Swing lift | 0.000001246 |
| Swing forward | 0 |
| Stance support | 0.237994 |
| Sliding penalty | -0.005420 |
| Lateral penalty | -0.027652 |
| Effort penalty | -0.054054 |
| Action-change penalty | -0.001407 |
| Fall penalty, averaged over episode | -0.238095 |

These results show that the policy can still earn velocity, posture, imitation, and support reward without stepping. The swing rewards are effectively absent. Averaging imitation error over all 29 joints also lets relatively quiet upper-body joints dilute leg errors. Low measured contact slipping does not imply good walking: the body can lean and fall over feet that remain approximately planted.

The primary observed problem is balance and support transfer. This simple sagittal reference has no lateral weight transfer; its intended leg motion is not realized as foot clearance under gravity and contact. Mean target-tracking RMSE is 0.1028 rad, with a maximum of 0.1647 rad. On average, 34.0% of residual actions exceed 95% of their allowed magnitude (maximum 58.6%). This suggests the limited correction range is frequently used up, but it does not establish that increasing it would fix balance. PD gains, loaded tracking, reference amplitude, and residual range need short controlled tests to separate their effects; this run alone cannot identify a unique cause.

The next useful investigation would be a short controller/reference test that demonstrates support transfer and actual foot clearance before another long PPO run, followed by revisiting reward incentives using that evidence. No additional long run or stair training was started.

## Supporting outputs
- v3_evaluation.json: complete three-episode results and component rewards.
- v3_run_config.json: saved PPO and controller configuration.
- v3_visuals/v3_rollout.png: chronological visual sequence, left to right then top to bottom.
- v3_visuals/v3_trace.json: every-step foot/contact/posture data and controller diagnostics.
- Project logs: C:\Projects\SKF_Project\logs\walk_gait_v3.

Final reference geometry cross-check (root held at reset height): at phase 0.25 the left sole is 0.043971 m above the floor and left ankle x relative to pelvis is +0.072182 m; at phase 0.75 the right foot has the same values. Neutral sole clearance is 0.008798 m and relative ankle x is -0.026002 m. Thus the commanded geometry provides about 3.5 cm additional swing clearance and 9.8 cm forward reach; the dynamic simulation fails to realize it. These static values are not claims of successful dynamic stepping.
