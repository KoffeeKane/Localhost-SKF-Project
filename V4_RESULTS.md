# G1 v4 weight-transfer residual RL

## Scope and local execution
All training and simulation use C:\Projects\SKF_Project\.venv\Scripts\python.exe in the existing project. No separate virtual environment and no stair training. Existing v1/v2/v3 and staircase model archives remain untouched. The full run is authorized only after a reference-only gate succeeds; train_walk.py enforces a passed validation file and matching hashes of all environment/controller dependencies.

## Files
- g1_stair_env.py: v4 reference, constrained inverse kinematics, force-gated phase advancement, COM/posture feedback, force diagnostics, new rewards.
- g1_gait_env_v3.py: preserved v3 implementation; v4 reuses model inspection and measurement code.
- g1_pd_env_v2.py: unchanged preserved v2 controller, still used for staircase behavior.
- gait_metrics.py: unload/lift/advance/replant step detector and episode diagnostics.
- train_walk.py: v4 output default, exact reference configuration metadata, enforced reference-validation gate.
- run_local.py: v4 deterministic evaluation default with new metrics.
- probe_reference.py, validate_reference.py: zero-residual debugging and required three-episode validation.
- inspect_gait_rollout.py: reference-only or trained-policy visual sequence and every-step JSON trace.
- test_weight_transfer.py: v4 tests; test_gait.py now explicitly checks preserved v3 behavior.
- Short development probes are saved in logs/v4_reference; no probe used PPO or long training.

Pre-v4 environment, evaluator, trainer, and metrics are backed up in backups/before_weight_v4. Old models require their matching environment/control behavior for replay.

## Observation and actions
Flat observation remains 73 float32 values: 36 qpos values (root xyz, quaternion wxyz, 29 joint angles), 35 qvel values (6 root velocities and 29 joint velocities), then sine and cosine of 2*pi*phase. Joint ordering is resolved from the loaded actuator-to-joint mappings, not guessed indices. Training and evaluation import the same class.

PPO action space is 29 values in [-1,1]. Residual position authority is exactly +/-0.05 rad per joint before final joint-limit clipping. Targets are nominal standing pose plus procedural reference offsets plus bounded reference-controller balance corrections plus PPO residuals. No raw-torque PPO control.

## Subphases and force gate
There is a 0.6-second initial settling period at phase zero. Nominal full cycle is 2.4 seconds, with one 1.2-second half-cycle per leg. Phase advances every 0.002-second physics step and the policy acts every five physics steps. Unloading holds can lengthen the cycle.

Let u=(2*phase) mod 1. Left swings in the first half-cycle, right in the second:

| Half-cycle fraction | Subphase |
|---|---|
| 0 to 0.22 | Weight shift toward stance |
| 0.22 to 0.32 | Unload swing foot |
| At 0.32 if necessary | Wait for measured support transfer |
| 0.32 to 0.47 | Lift |
| 0.47 to 0.65 | Forward swing |
| 0.65 to 0.82 | Lower and plant |
| 0.82 to 1 | Transfer toward next stance |

At u=0.32, phase stops until measured stance vertical force exceeds 55% body weight and swing vertical force is below 40% body weight. Force measurements used for the gate refresh every 0.01-second policy step. Joint targets remain continuous while phase waits; the speed of phase progression can change at gate release.

## Smooth reference equations and inverse kinematics
Define S(t)=6t^5-15t^4+10t^3 for t clipped to [0,1]. Lateral pelvis reference relative to the feet is:

    y_ref = stance_sign * 0.06 * S(u/0.22) * (1-S((u-0.82)/0.18))

stance_sign is +1 for left stance and -1 for right stance. To realize this through joints rather than moving the free root artificially, desired foot y relative to a fixed reference root is shifted by -y_ref. The reference does not apply external forces, pin the pelvis, or teleport the robot during simulation.

Neutral foot positions come from MuJoCo forward kinematics of the nominal standing pose. Both target foot x positions include a +0.015 m offset (forward_bias=-0.015). Relative to those neutral positions:

    swing_dx = 0.06*(S((u-0.47)/0.18)-0.5)
    stance_dx = 0.06*(0.5-S(u))
    unload_dz = 0.004*S((u-0.22)/0.10)
    swing_dz = (unload_dz + 0.071*S((u-0.32)/0.15)) * (1-S((u-0.65)/0.17))

Thus the geometric swing-height request is 7.5 cm, with the first 4 mm used to assist unloading. Foot orientation targets stay level. The nominal standing pose is hip pitch -0.10, knee +0.30, ankle pitch -0.20 radians; other joints zero.

A damped Jacobian inverse-kinematics solve generates 240 samples from these smooth functions using actual model geometry. Position and orientation are solved together (orientation residual weight 0.3; damping 0.001; at most 80 iterations; increments limited to 0.04 rad). It restarts from nominal for every sample to avoid switching leg configurations. Knee solutions are bounded to [0.08,1.4] rad, hip pitch to [-1,0.6], hip roll to [-0.35,0.35], alongside model limits with 0.055-rad margins. Periodic cubic Hermite interpolation provides continuous reference values and first derivatives. This is a procedural trajectory sampled for efficient execution, not manually authored animation keyframes. Continuity and bounds were tested.

Reset places the feet on the floor using their modeled sole geometry. No root-position override occurs after reset.

## Balance feedback and PD control
Purely timed lateral shifting overshot the stance foot in short tests. The reference controller therefore includes bounded ankle target corrections using root attitude, COM error, and COM velocity. Let c be whole-robot COM, vc its velocity, and fmid the mean ankle-body position. Define:

    e = c_xy - (fmid_xy + [0.035, y_ref])
    b = 2.0*e + 3.0*vc_xy
    ankle_pitch_correction = clip(1.5*root_pitch + 0.2*root_pitch_rate + b_x, -0.25, 0.25)
    ankle_roll_correction = clip(1.5*root_roll + 0.2*root_roll_rate - b_y, -0.15, 0.15)

Both ankles receive these reference-controller corrections before the +/-0.05-rad PPO residual. The attitude rates use root angular qvel components. Position errors for lateral feedback are in world coordinates; this flat, forward task has no turning command.

    target = clip(reference + balance_correction + 0.05*clip(action,-1,1), joint_limits)
    torque = clip(kp*(target-current_joint_position) - kd*joint_velocity, torque_limits)

PD updates every physics step. Gains kp/kd: hips 220/5, knees 300/6, ankles 100/3, waist 100/2, shoulders/elbows 40/1, wrists 10/0.5. Leg gains were increased based on the short reference-only tracking tests, not broad PPO tuning. Existing actuator transmission checks, unit gear checks, joint mapping, and model actuator/joint torque clipping remain.

## Contact forces, COM, and diagnostics
Feet are the inspected left_ankle_roll_link and right_ankle_roll_link bodies, each with four collision spheres. Sole height is the minimum collision-sphere bottom above the flat z=0 floor. Contact booleans use floor contact distances <=1 mm. Vertical ground-reaction forces sum mj_contactForce results transformed from contact to world coordinates with the appropriate sign for whether the foot is geom1 or geom2. Force units were checked against robot weight after settling: mass 35.112142 kg, weight approximately 344.45 N.

COM is MuJoCo subtree_com at pelvis, covering the full robot. COM velocity comes from the subtree COM Jacobian times qvel. Lateral support offset is COM y minus stance ankle-body y; this is a diagnostic, not a full support-polygon stability test. Contact-point horizontal velocity uses Jacobians at actual contacts; foot-body horizontal speed is logged separately.

Every-step diagnostic traces include phase/subphase, stance/swing sides, both knee angles, sole heights, foot positions/x, foot-body horizontal speeds, contact-point slip, vertical forces/contact states, COM x/y and stance offset, pelvis x, forward velocity, upright/torso scores, and residual saturation percentage.

## Reward formula
Normalize vertical foot forces by body weight. Let Ls and Lw be stance/swing normalized load, U=clip(1-Lw/0.2,0,1)*support, support=clip(Ls/0.7,0,1), H=clip(swing_sole_height/0.04,0,1), Q=U*H, F=clip((swing_relative_x+0.02)/0.08,0,1), and A=S((u-0.22)/0.10)*(1-S((u-0.82)/0.18)). Let slip2 be maximum squared horizontal velocity at stance-foot floor contacts, E mean squared reference error over the 12 leg joints, kstance stance knee angle, and pitch torso pitch.

The per-policy-step reward sums:

| Term | Formula |
|---|---|
| Unloading | 1.5*A*U |
| Actual lift | 4*A*Q |
| Forward swing | 3*A*Q*F |
| Missed lift | -2*(1-Q) when 0.47<=u<=0.65, otherwise zero |
| Velocity | 0.75*exp(-20*(v-0.4)^2)*(0.1+0.9*Q) |
| Progress | 2*clip(dx,-0.02,0.02)*Q |
| Stable stance | 0.4*A*support*exp(-slip2/0.01) |
| Stance slip | -2*min(slip2,4) |
| Crouch | -4*max(0,min(left_knee,right_knee)-0.55)^2 when both feet contact; zero otherwise |
| Stance knee | -(kstance-0.30)^2 |
| Root upright | 0.2*root_upright |
| Torso upright | 0.3*torso_upright |
| Torso pitch | -pitch^2 |
| Leg imitation | 0.15*exp(-10*E) |
| Lateral drift | -0.5*abs(pelvis_y) |
| Effort | -0.00005*sum(torque^2) |
| Action changes | -0.05*mean((action-previous_action)^2) |
| Fall | -30 on fall |

Normal single swing-knee flexion is exempt from the simultaneous-crouch penalty. Forward progress earns nothing without unloaded foot clearance; velocity reward is reduced to one tenth when that quality measure is zero.

## Valid-step definition
A step candidate requires the commanded swing foot to unload below 15% body weight while the opposite foot supports over 55%, sole clearance exceeds 2 cm over its most recent loaded/planted baseline, swing contact is absent, and both root and torso upright scores exceed 0.85. It must subsequently advance over 2 cm in world x and replant with force >15% body weight, contact present, height within 8 mm of baseline, for three consecutive policy samples. Upright thresholds must also hold when the step is counted. A lift without forward movement or replanting is not a valid step. Alternating pairs count adjacent completed steps of opposite sides (L,R,L has two such pairs).

Episode maximum clearance includes all frames; qualifying step/lift events separately exclude low-upright fall motion. Average stance slip is the contact-point horizontal speed averaged over all policy samples, with zero contribution when the designated stance foot has no floor contact. Crouch fraction is the fraction of policy samples with both feet contacting and both knees above 0.55 rad.

## Tests and reference-only gate
All 20 v2/v3/v4 tests passed: API/reset/shapes, random-step finiteness, phase wrapping and bilateral support gates, reference continuity/bounds/lateral symmetry, residual magnitude and clipped torques, crouch behavior, force units, stance slip, and valid-step detector including replant and forward-motion requirements.

Three zero-residual episodes each survived 453 steps and then fell. Each completed one left and one right valid step, one alternating pair, with true unloading on both sides. Maximum clearance was 6.49 cm left and 3.90 cm right; mean stance slip 0.0141 m/s; crouch fraction zero. Valid replant events were at steps 159 and 280. Left/right/left qualified lifts occurred at steps 122, 242, and 385. Maximum/final pelvis x was about 0.219 m. The rendered sequence was inspected and showed actual alternating foot lifting, followed later by lateral loss of balance. Passing this gate establishes an effective stepping prior, not stable sustained walking.

The final 2,048-step PPO sanity run passed. Full-run settings remain v3's: learning_rate=0.0002, ent_coef=0.001, target_kl=0.02, n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_range=0.2, CPU, seed 42, fresh MlpPolicy, default 10 optimization epochs with KL early stopping.

## Full training result
Saved C:\Projects\SKF_Project\g1_walk_ppo_v4.zip after 501,760 actual timesteps (500,000 requested, rounded to complete PPO rollouts). Training took approximately 1,025 seconds (17 minutes 5 seconds), averaging 489 steps/second. Final logged rolling mean episode length was 415 and reward approximately 1,080. Approximate KL was 0.019675, clip fraction 0.202, explained variance 0.968, and saved action standard deviation mean 0.904790. The reward formula changed, so raw reward is not directly comparable with earlier versions.

The full environment and detector hashes were verified unchanged between reference validation, training, and evaluation. No previous model archives were overwritten.

## Three deterministic local viewer evaluations
All episodes use the same reset, so identical outcomes establish repeatability rather than robustness to perturbations.

| Episode | Steps survived | Max x | Final x | Fell | Valid left | Valid right | Alternating pairs | Max left clearance | Max right clearance | Average stance slip |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 461 | 0.233473 m | 0.226704 m | Yes | 2 | 1 | 2 | 0.100582 m | 0.050578 m | 0.026626 m/s |
| 2 | 461 | 0.233473 m | 0.226704 m | Yes | 2 | 1 | 2 | 0.100582 m | 0.050578 m | 0.026626 m/s |
| 3 | 461 | 0.233473 m | 0.226704 m | Yes | 2 | 1 | 2 | 0.100582 m | 0.050578 m | 0.026626 m/s |

Clearance maxima are whole-episode values above measured planted baselines, including motion during the eventual fall. In particular, the left 10.06 cm maximum must not be interpreted as a successful swing height; supported, upright swing frames separately show roughly 8 cm left and 5 cm right. The qualifying lift/replant events, rather than whole-episode maxima, establish real stepping.

In each episode:
- Left/right/left qualified lifts occurred at steps 110, 230, and 361.
- Completed replant events occurred at steps 153, 277, and 415, with measured forward foot advances of 0.2134, 0.2981, and 0.1658 m since the preceding loaded baseline.
- Both feet genuinely unloaded. At the first left lift, left/right vertical force was 0/384.83 N; at the right lift it was 384.41/0 N.
- Excessive simultaneous planted-knee crouch fraction was zero under the documented 0.55-rad threshold. This does not mean the knees are perfectly straight.
- Mean residual saturation was 19.22% of action components; joint target-tracking RMSE averaged 0.0507 rad and peaked at 0.0906 rad.
- The 2 m milestone was not reached.

The recorded sequence was visually inspected: it shows alternating foot lift, knee flexion, forward placement, and an eventual sideways fall. This is real but short-lived stepping, not sustained natural walking. A continuous local MuJoCo replay was launched after evaluation and was confirmed producing episodes without viewer errors; close its window to stop.

## Reward evidence and failure diagnosis
Average per-policy-step reward contributions were:

| Term | Mean |
|---|---:|
| Unload | 0.542264 |
| Lift | 1.090694 |
| Forward swing | 0.623806 |
| Missed lift | -0.001892 |
| Velocity | 0.039786 |
| Progress | 0.000443 |
| Stable stance | 0.183407 |
| Stance slip | -0.011625 |
| Crouch | 0 |
| Stance knee | -0.002377 |
| Root upright | 0.194915 |
| Torso upright | 0.291444 |
| Torso pitch | -0.016751 |
| Leg imitation | 0.141646 |
| Lateral drift | -0.063379 |
| Effort | -0.084000 |
| Action changes | -0.001072 |
| Fall, averaged | -0.065076 |

Unlike v3, actual lift and forward-swing rewards are now materially earned. Contact unloading and basic gait structure work for both sides. Excessive simultaneous knee crouching is not the dominant remaining failure. PD tracking is improved, although these tests do not establish optimal gains.

The dominant observed failure is accumulated lateral balance error and inadequate transfer onto the newly planted foot. At step 361 the COM was only 0.004 m laterally from the stance ankle. By step 390 the offset was -0.070 m, by step 415 it was -0.163 m, and by the fall it was -0.619 m relative to the designated stance ankle. After the third replant the new stance foot did not retain sufficient support. The observed third completed step therefore does not imply a stable subsequent transfer.

Mean contact slipping increased from reference-only 0.0141 m/s to trained 0.0266 m/s; it remains measurable and may contribute, but the trace points more strongly to lateral support control than a return to v3-style planted-foot dragging. The reference's simple feedback uses mean foot position rather than a full support-polygon or capture-point controller. It also lacks a dedicated heading controller. These are limits of this experiment, not proven isolated causes.

PPO's gain over the already-effective reference is modest: three completed steps instead of two and 461 surviving steps instead of 453. The reduced +/-0.05 residual range and 19.22% saturation show limited authority is still frequently used, but saturation alone does not establish that increasing it would fix the failure.

The conservative 0.06 m stride parameter and 2.4-second cycle prioritize unloading over speed; their nominal progression scale is around 0.05 m/s, well below the retained 0.4 m/s reward target. The 10-second episode horizon and small steps also make 2 m an ambitious target for this reference. Reward now values actual stepping, but sustained lateral stability is still missing.

No second long training run and no stair training were started. This experiment is complete with partial stepping success, failed sustained balance, and the 2 m milestone unmet.

## Supporting outputs
- v4_evaluation.json: complete three-episode results and reward components.
- v4_run_config.json: exact saved PPO and reference-controller configuration.
- v4_reference_validation.json: three zero-residual results and hashes enforced before training.
- v4_reference_visuals/v4_rollout.png: pre-training reference-only sequence.
- v4_visuals/v4_rollout.png: trained-policy sequence, chronological left to right then top to bottom.
- v4_visuals/v4_trace.json: every-step diagnostic data.
- Full local logs: C:\Projects\SKF_Project\logs\walk_weight_v4 and logs\v4_reference.
