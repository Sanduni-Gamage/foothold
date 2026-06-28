# Hierarchical RL for Go2: high-level over a velocity-tracker low-level

Research record on what's published for adding a high-level policy that
selects footholds (or foothold-equivalent commands) on top of a low-level
locomotion policy. Focus: Unitree Go2, Isaac Lab v2.3.2 + rsl_rl, real-Go2-
sensors-only constraint (proprio + L1 lidar; no scandots / height_scanner /
foot contact sensors at deploy time).

---

## 1. The mechanical problem with the user's stated design

> "High-level outputs foot-position targets; existing `Unitree-Go2-Velocity`
> low-level tracks them."

**This does not work without retraining the low-level.** Every published
successful foothold-target hierarchy trained the LLP **with foothold
conditioning baked into its observation space from scratch** (DeepGait
2020, RLOC 2022, Coholich 2025, Choi/Raibo 2025, PUMA 2026). A
velocity-tracking LLP has no learned response to xy-foothold-target input
dimensions — those dimensions are outside its training distribution; the
policy will ignore them or behave unpredictably.

Source: every comparison paper in §6 below; explicitly stated in
Coholich 2025 ([arXiv:2506.20036][coholich]) §III.B.

[coholich]: https://arxiv.org/abs/2506.20036

## 2. Three architectural options that are real

| Option | What it is | Cost | Deploy obs |
|---|---|---|---|
| **A. Foothold-aware LLP, retrain** | Extend obs with 8-dim per-foot xy target; retrain LLP with foothold-tracking rewards. Coholich / Raibo / PUMA template. | New training run (~3 k iters typical) | proprio + foot-target vector + (optional) lidar-derived terrain |
| **B. HLP outputs velocity** | Keep current LLP unchanged. HLP emits (vx, vy, ωz) commands (possibly conditioned on lidar / terrain). The ABS pattern. | No LLP retrain; new HLP only | proprio only |
| **C. Walk-These-Ways extension** | HLP outputs velocity + gait parameters (footswing height, gait phase, body height). Single policy with conditioning. | Extend obs and retrain LLP — but reuses the WTW recipe directly | proprio + gait commands |

The "no published precedent" entry is exactly the user's verbatim idea
(stock velocity LLP + foothold HLP). It's open territory because the
interface mismatch is known, not because no one looked.

## 3. Critical reframing of the user's constraint

> "No scandots / height_scanner / foot sensors" applies to **policy input
> at deployment**, not to the **reward function during training**.

The Isaac Lab heightfield is queryable in sim; you can sample it at
proposed foot targets to compute reward without putting any of that data
in the obs. This is the standard "privileged-at-train, proprio-at-deploy"
trick (Lee 2020, Kumar RMA 2021). Source: §III.A of [Lee 2020][lee20]
("Learning quadrupedal locomotion over challenging terrain").

[lee20]: https://www.science.org/doi/10.1126/scirobotics.abc5986

This means: foothold-quality rewards using privileged terrain are
allowed under your constraint — which unblocks every reward formulation
in §4 below.

## 4. Foothold-quality reward terms used in published work

Each is a real reward expression from a cited paper. Implementation
typically consists of (a) querying the Isaac Lab heightfield at the
candidate foothold xy, (b) computing the reward at contact event, (c)
adding to the PPO advantage.

| Reward term | Math | Citation |
|---|---|---|
| **Sparse exponential touchdown error** | `r = κ·exp(−‖p_foot − p_target‖² / σ²)` gated on contact | Coholich 2025 §III.C; Choi/Raibo 2025 |
| **Foot-target reward (event-based)** | `κ_FT[2Π(h)+1]·Σ h_i,t·[1 + 0.5(1 − d_i,t/d_hit)]` activates within 7.5 cm of target with ≥5 N normal force | [Coholich 2025][coholich] eq. 4 |
| **Velocity-toward-target** | `r_VT = κ·Σ ḋ_i,t` | Coholich eq. 5 |
| **Foot stay** | reward keeping foot at hit target during stance | Coholich, Raibo |
| **Foot-slip penalty** | penalize >2 cm xy translation while in contact | universal in legged-RL |
| **Perpendicular impact velocity** | penalty on touchdown speed normal to local surface — improves stepping-stone success | [Choi/Raibo 2025][raibo] |
| **Edge-distance penalty** | `−κ·max(0, ε − d_edge)`, distance from target to nearest terrain discontinuity | [TAMOLS / RSL][tamols] |
| **Terrain-gradient safety classifier** | penalize stepping in cells with high-roughness/slope; or use as action mask | [Magaña 2019][magana]; [SafeSteps 2023][safesteps] |
| **Polygonal foot-overlap** | reward ∝ contact-area overlap with safe-foothold region | humanoid stepping-stone lit |
| **Non-foot-body collision** | `−κ·#collisions` | universal |

[raibo]: https://arxiv.org/abs/2506.02835
[tamols]: https://arxiv.org/abs/2206.14049
[magana]: https://arxiv.org/abs/1809.09759
[safesteps]: https://arxiv.org/abs/2307.12664

## 5. Foothold-target parameterization

| Paper | Target space | Dim | Feet at once |
|---|---|---|---|
| DeepGait | Cartesian xyz (world frame) | 3/foot | next swing only |
| RLOC | Cartesian xy + timing | ~3/foot | next 1–2 swings |
| Coholich 2025 | robot-frame xy, **all four feet** | **8 total** | all four (next pair active) |
| Choi/Raibo 2025 | robot-frame xy displacement from base, **n=2 lookahead** per foot | 4 D × 2 × foot | all four |
| PUMA | polar (distance + heading) | 2/foot × 2 feet | forefeet only |
| Walk-These-Ways | footswing height (gait param, not xy) | 1 global | N/A |
| SafeSteps | discrete index over candidate patches | log₂ K | one swing |

Dominant pattern: **robot-frame Cartesian xy for one or two upcoming
swings, all four feet predicted at once**. PUMA's polar parameterization
is newer and reportedly easier to regress.

## 6. Top 5 most relevant published architectures

| Architecture | HLP input | HLP output | LLP input | LLP train method | Foothold-aware LLP? | Real robot |
|---|---|---|---|---|---|---|
| [Coholich 2025][coholich] | proprio + yaw | 8-D xy footholds (4 feet) | proprio + 8-D rel-target + scan | PPO | **yes** (joint training) | sim only (Aliengo) |
| [Choi/Raibo 2025][raibo] | privileged heightmap | xy targets, n=2 lookahead/foot | proprio + base→target vec | PPO + adv. generator | **yes** | real (Raibo) |
| [RLOC (Gangapurwala 2022)][rloc] | proprio + exteroceptive + v_cmd | footstep plan | model-based MPC + RL corrections | PPO + WBC | LLP is MPC | real (ANYmal B/C) |
| [DeepGait (Tsounis 2020)][deepgait] | elevation map + base pose | xyz foothold sequence | foothold targets + base | PPO independent | **yes** | sim only (ANYmal) |
| [PUMA 2026][puma] | proprio + depth | polar foothold prior + v + 64-D latent | same (single policy) | multi-critic PPO | **yes** (single-stage) | real (Lite3) |

[rloc]: https://arxiv.org/abs/2012.03094
[deepgait]: https://arxiv.org/abs/1909.08399
[puma]: https://arxiv.org/html/2601.15995

## 7. Other relevant work

- [Margolis Walk-These-Ways (CoRL 2022)][wtw] — gait parameter conditioning
  (footswing height as global parameter, not xy footholds). Real Go1.
  Useful as the lowest-cost extension (Option C above).
- [Cheng Extreme Parkour (ICRA 2024)][parkour] — single-stage end-to-end,
  depth + scandots. No foothold-target hierarchy.
- [Zhuang Robot Parkour Learning (CoRL 2023)][rpl] — same structure.
- [Zhuang ABS (RSS 2024)][abs] — agile/recovery hierarchy, **velocity** is
  the LLP signal, not footholds. Real Go1.
- [Ji RAL 2022][ji] — proprio-only, single-stage, concurrent state estimator.
  This is essentially what `unitree_rl_lab.Unitree-Go2-Velocity` is.
- [Bellicoso 2018][bellicoso] / ALMA — classical MPC + foothold pipeline.
- [Wermelinger IROS 2016][wermelinger] — original RSL elevation-map
  traversability cost (slope, step, roughness) — most learned foothold
  rewards reduce to recomputations of these classical costs.
- [Magaña 2019][magana] — CNN classifier predicts safe foothold
  corrections; HyQ.
- [Hierarchical Vision Navigation (Sensors 2023)][hvn] — auto-annotated
  supervised, not RL.
- [github.com/AlmondGod/go2-hrl][go2hrl] — small Go2 + TrajOpt project,
  not research-grade.

[wtw]: https://arxiv.org/abs/2212.03238
[parkour]: https://arxiv.org/abs/2309.14341
[rpl]: https://arxiv.org/abs/2309.05665
[abs]: https://arxiv.org/abs/2401.17583
[ji]: https://arxiv.org/abs/2202.05481
[bellicoso]: https://ieeexplore.ieee.org/document/8460731
[wermelinger]: https://ieeexplore.ieee.org/document/7759278
[hvn]: https://www.mdpi.com/1424-8220/23/11/5194
[go2hrl]: https://github.com/AlmondGod/go2-hrl

## 8. Two-stage training mechanics (rates, freezing, training distribution)

- LLP runs at **50 Hz** in our setup (decimation 4 on 200 Hz physics);
  most published Isaac Lab policies use the same.
- HLP runs at **5–25 Hz** (Coholich: once per stride; Raibo tracker: 100 Hz
  on a 2-step horizon).
- For staged training, the LLP must see foothold targets *during its
  training* (not just at deploy) — this is what every successful
  foothold-conditioned LLP does. Otherwise the LLP has no learned
  function from those obs dims.
- Joint vs. staged: published work splits roughly 50/50. DeepGait trains
  independent. Coholich freezes LLP after step 1, then optimizes HLP
  via value-gradient ascent (no HLP training, just optimization). Raibo
  uses a separately-trained adversarial generator.

## 9. Path forward — what a future agent should evaluate

If this becomes the next training run, the literature-supported choices
in priority order:

1. **Option A retrofitted** — extend `unitree_rl_lab`'s Go2-Velocity obs
   space with `next_foot_target_xy_per_foot` (8 dims), keep velocity
   command obs, add a foothold-target reward (sparse exponential at
   contact, eq. 4 of Coholich). Train from scratch. ~3 k iters per
   published norms. Deploy with footholds emitted by an HLP that uses
   L1-derived terrain (the user's existing path). This is the
   Coholich + Raibo template.

2. **Option B (low-risk)** — HLP outputs velocity, not footholds.
   Current LLP is reusable as-is. Foothold-quality rewards apply at the
   HLP stage instead. Pattern: ABS.

3. **Option C** — extend with Walk-These-Ways gait parameters. Smallest
   obs delta after option B; published recipe; real Go1.

The user has expressed interest specifically in foothold targets, which
points at Option A. The foothold-quality reward terms in §4 directly
address the user's stated goal of "rewarding the robot for picking
good footholds."

## 10. Pass 2 findings — implementation-grade detail

Verified by going back to primary papers for exact obs vectors, networks, hyperparameters.

### 10.1 Choi/Raibo 2025 (the strongest sim-to-real template)

- **Actor obs** (167-D): proprio (orientation, ang vel, joint pos/vel) + 3-step joint history at 10/20/30 ms back + **displacement vectors to next n=2 footholds per foot in robot frame** + estimated 3-D linear velocity.
- **Networks**: actor/critic MLP `[512, 128]`; GRU(128) + MLP `[64, 16]` state estimator; CVAE map generator encoder `[512, 128]` / decoder `[128, 512]`.
- **PPO**: 16 epochs, γ=0.995, λ=0.95, lr=2e-4, max grad norm 0.5.
- **Reward weights (verbatim)**: sparse target k_ts1=9.4, target progression k_ts2=0.97, distance scaling k_ts3=6.0, dense target k_td=0.30. Foot-slip penalty (>2 cm in contact). Perpendicular-impact-velocity penalty.
- **Curriculum**: two-stage. Stage 1 fixed; stage 2 competitive CVAE retraining triggered when **success > 9.3/10 stepping stones**.
- **Domain rand** (10 channels): PD ±10%, control delay 0–30%, obs noise ±10%, init state ±10%, history obs ±10%, foot mass ±7%, base mass 0–40%, COM 0–15 cm, inertia 0–50%, friction 0.4–1.0.
- **Sim-to-real failure**: at 4 m/s, foothold target leaves camera FOV → resolved with pre-acquired heightmap + mocap (NOT autonomous). Real-world Go2 deployment may need similar handling.
- **No open code.**

### 10.2 START 2025 (arXiv:2512.13153) — most relevant new entry

- Go2-class robot, sparse footholds at 1.5 m/s, **only onboard egocentric depth** (Go2-class deployment constraint).
- Proprio obs = 45-D (ω³ + g³ + cmd³ + q¹² + q̇¹² + a¹²). TR-Net = CNN(depth 60×60) → GRU → MLP + U-Net heightmap decoder. I-E estimator fuses GRU(proprio) + CNN(heightmap) via transformer encoder.
- 3072 envs on A6000, 10k iterations / 16.6 h.
- **Reward weights**: lin-vel 1.5, ang-vel 0.5, collision −10, stumble −1, **foot-edge −1.0** with 2.5–5.0 cm tolerance bands [1.0, 0.5].
- **Curriculum**: probabilistic transition `p_advance = clip((T − T_start) / (T_end − T_start) · p_max, 0, p_max)`.
- **Ablations worth knowing**: no-heightmap → "single rigid action pattern"; no-temporal-memory → cannot infer occluded terrain; no-edge-penalty → frequent slips.
- Not on GitHub yet.

### 10.3 BeamDojo 2025 (arXiv:2502.10363) — humanoid, but importable

- **Double-critic**: one for dense locomotion reward, one for sparse foothold reward. Each normalized independently. Foothold reward `r_foothold = −Σ_i C_i · Σ_j 𝟙{d_ij < ϵ}` punishes contact samples outside safe regions.
- **Curriculum advance**: 3 successful traversals in a row.
- Project: why618188.github.io/beamdojo (no code yet).

### 10.4 Risky Terrains (arXiv:2311.10484)

- Heightmap 25×16 @ 7 cm grid, MLP [512, 256, 128], 4096 envs, 48 steps/iter.
- **Reward weights numeric**: pos-track 10, head-track 5, termination −200, aggressive-motion −5, torque −2e-5, contact −2.5e-5.
- **Two-stage generalist→specialist** (NOT teacher-student): finetune from Stones-Everywhere generalist into Stones-2Rows / Balance-Beams specialists.
- No explicit foothold conditioning — relies on heightmap and end-to-end discovery.
- Lesson: Risky-Terrains explicitly **disables the standard "demote on failure"** rule because it interacts poorly with sparse rewards.

### 10.5 Open code paths for plug-and-play

| Need | Repo | Notes |
|---|---|---|
| Base Go2 RL in Isaac Lab v2.3.2 | https://github.com/unitreerobotics/unitree_rl_lab | Already cloned; extend velocity task to add foot-target obs + sparse reward. |
| Stepping-stone terrain | `isaaclab.terrains.HfSteppingStonesTerrainCfg` (in https://github.com/isaac-sim/IsaacLab) | Exists and works. |
| Hierarchical scaffolding | https://github.com/LucaFrat/Anymal_Navigation | Adapt: replace velocity-cmd HLP output with foothold-target output. |
| Classical HLP (foothold MPC) | https://github.com/leggedrobotics/ocs2 + https://github.com/qiayuanl/legged_control | OCS2 BSD-3; legged_control needs Go2 URDF patch. **TAMOLS itself is NOT released.** |
| Forward dynamics model | https://github.com/leggedrobotics/fdm | Useful for Coholich-style HLP planning. |
| RL extension scaffolding | https://github.com/fan-ziqi/robot_lab | Cleanest external-extension pattern. |

### 10.6 Confirmation: no open Isaac Lab task implements foothold-target conditioning

Verified by searching every public Go2/quadruped Isaac Lab repo (`unitree_rl_lab`, `fan-ziqi/robot_lab`, `iit-DLSLab/basic-locomotion-isaaclab`, `LucaFrat/Anymal_Navigation`, `huangfq07/IsaacLab-Quadruped-Locomotion`, `mturan33/isaaclab-anymal-locomotion`, `dyumanaditya/isaac-quad-loco`, `abizovnuralem/go2_omniverse`, `Zhefan-Xu/isaac-go2-ros2`, `CLeARoboticsLab/go2_isaac_ros2`). No `foot_target` / `foothold_target` / `next_foothold` obs term. The user would be the first.

`LucaFrat/Anymal_Navigation` is the closest analog — hierarchical RL planner over rough terrain, but HLP outputs velocity commands, not foothold targets.

---

## 11. Pass 3 — "just add a reward, no architecture change" route

User question: *"reward for putting a foot in the right place might be enough"*. Audited the literature for foot-target rewards in **single-stage** policies (no separate HLP).

### 11.1 The smoking gun — Walk-These-Ways' Raibert-heuristic reward

[Margolis & Agrawal, CoRL 2022](https://arxiv.org/abs/2212.03238), [code](https://github.com/Improbable-AI/walk-these-ways), `_reward_raibert_heuristic` in `rewards.py`. The recipe:

- Compute desired foot-landing xy **from base velocity + commanded gait phase**:
  `p*_xy = p_hip_xy + (T_stance / 2) · v_base + k · (v_base − v_cmd)`
- Penalize during stance touchdown: `−κ · ‖p_foot,xy − p*_xy‖²`
- **No foot-target obs** — only velocity command + gait phase.
- Real Go1 sim-to-real success.

This is the **lightest possible extension** of the user's current `Unitree-Go2-Velocity` LLP. No architecture change, no new obs, just a new reward term. Battle-tested. The simplest answer to the user's question.

### 11.2 Pedipulate (foot-as-end-effector for ANYmal) — when foot-target obs IS in the obs

[Arm et al., ICRA 2024, arXiv:2402.10837](https://arxiv.org/abs/2402.10837). Single PPO policy. Obs includes 3-D target for **one** foot. Reward: `R_e = 15·exp(−‖r_f − r_f*‖² / 0.8)`. Other three feet emerge as base-stabilizing gait. **One foot, sparse target — works.** Per-step swing reference for all four feet — not what they tried.

### 11.3 AMP-style adversarial reward on foot trajectories

[AMP_Locomotion, Escontrela et al., IROS 2022](https://xbpeng.github.io/projects/AMP_Locomotion/index.html). Discriminator over (s, s′) where s includes foot positions/velocities. Style reward, not pointwise. Successfully trained on A1, ANYmal, Go1. Useful when the goal is "natural-looking foot trajectories" rather than "exact target positions".

### 11.4 Counter-evidence — when foot-target reward FAILS

- **MULE** ([arXiv:2505.00488](https://arxiv.org/html/2505.00488v1)): rigid per-foot xy tracking rewards over-constrain the policy on uneven terrain, hurting robustness vs. velocity-only rewards.
- **Coholich end-to-end baseline** ([arXiv:2506.20036](https://arxiv.org/abs/2506.20036)): single-stage policy with foot-target obs + reward works on flat/easy terrain but degrades on discontinuous terrain. Hierarchical version beats it.

The recurring failure mode: a hard `‖p_foot − p*‖²` reward fights the contact dynamics on irregular terrain. Either ignored (if down-weighted) or trips the robot (if up-weighted).

### 11.5 The bottom-line answer to the user's question

**For heuristic-derived foot targets** (compute target xy automatically from velocity + gait phase, à la WTW): yes, just adding a reward is well-attested and works. Lightest-cost win. Add `_reward_raibert_heuristic` and `_reward_foot_clearance` to `unitree_rl_lab.Unitree-Go2-Velocity`'s reward config.

**For externally-commanded foot targets** (HLP emits xy → LLP rewarded for hitting it): the literature is ambiguous-to-negative. Pedipulate works for **one** foot with target-as-obs. Coholich's no-obs/no-HLP baseline degrades. MULE warns of over-constraint. There's no working precedent for "commanded footstep xy reward without target obs and without HLP".

---

## 12. Pass 4 — body-part target poses (generalization beyond feet)

User question: *"also search for body part target poses"*. Audited literature for tracking arbitrary body-link poses, not just feet.

### 12.1 DeepMimic family (the canonical recipe)

[DeepMimic, Peng SIGGRAPH 2018](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf), [code](https://github.com/xbpeng/DeepMimic). Per-link relative position + quaternion + velocities in root-local frame, plus phase φ. Reward = weighted sum of:
- `r_p` joint quaternion difference (weight 0.65)
- `r_v` joint-velocity L2 (0.10)
- `r_e` end-effector Cartesian (hands/feet) (0.15)
- `r_c` center-of-mass (0.10)

**Key insight**: joint-angle tracking handles "shape", end-effector tracking handles "where in space", COM term anchors the base. This decomposition recurs across the family.

### 12.2 Quadruped-portable extensions

- **AMP** ([Peng 2021](https://xbpeng.github.io/projects/AMP_Locomotion/index.html)) replaces pointwise reward with discriminator over (s, s′). Successful on A1, ANYmal, Go1, and downstream Go2 follow-ups (BCAMP, CAMP, [Multi-Skill CAMP](https://arxiv.org/html/2509.21810)). More forgiving than pointwise.
- **PHC** ([Luo ICCV 2023](https://github.com/ZhengyiLuo/PHC)): per-link 3D position differences for ~24 SMPL-aligned body links + AMP discriminator. Humanoid.
- **BeyondMimic** ([arXiv:2508.08241](https://arxiv.org/abs/2508.08241), [code](https://github.com/HybridRobotics/whole_body_tracking)): three regularization terms + one unified task reward. Unitree G1. Single-hyperparameter set across cartwheels/spin-kicks. Tracking layer is morphology-agnostic.

### 12.3 Whole-body teleop with multi-keypoint targets

- [OmniH2O / H2O](https://github.com/LeCAR-Lab/human2humanoid) (CoRL 2024): **23 body-keypoint 3D positions** as universal interface, distilled teacher→student. Humanoid.
- [ExBody2 (arXiv:2412.13196)](https://arxiv.org/abs/2412.13196): explicitly **decouples keypoint tracking (positional) from velocity tracking** — relaxes lower-body to velocity targets and keeps upper-body as positional keypoints because rigid lower-body positional tracking destroys sim-to-real. **This is the most important counter-evidence for tracking all four feet positionally.**

### 12.4 Goal-conditioned legged manipulation

- [Pedipulate (arXiv:2402.10837)](https://arxiv.org/html/2402.10837): one foot target. ANYmal. Works.
- [WB-EE Pose Tracking (arXiv:2409.16048)](https://arxiv.org/pdf/2409.16048): obs = 45-D base + **9-D pose command encoded as 3 keypoints** (avoids quaternion discontinuities). Reward = 13·R_t + 80·R_p (progress) + 0.015·R_f + 0.4·R_q. Only EE has pose target. ANYmal D + DynaArm.
- [Multi-critic Twist Tracking (arXiv:2507.08656)](https://arxiv.org/html/2507.08656v1) — **most relevant prior art**. Three critics: (a) locomotion (base velocity, height, roll/pitch); (b) manipulation (EE 6D twist); (c) contact-schedule (per-foot swing height + contact phase). Single-critic ablations "completely fail to learn locomotion" — base-orientation rewards conflict with reach. ETH RSL, RSS-W 2025. Code not released.

### 12.5 What this implies for the user's decomposition

The user's contemplated decomposition — *"track velocity AND base height + roll/pitch AND per-foot xy during swing"* — IS a published recipe, almost exactly **multi-critic 2025 minus the manipulator**. It also matches the implicit reward structure of standard `legged_gym` Anymal/Go1/Go2 configs (which already include `tracking_lin_vel`, `tracking_ang_vel`, `base_height`, `orientation`, `feet_air_time` / per-foot swing rewards).

**What's NOT standard** is making foot xy a tracked target rather than a swing-time bonus. Pedipulate (1 foot) and WB-EE (1 EE) demonstrate per-link Cartesian-target Gaussians work at σ ~ 0.05–0.8 m. PMTG-style hierarchies do this for all four feet via residuals on a Bézier trajectory generator. ExBody2 demonstrates rigid all-four-feet tracking on humanoids breaks sim-to-real.

### 12.6 Counter-evidence — failure modes for body-part target tracking

- **ExBody2**: rigid lower-body keypoint tracking destroys sim-to-real on humanoids. Solution: relax to velocity tracking on legs, keep positional only on upper body.
- **Multi-critic 2025**: pose-based EE tracking caused "jerky end-effector motions" — switched to twist (velocity) representation. Single-critic with combined reward fails due to objective conflict.
- General reviews ([athletic loco-manipulation](https://arxiv.org/html/2502.10894), [APEX](https://arxiv.org/html/2505.10022)): dense tracking objectives over-regularize exploration when reference data is poor.

### 12.7 Recommendation drawn from strongest evidence

For a Go2-class quadruped extending velocity tracking with body-part targets:

1. **Keep base as velocity targets** (v_xy, ω_z) + low-weight Gaussian on base height and projected gravity (roll/pitch implicit). Standard legged_gym already does this.
2. **For feet, prefer velocity / swing-phase tracking over rigid xy positional targets.** Pedipulate-style Gaussian (σ ≈ 0.1–0.3 m) works for 1 foot, but ExBody2 + Multi-critic-Twist demonstrate that rigidly tracking all four foot positions degrades sim-to-real.
3. **For stylistic naturalness without writing per-link rewards, drop in an AMP discriminator** on (s, s′). Existing velocity+height+rp rewards stay as task term. Replicated on Go1/Go2 multiple times.
4. **If you must track all four feet positionally, use multi-critic decomposition** (locomotion / contact-schedule / per-foot tracking) rather than summing into one critic.

Single-link target tracking (base height OR one swinging foot OR one EE) — well-supported. All-four-feet positional tracking — failure-prone, needs AMP or multi-critic.

---

## 13. Synthesized day-1 recipe options for the user's situation

Three concrete paths drawn from the strongest evidence, ordered by cost:

### Path α — Reward-only extension (lightest)

Drop into the existing `Unitree-Go2-Velocity` LLP:
- `_reward_raibert_heuristic`: penalty on touchdown xy vs. WTW Raibert formula. Weight ~0.5–1.0 vs 1.0 on velocity tracking.
- `_reward_foot_clearance`: Gaussian on swing foot height vs. desired clearance. Weight ~0.2.
- (Optional) AMP discriminator over (proprio, prev_action) using a reference dataset of natural Go2 walking / trotting clips.

No architecture change, no new obs, no HLP. Battle-tested by Walk-These-Ways. Should produce visibly better gait quality within the existing 24h training budget.

### Path β — Foothold-aware LLP retrain (medium)

Implement the Choi/Raibo template:
- Add to obs: 8 dims = robot-frame xy of next foothold target per foot (4 feet × 2-D xy), 1-step lookahead. Or PUMA-cheap variant: 4 polar scalars.
- Add reward: sparse exponential touchdown error (κ=9.4·exp(−‖p − p*‖²/σ²)), foot-stay during stance, foot-slip penalty.
- Footsteps emitted by a deterministic generator at training time (Raibert with terrain-corrected anchors).
- Train fresh ~3-10k iters.

Deploy with footstep generator running on real Go2 (uses L1 lidar + state estimator).

### Path γ — Two-stage with multi-critic (most engineering)

Three independent critics — locomotion / contact-schedule / per-foot tracking — each normalized separately. Multi-critic Twist 2025 template. Train fresh; HLP can be RL or classical (OCS2).

### What the user actually said

The user explicitly asked about (1) rewarding good footholds and (2) body-part target poses. Both pass-3 and pass-4 evidence point to the **same conclusion**: the lightest, most-evidence-supported path is **Path α** — add Raibert-heuristic foot reward + foot clearance reward to the existing LLP, no architecture change. If that's insufficient, Path β is the next-most-precedented step.

---

## 14. The permutation-invariant foothold problem

User's question, verbatim: *"the best method for single foot reaching a foothold (but any of the feet can reach any of the footholds)"*.

This is a different problem statement than every published recipe surveyed
in §6, §10, §11, §12. Worth treating carefully.

### 14.1 What the problem actually is

Standard published quadruped foothold tracking (DeepGait, Coholich, Choi/Raibo, PUMA) **pre-assigns each foot to its own target stream** — front-left foot tracks a sequence of FL targets, etc. The natural gait pattern dictates assignment.

The user is asking about a different formulation:
- A **set** of footholds available (e.g. stepping stones in a chain)
- **Any** of the 4 feet may step on **any** of the footholds
- The policy implicitly chooses the assignment

This breaks the standard one-to-one foot-target index mapping. With 4 feet × K footholds in the active window, there are O(4K) possible "next-step" choices instead of 4 (one per foot).

### 14.2 Why permutation-invariant assignment matters

- Sparse stepping stones where standard trot/walk can't reach the next stone with the "next" gait foot
- Sidestepping, turning, hopping over discrete terrain
- Recovery — when the natural foot has been pushed off-target, swap to a different foot
- Generalizes to "irregular feet" robots (3 legs, 5 legs) without architectural change

### 14.3 The literature is silent on this for quadrupeds

Surveyed every paper found in the four research passes. The closest precedents:
- **SafeSteps** (arXiv:2307.12664): discrete choice over candidate patches, but **one swing leg at a time, pre-assigned by gait phase**.
- **Pedipulate** (arXiv:2402.10837): one foot, one target, no permutation.
- **DeepGait / Coholich / Choi/Raibo / PUMA**: all pre-assign by foot index.
- **TAMOLS** classical optimization: assigns within optimization, but the assignment is fixed by gait scheduling.
- **Bipedal humanoid step planners** (ALLSTEPS, BeamDojo): one swing leg at a time, alternating L/R.

**No published quadruped paper does true permutation-invariant foot-to-foothold assignment.** This is open territory.

### 14.4 Architectural options (and their tradeoffs)

| Option | Mechanism | Cost | Risk |
|---|---|---|---|
| **(a) "Closest foothold gets credit"** | At each contact event, compute distance from landing foot to nearest unclaimed foothold; reward if within ε. Foothold is claimed, removed from active set. | Lowest. Pure reward shaping. | Policy may step all feet on the same closest stone; needs ordering constraint. |
| **(b) Hungarian assignment at train time** | At each step, solve optimal assignment between feet-in-swing and reachable footholds via Hungarian algorithm; evaluate touchdown reward under that assignment. | Medium. Stateful tracking + per-step solver. | Solver overhead; may not generalize to deploy where future footholds unknown. |
| **(c) Set-encoder (DeepSets / cross-attention)** | Encode the set of footholds with a permutation-invariant encoder; each foot attends over the set; reward triggers on any-foot, any-foothold contact. | High. New network architecture, attention over varying set size. | Increased training time; may overfit to attention pattern. |
| **(d) Discrete "active foothold" pointer** | Track which footholds are unclaimed; present the next K unclaimed in obs (sorted by distance from base); reward triggers on any foot landing within tolerance of an active one. | Low-medium. Fixed-K obs slot; stateful claim tracking. | Fixed K limits attention range. |
| **(e) Auxiliary assignment head** | Policy outputs joint actions AND a soft assignment over (foot, foothold) pairs. Reward includes assignment-conditioned tracking. | High. Two-headed policy, supervised assignment loss. | Hardest to tune. |

### 14.5 Recommended method (synthesized)

Combine (a) and (d) — **closest-foothold reward + nearest-K active foothold obs**, with a progress shaping term to enforce ordering:

**Obs additions to current `Unitree-Go2-Velocity` PolicyCfg** (~24 dims):
- `nearest_unclaimed_footholds`: K=4 nearest *unclaimed* foothold xyz positions in robot frame, sorted by distance from base center. 12 dims (4 × 3).
- `foot_positions_robot_frame`: current xyz positions of all 4 feet in robot frame. 12 dims (4 × 3).
- (optional) `foothold_visit_mask`: K-dim 0/1 vector encoding claim status, in case the encoder benefits from explicit claim signal.

**Reward additions** (drawn from Choi/Raibo and START):
- `foothold_landing` (sparse, event-based): on each swing→stance contact event for foot f at position p_f, find nearest unclaimed foothold s\* in F. If ‖p_f − s\*‖ < ε (e.g. 5 cm) AND normal force > 5 N, give reward `κ · exp(−‖p_f − s\*‖² / σ²)` and mark s\* claimed. **Permutation-invariant by construction**: the foot-id is irrelevant to the reward. κ = 9.4 (Choi/Raibo k_ts1).
- `foothold_progress` (dense): for each foot, reward the velocity component toward its nearest unclaimed foothold. Weight 0.30 (Choi/Raibo k_td).
- `off_foothold_landing` (penalty): contact event with ‖p_f − s\*‖ > ε for all s\* gets penalty −1.0 (START's edge-penalty weight).
- `claim_advance` (shaping): bonus when number of claimed footholds in episode increases — encourages ordering. Weight 0.5.
- Existing terms unchanged: `feet_air_time`, `feet_slide`, `joint_pos`, `action_rate`, `dof_pos_limits`.

**Network**: standard MLP unchanged from baseline; just larger input (24 extra dims). No attention, no DeepSets in v1 — fixed-K-sorted obs is sufficient for K=4.

**Curriculum**:
- Stage 1: dense stepping stones (spacing 0.30 m, stone diameter 0.30 m) — any-foot any-stone trivially satisfiable.
- Stage 2: medium (spacing 0.50 m, stones 0.20 m) — assignment starts mattering.
- Stage 3: sparse (spacing 0.70 m, stones 0.15 m) — flexible foot choice required.
- Advance criterion: success ≥ 9/10 stones traversed in episode (Choi/Raibo gate).
- Demote rule **disabled** (Risky-Terrains lesson — interacts poorly with sparse rewards).

**Training**: warm-start from current `model_*.pt` velocity-LLP weights via PPO finetuning, or train from scratch. Estimate ~3–10k iters per published norms.

**AMP discriminator (optional)**: bolt on AMP_Locomotion-style discriminator over (proprio, prev_action) using a reference dataset of natural Go2 walking. Keeps gait quality from collapsing into "weird step pattern that hits the stones."

### 14.6 Why this synthesis specifically

- **Permutation invariance via reward, not architecture**. The reward function is symmetric in foot index. The policy never sees per-foot target assignment — it sees the set, and emergent gait dynamics do the assignment.
- **Standard MLP, no attention**. Fixed-K nearest-unclaimed-sorted is the simplest permutation-invariant obs encoding that doesn't require a new network architecture. Empirical evidence from Pedipulate, Coholich, START — they all use vanilla MLPs successfully.
- **Closest-foothold matching is greedy local**. Combined with progress shaping, the policy can't "double-step" on the closest stone (it gets claimed) and is incentivized to advance through the chain.
- **Choi/Raibo κ values transfer well**. Reused as initial weights since the obs structure is similar (per-foot target → per-foot-and-set target).

### 14.7 Risks specific to this approach

- **Policy may converge to "stand still on first stone"** if exploration too low. Mitigation: start with high entropy, anneal slowly; ensure terrain time-out termination penalty is meaningful.
- **Off-foothold penalty fights contact dynamics**. Same failure mode as MULE / Coholich end-to-end. Mitigation: tolerance ε on the LIBERAL side (5 cm not 1 cm); scale penalty smoothly with distance rather than hard threshold.
- **Order-of-stones ambiguity**: if multiple unclaimed stones are equidistant, the assignment of "nearest" flickers. Mitigation: stable tie-breaking (by stone-id index) or hysteresis on claim assignment.
- **Out-of-distribution foothold geometry**: if real-world stones differ from training distribution. Mitigation: domain-randomize stone size, spacing, height variation.

### 14.8 Honest caveat

This combined recipe is **drawn from primary sources but not itself published** — no Go2 paper has done permutation-invariant foothold reaching with the exact obs+reward structure above. The components (sparse Gaussian reward, nearest-K obs, progress shaping, AMP discriminator) are individually published; the combination is novel.

The expected failure mode if this doesn't converge: the policy ignores the foothold obs (treats them as constant noise) because the velocity tracker baseline doesn't need them. This is the same risk as the lidar run §10 of the lidar experiment doc. Mitigation: train from scratch (not warm-start) AND ensure the foothold reward dominates the velocity tracking reward early in training.

---

## 15. Decision matrix for the next step

Three concrete paths, ordered by how directly they address the user's stated objective ("rewarding for picking good footholds, with any foot able to reach any foothold"):

| Path | Cost | Permutation-invariant? | Published precedent? | Match to user goal |
|---|---|---|---|---|
| **α — Reward-only WTW Raibert extension to current LLP** | Lowest | No (foot-indexed) | Yes (Walk-These-Ways) | Partial: improves gait, no foothold targeting |
| **β — Choi/Raibo retrain (per-foot target obs, fixed assignment)** | Medium | No (foot-indexed) | Yes (Choi/Raibo) | Partial: foothold tracking, but pre-assigned |
| **γ — Synthesized permutation-invariant (§14.5)** | Medium-high | **Yes** | No (synthesis of components) | Full match |

Path γ is the direct match to the user's question. It's also the most novel — the combination has not been published. Implementation in `unitree_rl_lab` would be:
- New env config `Unitree-Go2-Footholds` with stepping-stone terrain and foothold tracking
- New MDP obs functions: `nearest_unclaimed_footholds`, `foot_positions_rf`
- New MDP reward functions: `foothold_landing` (event-based), `foothold_progress`, `off_foothold_landing`, `claim_advance`
- Stateful per-env "claimed" tracker (extend env state)
- Curriculum hook to ramp stone spacing
- Estimated implementation: 200–400 LOC across 3 new files, ~1 day with the existing scaffolding

---

## 15b. Sim2real gap noted: sensor-rate / staleness / latency

For Jetson deployment, the current foothold task does **not** simulate:

- **Foothold planner staleness.** Real planner emits ~10–25 Hz; policy reads
  at 50 Hz; same value held 2–5 steps. Sim feeds fresh ground truth every
  step.
- **Sensor-to-policy latency.** ~5–20 ms on Jetson; sim is 0 ms.
- **Joint encoder / FK noise on `foot_positions`.** Real ~0.001 rad encoder
  noise; sim is noise-free.
- **State-estimator drift.** Real `base_ang_vel` and `projected_gravity`
  come from a Kalman/EKF that drifts; sim is direct-from-physics.

The base proprio terms have `Unoise` on `base_ang_vel`, `projected_gravity`,
`joint_pos_rel`, `joint_vel_rel` already, so most of the high-rate-sensor
gap is partially covered. The biggest unaddressed gap is **`nearest_footholds`
staleness** — for hardware deployment, add a domain-randomization layer that
holds the foothold obs for a random 0–5 policy steps before refreshing. ~30
LOC. Not implemented in v2; flag as Stage-3.5 work if/when sim-to-real is
attempted.

Standard recipe for deployable foothold policies (Choi/Raibo, Coholich) all
include planner-noise / latency randomization at training time. Without it,
the policy is brittle to real planner output.

---

## 16. Gaps and uncertainties

- No published Go2-specific paper does foothold-target HLP → proprio
  LLP. Open niche.
- Coholich is sim-only; Raibo is non-Go2. Direct sim-to-real precedent
  for Go2 + foothold conditioning doesn't yet exist in the open
  literature (as of May 2026).
- The `unitreerobotics/unitree_rl_lab` repo has zero foothold-related
  PRs or issues. Confirmed in the previous research pass.
- Curriculum design for foothold training (how to ramp difficulty:
  flat → sparse stepping stones → gaps → discrete pyramids) is
  under-documented in published papers — would require empirical tuning.
