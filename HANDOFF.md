# Handoff — Unitree Go2 RL training on this machine

**Read this whole document before touching anything.** Written so another
agent can pick up cold and know exactly what's where, what's been tried,
what's working, what's broken, and what to try next.

---

## 0. Project context (one paragraph)

User is training a Unitree Go2 quadruped to walk over custom terrains in
NVIDIA Isaac Lab. Training started from the standard `unitree_rl_lab` Go2
velocity-tracking task and evolved through several terrain variants. The
current open question is: **how do you get the policy to land its feet
specifically on the flat regions of obstacle-rich terrain, not on slopes
or peaks**. The latest run is on a custom "pyramid-ridged-stairs" terrain
and is in a degenerate local minimum (shuffling). Multiple alternative
research directions have been investigated and documented; some were
explicitly forbidden by the user (e.g. height_scanner / foot contact
sensors in the policy obs because they don't exist on real Go2 hardware).
The training pipeline is configured for sim-to-real on a Jetson Orin.

---

## 1. Machine

| Field | Value |
|---|---|
| Hostname | `doubled` |
| User | `anyone` |
| Project root | `/home/anyone/projects/doga` |
| OS | Ubuntu 24.04.3 LTS (kernel 6.14.0-37-generic) |
| CPU | Intel Core i5-14400F, 10 cores / 20 threads |
| RAM | 64 GB |
| GPUs | 2× RTX 5060 Ti 16 GB (Blackwell, sm_120) |
| NVIDIA driver | 580.126.09 (CUDA 13.0 driver) |
| Disk | 457 GB root, ~140 GB used at writing |
| LAN IP | `192.168.1.127` |

**GPU policy used by every command in this project: `CUDA_VISIBLE_DEVICES=1`.** GPU 0 is reserved for the user's other work (`cnn3d/train.py`).

---

## 2. Software stack

Conda env is the only Python env we use. **Always activate before running anything.**

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME            # avoids leaking system CUDA
export OMNI_KIT_ACCEPT_EULA=YES   # already persisted via conda env config vars
```

| Component | Version | Where |
|---|---|---|
| Python | 3.11.15 | `~/miniforge3/envs/env_isaaclab` |
| Isaac Sim | 5.1.0.0 | pip in env, NVIDIA wheel index |
| Isaac Lab | v2.3.2 (commit 37ddf62) | `~/IsaacLab` (editable installs) |
| PyTorch | 2.7.0+cu128 (Blackwell sm_120 confirmed) | pip in env |
| rsl-rl-lib | 3.1.2 | pip in env |
| skrl | 2.0.0 | pip in env |
| unitree_rl_lab | editable, modified extensively | `~/unitree_rl_lab` |
| unitree_model (Go2 USD via Git-LFS) | latest of main | `~/unitree_model` |

**Critical pinned env var if torch ever gets reinstalled:** must use
`--index-url https://download.pytorch.org/whl/cu128`, NOT the cu126 default.
Verify with `python -c "import torch; print(torch.cuda.get_arch_list())"` —
must include `sm_120`.

---

## 3. Repos cloned

```
~/IsaacLab/                    # NVIDIA Isaac Lab v2.3.2 — DO NOT MODIFY
~/unitree_rl_lab/              # Modified extensively. See §6.
~/unitree_model/                # Unitree's Go2 USD via Git LFS
~/projects/doga/                # Project work dir, all docs + launchers + logs
```

`unitree_ros` is referenced via env var `UNITREE_ROS_DIR` but not cloned (URDF spawn path is unused).

---

## 4. Documentation already written (READ FIRST)

All in `~/projects/doga/`:

| Doc | What it covers |
|---|---|
| `INSTALL.md` | The end-to-end install procedure. Already ran. |
| `LIDAR-RESEARCH.md` | Research on lidar-based perceptive locomotion. Lidar variant was tried and abandoned. |
| `HIGH-LEVEL-RESEARCH.md` | 16 sections on hierarchical RL / foothold targeting / body-part target poses. The synthesised recipes for permutation-invariant footholds and the rationale behind §15b sim2real gaps live here. **Most important research doc.** |
| `RUN-LIDAR-EXPERIMENT.md` | Record of the lidar experiment (what didn't work). |
| `JETSON-DEPLOY.md` | Deployment guide for the Jetson Orin on the Go2 — what to ship, the 65-d obs vector layout, ONNX runtime, joint SDK ordering, the foothold-planner gap. |
| `HANDOFF.md` | This document. Supersedes earlier shorter version. |

The user has been clear about a few things — **respect these constraints**:
- Real Go2 sensors only at deploy: proprioception (IMU + joint encoders) +
  L1 lidar (if equipped). **No height_scanner, no foot contact sensors,
  no scandots in the policy obs.** Privileged terrain info IS allowed at
  training time (RMA / privileged-at-train pattern) for reward computation.
- Brief responses preferred (small terminal pane).
- They set the design rules; do not propose forbidden things.
- Report facts and primary sources, not assumptions.
- When given a directive, execute and document; do not editorialize.

---

## 5. Training history (chronological)

Every training run is recorded under `~/projects/doga/logs/rsl_rl/<task>/<run_ts>/` with
`model_*.pt` checkpoints (every 100 iters), `events.out.tfevents.*` (TensorBoard),
`params/` (config snapshot), and `git/` (diff at run start). All long-running
runs were launched with `setsid nohup` so they survive logout. Process tree
verification: `pstree -p -s <pid>` should top out at `systemd(1)` not bash/claude.

Runs in order:

| # | Stage | Task | Run dir | Iter range | Wall time | Outcome |
|---|---|---|---|---|---|---|
| 0 | Smoke | `Unitree-Go2-Velocity-Lidar` | various early `2026-05-02_00-3*` | 0 → small | minutes | env wiring tests |
| 1 | Lidar Phase 1 | `Unitree-Go2-Velocity-Lidar` | `unitree_go2_velocity_lidar/2026-05-02_00-39-44/` | 0 → 7300 | ~8.5 h | bad_orientation 90%, LR pinned at floor — lidar obs is uninformative when robot is upright |
| 2 | Lidar Phase 2 (resumed) | same | `unitree_go2_velocity_lidar/2026-05-02_10-00-20/` | 7300 → 12300 | ~6.7 h | bad_orientation 79%, terrain_levels stuck at 0 — abandoned |
| 3 | Baseline Stage 1 (flat) | `Unitree-Go2-Velocity` | `unitree_go2_velocity/2026-05-02_16-47-49/` | 0 → 2600 | ~5 h | converged on flat ground; `lin_vel_cmd_levels` STUCK at 0.10 (curriculum bug) |
| 4 | Baseline Stage 1 (resumed, curriculum fixed) | same | `unitree_go2_velocity/2026-05-02_21-51-16/` | 2600 → 13400 | ~9 h | `lin_vel_cmd_levels` saturated 1.0 by iter ~12k, full ±1 m/s walker on flat |
| 5 | Baseline Stage 2 (rough) | same task, `terrain=ROUGH_TERRAINS_CFG` + `feet_clearance` reward | `unitree_go2_velocity/2026-05-03_18-37-15/` | 13400 → **25600** | 24 h | Rough-terrain walker. **`model_25600.pt` is the best general-purpose locomotion checkpoint we have.** terrain_levels plateau ~0.85, bad_orientation 6%, full ±1 m/s |
| 6 | Foothold Stage-3 v1 (broken) | `Unitree-Go2-Footholds` | `unitree_go2_footholds/2026-05-04_22-50-51/` | 0 → 56k | 22 h | broken — `off_foothold_landing` was continuous instead of event-based; killed |
| 7 | Foothold Stage-3 v2 (fixed) | same | `unitree_go2_footholds/2026-05-05_21-37-06/` | 0 → **62600** | 24 h | walks well on flat, claims ~50% of virtual footholds per episode. Visualisation markers wired into play.py |
| 8 | Foothold v2 resume | same | `unitree_go2_footholds/2026-05-06_23-26-37/` | 62600 → ~few k | killed early (was just polishing) | n/a |
| 9 | Trapezoidal bumps | `Unitree-Go2-Velocity` (custom terrain) | `unitree_go2_velocity/2026-05-07_23-28-00/` | 25600 → **40300** | 24 h | Walks 80–100 cm-period 45° trapezoidal bumps at full ±1 m/s, falls 13%, slides on slopes (μ=1.0 friction was right at slope-grip threshold) |
| 10 | Pyramid-ridged-stairs (current) | same | `unitree_go2_velocity/2026-05-09_01-07-01/` | 40300 → 46500+ | RUNNING (12h+ in) | **DEGENERATE** — `feet_clearance=0.03`, `terrain_levels=0.02`, robot shuffles instead of stepping |

Backup of all rsl_rl runs (excluding run 10 which was created after backup) at
`~/projects/doga/checkpoints_backup/` (5.2 GB, 391 ckpts).

---

## 6. Code modifications — exhaustive list

### 6.1 `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/`

#### `assets/robots/unitree.py`
- Lines 20-21: changed placeholder paths to env-var lookup with HOME defaults:
  ```python
  UNITREE_MODEL_DIR = os.environ.get("UNITREE_MODEL_DIR", os.path.expanduser("~/unitree_model"))
  UNITREE_ROS_DIR = os.environ.get("UNITREE_ROS_DIR", os.path.expanduser("~/unitree_ros"))
  ```

#### `tasks/locomotion/mdp/curriculums.py`
- **Critical bug fix** (the curriculum was the root cause of slow training in run 3).
- `lin_vel_cmd_levels` and `ang_vel_cmd_levels` had a global-step gate
  `if env.common_step_counter % env.max_episode_length == 0` that fired only
  rarely with non-empty `env_ids`, preventing advancement.
- Replaced gate with a per-env stored last-advance-step rate limiter using
  attributes `_lin_vel_cmd_levels_last_step` and `_ang_vel_cmd_levels_last_step`.

#### `tasks/locomotion/mdp/observations.py`
- Original `gait_phase` retained.
- Added `lidar_distances(env, sensor_cfg, max_range=30, min_range=0.05, noise_std=0.02, dropout_prob=0.10, log_scale=True)` — for the lidar variant. Computes per-ray distances from `RayCaster.data.ray_hits_w`, applies Gaussian noise + Bernoulli dropout + log1p compression. Used by `Unitree-Go2-Velocity-Lidar` task only.
- Added `nearest_unclaimed_footholds(env, asset_cfg, k=4)` — returns K nearest unclaimed foothold xy in robot frame (sorted by distance). Used by foothold task.
- Added `foot_positions_robot_frame(env, asset_cfg)` — kinematic foot xyz in robot frame via FK on joint state. Allowed under user's "real Go2 sensors only" rule.

#### `tasks/locomotion/mdp/rewards.py`
- Original rewards retained, including:
  - `feet_height_body`, `foot_clearance_reward`, `feet_too_near`, `feet_contact_without_cmd`, `feet_gait` etc.
- **`foot_clearance_reward` is used in the modified `velocity_env_cfg.py`** at weight 1.0 with target_height=0.08, std=0.05, tanh_mult=2.0.
- Added foothold rewards (used by `Unitree-Go2-Footholds` task only):
  - `foothold_landing(env, sensor_cfg, asset_cfg, tolerance, sigma, contact_threshold)` — event-based, MUST run first in RewardsCfg ordering. Drives `update_claims` from `foothold_state` so other foothold rewards see the just-claimed flags.
  - `foothold_progress(env, asset_cfg, contact_sensor_cfg)` — dense, per-swing-foot velocity component toward nearest unclaimed foothold.
  - `off_foothold_landing(env, sensor_cfg, asset_cfg, tolerance, contact_threshold)` — smooth penalty, **event-based** (gated on `_foothold_just_landed`), graded by distance excess. v1 was continuous-contact and broke training.

#### `tasks/locomotion/mdp/foothold_state.py` (NEW)
- Per-env state buffers attached lazily to env on first call.
- Buffers: `_foothold_positions (E,S,2)`, `_foothold_claimed (E,S)`, `_foothold_active (E,S)`, `_foothold_just_claimed (E,S)`, `_foothold_was_contact (E,4)` initialised TRUE (robot spawns in stance, NOT a transition), `_foothold_just_landed (E,4)`, `_foothold_claim_count_prev (E,)`.
- `reset_footholds(env, env_ids, n_stones, spacing, jitter)` — episode-reset hook. Generates a forward chain of stones in **robot's initial yaw direction** (not world +X — important fix; robots spawn with random yaw).
- `update_claims(env, sensor_cfg, asset_cfg, tolerance, contact_threshold)` — per-step claim update. Detects swing→stance transitions (was-not-contact AND is-now-contact), finds nearest unclaimed foothold, claims if within tolerance.
- `update_markers(env)` — opt-in visualisation. Yellow sphere = unclaimed, green = claimed. Gated by `env._foothold_show_markers = True`. Only set in play.py for tasks with "Foothold" in the name.

#### `tasks/locomotion/robots/go2/__init__.py`
Three gym IDs registered:
- `Unitree-Go2-Velocity` — standard task (modified terrain, see below)
- `Unitree-Go2-Velocity-Lidar` — lidar variant (abandoned, kept for reference)
- `Unitree-Go2-Velocity-Lidar-FlatPlay` — flat play variant for lidar
- `Unitree-Go2-Footholds` — foothold task

#### `tasks/locomotion/robots/go2/velocity_env_cfg.py`
- Imports `PYRAMID_RIDGED_STAIRS_CFG`, `TRAPEZOIDAL_BUMPS_CFG` from custom terrains module.
- `RobotSceneCfg.terrain.terrain_generator` is currently `PYRAMID_RIDGED_STAIRS_CFG` (was `COBBLESTONE_ROAD_CFG` originally, then `ROUGH_TERRAINS_CFG` for Stage 2, then `TRAPEZOIDAL_BUMPS_CFG`).
- Friction in `physics_material`: currently `static_friction=0.75`, `dynamic_friction=0.75` (originally 1.0, then 0.5).
- Added `feet_clearance` reward in `RewardsCfg` (mdp.foot_clearance_reward, weight 1.0, target 0.08m).
- `RobotPlayEnvCfg.__post_init__` adds side-on viewer config:
  ```python
  self.viewer.origin_type = "asset_root"; self.viewer.asset_name = "robot"
  self.viewer.eye = (0, 4, 0.5); self.viewer.lookat = (0, 0, 0.3)
  ```

#### `tasks/locomotion/robots/go2/velocity_lidar_env_cfg.py` (NEW)
- Lidar variant of the velocity task. Adds `lidar_scanner` RayCasterCfg with `LidarPatternCfg` (32 channels × 72 azimuth = 2304 rays, 360°×90° upper hemisphere, 11 Hz). Asymmetric actor-critic (critic gets clean lidar). Inherited rough terrain.

#### `tasks/locomotion/robots/go2/velocity_footholds_env_cfg.py` (NEW)
- Foothold task. Currently uses `ROUGH_TERRAINS_CFG` (was originally flat plane in v1/v2 — change happened mid-experiment).
- Extra obs: `nearest_footholds` (8-dim, K=4 stones × xy in robot frame), `foot_positions` (12-dim, 4 feet × xyz). Both available in policy and critic.
- Extra rewards: `foothold_landing` (κ=9.4, MUST be first), `foothold_progress` (0.3), `off_foothold_landing` (0.3). `claim_advance` was removed in v2 as redundant with `foothold_landing`.
- `FootholdEventCfg.reset_footholds` event runs at `mode="reset"` after `reset_base` — calls `foothold_state.reset_footholds`.

#### `terrains/__init__.py` (NEW)
Re-exports `TRAPEZOIDAL_BUMPS_CFG`, `PYRAMID_RIDGED_STAIRS_CFG` and their cfg classes.

#### `terrains/trapezoidal_bumps.py` (NEW)
- `HfTrapezoidalBumpsTerrainCfg` + `trapezoidal_bumps_terrain` heightfield function.
- Profile per period: flat-low (30 cm) + 45° up-ramp + flat-top (30 cm) + 45° down-ramp.
- Constants: `horizontal_scale=0.05`, `vertical_scale=0.005`, `slope_threshold=1.5` (preserves 45° slopes — must be > 1.0).
- `bump_height_range=(0.10, 0.20)`, `border_width=5.0`, grid 10×10 of 8m × 8m tiles.

#### `terrains/pyramid_ridged_stairs.py` (NEW)
- `HfPyramidRidgedStairsTerrainCfg` + `pyramid_ridged_stairs_terrain`.
- **All-45° geometry** — no vertical risers anywhere.
- Per step ascending (+X): 45° up-ramp (rises step+ridge height) → 45° small down-ramp (drops by ridge_height) → flat tread.
- Pyramid is mirrored about the peak: ascending half + flat top + reversed descending half.
- Current params: `step_height_range=(0.05, 0.10)`, `ridge_height=0.05`, `tread_width=0.20` (halved from 0.40), `n_steps_per_side=20` (5x extension), tile `size=(20, 40)` (5x extension in BOTH X and Y), `border_width=2.0`, grid 10×1.
- Heightfield density: 7.8 M triangles total — within PhysX limits.

### 6.2 `~/unitree_rl_lab/scripts/rsl_rl/play.py`
- Two patches:
  1. `from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint` wrapped in try/except since this module doesn't exist in Isaac Lab 2.3.2 (added later upstream).
  2. After env creation, sets `env.unwrapped._foothold_show_markers = True` if "Foothold" is in the task name. Enables the foothold visualisation markers (yellow=unclaimed, green=claimed) in play mode only.

### 6.3 `~/projects/doga/` launcher scripts

All scripts use `setsid nohup` pattern in their invocation (you must call them like `setsid nohup ~/projects/doga/launch-X.sh </dev/null >/dev/null 2>&1 & disown`). Each script:
- Activates conda env, unsets CUDA_HOME
- Sets CUDA_VISIBLE_DEVICES=1
- Creates timestamped log dir under `logs/`
- Uses `timeout --preserve-status --signal=TERM 86400` (24h) wrap

| Script | Purpose |
|---|---|
| `00-system-deps.sh` | sudo apt install cmake build-essential vulkan-tools |
| `01-install-git-lfs.sh` | sudo apt install git-lfs |
| `launch-12h-train.sh` | 12h fresh start of `Unitree-Go2-Velocity-Lidar` (historical) |
| `launch-24h-resume.sh` | 24h lidar resume (historical) |
| `launch-baseline-resume-curriculum-fix.sh` | Stage-1 → 2 baseline transition (historical) |
| `launch-24h-baseline.sh` | Stage-1 baseline (historical) |
| `launch-24h-footholds.sh` | Foothold task fresh, 24h |
| `launch-24h-footholds-resume.sh` | Foothold task resume (historical) |
| `launch-24h-trapezoidal.sh` | Trapezoidal-bumps fresh, resume from `model_25600.pt` |
| `launch-24h-pyramid.sh` | **Current**: pyramid-ridged-stairs, 24h, resume from `model_40300.pt` |

To create a new launcher, copy `launch-24h-pyramid.sh` and edit `--load_run`, `--checkpoint`, `LOG_DIR`. The relevant timeout is `86400` for 24h; halve for 12h.

---

## 7. How to find things

### Latest checkpoint of any task
```bash
ls -t ~/projects/doga/logs/rsl_rl/<task_id_lowercase>/*/model_*.pt | head -1
```

E.g. for `Unitree-Go2-Velocity` the dir is `unitree_go2_velocity` (snake-case lowercase).

### Latest training log
```bash
ls -td ~/projects/doga/logs/{baseline,trapezoidal,pyramid,footholds,lidar}_train_*/ | head -1
# then `tail -33 <that_dir>/train.log`
```

### TensorBoard
```bash
tensorboard --logdir ~/projects/doga/logs/rsl_rl
```

### Process status
```bash
pgrep -af "python.*train.py.*Unitree-Go2-" | grep -v pgrep
pstree -p -s <pid>     # top of tree should be systemd(1) for detached runs
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```

### Curriculum / reward trajectory from TensorBoard (Python)
```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
fs = sorted(glob.glob('<run_dir>/events.out.tfevents*'))
ea = EventAccumulator(fs[-1], size_guidance={'scalars': 0}); ea.Reload()
for tag in ['Curriculum/lin_vel_cmd_levels', 'Curriculum/terrain_levels', 'Episode_Reward/track_lin_vel_xy', 'Episode_Termination/bad_orientation']:
    ev = ea.Scalars(tag); n = len(ev)
    print(f'{tag}: first {ev[0].value:.3f} -> last {ev[-1].value:.3f}, n={n}')
```

---

## 8. How to run things

### Train (fresh)
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab
unset CUDA_HOME
CUDA_VISIBLE_DEVICES=1 python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
  --task Unitree-Go2-Velocity --headless --num_envs 4096 --seed 42
```

### Train (resume from checkpoint)
Add `--resume --load_run <run_ts_dirname> --checkpoint <model_X.pt>`.

### Train detached (survives logout) — use the launcher pattern
```bash
setsid nohup ~/projects/doga/launch-24h-pyramid.sh </dev/null >/dev/null 2>&1 &
disown
```
After ~30 s, verify:
```bash
PID=$(pgrep -f '^python.*train.py.*Unitree-Go2-Velocity ' | head -1)
pstree -p -s $PID | head -2     # must show systemd at top
```

### Stop training cleanly
```bash
PID=$(pgrep -f '^python.*train.py.*Unitree-Go2-Velocity ' | head -1)
TPID=$(pgrep -f 'timeout.*train.py.*Unitree-Go2-Velocity ' | head -1)
kill -TERM $TPID
sleep 8
if kill -0 $PID 2>/dev/null; then kill -KILL $PID; fi
```
Final checkpoint will be saved BEFORE termination if you let timeout fire (24h budget). The "Mutex Recursion not allowed" error during shutdown is cosmetic.

### Play (no video, just simulate)
```bash
CUDA_VISIBLE_DEVICES=1 python ~/unitree_rl_lab/scripts/rsl_rl/play.py \
  --task Unitree-Go2-Velocity --num_envs 8 --headless \
  --checkpoint <abs/path/to/model_X.pt>
```
This will also export `policy.pt` (TorchScript) and `policy.onnx` to `<ckpt_dir>/exported/` — useful for Jetson deployment.

### Record video
Add `--video --video_length 3000` (= 60 s at 50 Hz). Output goes to
`<ckpt_dir>/videos/play/rl-video-step-0.mp4` (1280×720 50 fps).
Side-on viewer is configured in `RobotPlayEnvCfg.__post_init__`.

For the foothold task, markers will render automatically (yellow=unclaimed, green=claimed).

### IMPORTANT: do NOT play.py while training is running on the same GPU
PhysX cooking conflicts. Either kill training first, or pin play to GPU 0
(if free). Last attempt to share GPU with the foothold training caused a
PhysX `getArticulationData` failure.

---

## 9. Custom terrain quick reference

| Terrain | Module | Used by | Key parameters |
|---|---|---|---|
| `COBBLESTONE_ROAD_CFG` (orig) | `velocity_env_cfg.py` | Stage 1 baseline (flat-only, deprecated for our use) | only "flat" subterrain enabled |
| `ROUGH_TERRAINS_CFG` | Isaac Lab `isaaclab.terrains.config.rough` | Stage 2 baseline + foothold-rough variants | stairs/slopes/boxes/random_rough mix |
| `TRAPEZOIDAL_BUMPS_CFG` | `~/unitree_rl_lab/.../terrains/trapezoidal_bumps.py` | Trapezoidal run (model_40300) | flat-low (30cm) + 45° up-ramp + flat-top (30cm) + 45° down-ramp; bumps 10–20cm |
| `PYRAMID_RIDGED_STAIRS_CFG` | `~/unitree_rl_lab/.../terrains/pyramid_ridged_stairs.py` | **Current** pyramid run | up 45° + small down 45° + flat tread, 5x extended both axes (20m × 40m tile), 20 steps/side, 0.20m treads, 0.05m ridges, μ=0.75, friction multiply combine |

To swap terrains: edit `velocity_env_cfg.py`'s `RobotSceneCfg.terrain.terrain_generator = X_CFG`. **Do not touch the curriculum config when swapping** — `terrain_levels_vel` and `lin_vel_cmd_levels` work with any subterrain that respects the difficulty parameter.

---

## 10. Known issues / caveats

### Critical
- **`rsl_rl` checkpoint resume does NOT persist curriculum state.** `lin_vel_cmd_levels` resets to 0.10 on resume; re-advances within ~2 k iters. `terrain_levels` resets to `max_init_terrain_level=1`.
- **Sim2real gap on lidar**: §15b of `HIGH-LEVEL-RESEARCH.md`. Foothold staleness, sensor latency, IMU drift not modelled.
- **The pyramid current run is in a degenerate local minimum** (shuffle gait, feet_clearance 0.03, terrain_levels 0.02). See §11 for how to fix.

### Minor
- `attach_yaw_only` is deprecated in IsaacLab 2.3.2 — silently overrides to `ray_alignment="base"` (cosmetic warning only).
- "Mutex Recursion not allowed" error on SIGTERM shutdown is cosmetic.
- play.py's `pretrained_checkpoint` import was patched (try/except) since the module doesn't exist in 2.3.2.
- `CUDA_HOME=/usr` in user shell — always `unset CUDA_HOME` before pip-built extensions or training.
- Setuptools 79.0.1 pinned (80+ removed `pkg_resources` which `flatdict` imports).

---

## 11. Where things stand AT HANDOFF TIME and recommended next steps

### Current state (run #10, pyramid-ridged-stairs)

- Process: PID **1263174**, `~12 h` elapsed of 24 h budget at handoff time. **STILL RUNNING.**
- Latest ckpt: `model_46500+` under `~/projects/doga/logs/rsl_rl/unitree_go2_velocity/2026-05-09_01-07-01/`
- Log: `~/projects/doga/logs/pyramid_train_20260509_010656/train.log`
- **Status: Degenerate.** Policy converged within ~1.3 k iters of resume to a "shuffle and track velocity" policy:
  - `feet_clearance` reward = 0.03 (was 0.88 on trapezoidal — feet barely leave ground)
  - `terrain_levels` curriculum collapsed from 1.49 → 0.02 and stuck (robots get demoted to easiest possible terrain because shuffling can't traverse harder rows)
  - `track_lin_vel_xy` = 1.35 (still near-perfect; that's the local minimum the policy fell into)
  - `bad_orientation` = 1.0% (fine)
  - `feet_slide` = −0.075 (consistent — robot IS sliding constantly, but consequences are too small to matter)

The user has approved killing this run and trying option 1 of the next-step
plan (explicit slope-contact penalty). **Do that next.**

### Why this is happening (ultrathink)

Three rewards interact pathologically:
- `track_lin_vel_xy` (+1.5 weight) rewards velocity tracking — easy to satisfy by sliding
- `feet_slide` (−0.1 weight) penalises foot motion during contact — small enough to ignore
- `feet_clearance` (+1.0 weight) rewards swing-foot height — but only fires when swing happens; the policy decided "don't swing = no slip = no need for clearance"

The terrain physics alone (μ=0.75 on 45° slopes) is a real signal but its
gradient (via `feet_slide` penalty) is too small relative to the dominant
velocity-tracking reward. The policy found a corner of action space where
sliding-while-tracking is the locally optimal strategy and the gradient
toward "lift feet" is masked by the immediate cost of joint torque/velocity
penalties.

**The terrain setup is fine; the reward shaping is the bottleneck.**

### Recommended next steps (in priority order)

#### Step 1 — kill current run, document final state

```bash
PID=$(pgrep -f '^python.*train.py.*Unitree-Go2-Velocity ' | head -1)
TPID=$(pgrep -f 'timeout.*train.py.*Unitree-Go2-Velocity ' | head -1)
kill -TERM $TPID; sleep 8
if kill -0 $PID 2>/dev/null; then kill -KILL $PID; fi
```

Note: most-recent checkpoint dir is `unitree_go2_velocity/2026-05-09_01-07-01/` — DO NOT delete; we may use early ckpts if needed.

#### Step 2 — implement explicit slope-contact penalty (the user-approved fix)

The terrain physics signal is too weak; we need an explicit reward signal.
Implementation outline (target ~50 LOC in `mdp/rewards.py`):

```python
def feet_on_slope_penalty(
    env, sensor_cfg, asset_cfg,
    contact_threshold=5.0,
    slope_threshold_rad=0.30,   # ~17° — anything steeper than this is "slope"
    terrain_query_radius=0.025,  # m, footprint radius for gradient query
):
    """Penalty when a foot is in contact at a position where the local terrain
    gradient exceeds slope_threshold_rad.

    Privileged-at-train (uses terrain heightfield ground truth) — does NOT
    appear in policy obs, so it satisfies the user's real-Go2-sensors rule.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    # foot world xyz
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # foot in contact?
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = forces[..., 2] > contact_threshold
    # query the terrain heightfield for local gradient
    # — env.scene.terrain.terrain_origins gives per-env spawn origin
    # — env.scene.terrain.height_field gives the heightfield raster
    # Sample slope = max gradient magnitude in a 3x3 cell window around foot xy
    # (You'll need to write a helper that does this on GPU; the heightfield is
    # accessible via env.scene.terrain.terrain_generator's pre-baked array;
    # see ~/IsaacLab/source/isaaclab/isaaclab/terrains/terrain_importer.py for
    # how to read it.)
    slope_mag = ...  # gradient magnitude per foot, shape (E, F)
    # mask: in contact AND on slope
    bad = in_contact & (slope_mag > slope_threshold_rad)
    return -bad.float().sum(dim=-1)
```

**Add to RewardsCfg with weight `-1.0` initially** (tunable). Once added:

- Change `velocity_env_cfg.py` to register the new reward:
  ```python
  feet_on_slope = RewTerm(
      func=mdp.feet_on_slope_penalty,
      weight=-1.0,
      params={
          "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
          "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
          "slope_threshold_rad": 0.30,
      },
  )
  ```
- The penalty is "privileged-at-train" — does not change the policy obs space
  so it does NOT break warm-start from `model_40300.pt`.

#### Step 3 — rebalance reward weights

After adding the slope penalty, if shuffling persists, tune the balance:
- Bump `feet_clearance` from 1.0 → 5.0 (force lifting)
- Or reduce `track_lin_vel_xy` from 1.5 → 1.0
- Or add a `forward_progress` reward computing actual displacement per step

The user already noted these as candidate options (4, 2, 1 in our earlier
exchange).

#### Step 4 — relaunch detached, warm-start from `model_40300.pt` (NOT 46500+)

The current run's policy is in a bad local minimum; warm-starting from
`model_46500` would inherit the shuffle behaviour. **Restart from
`model_40300.pt`** (the trapezoidal-trained walker — the last known
non-degenerate checkpoint).

```bash
# update launch-24h-pyramid.sh's --load_run and --checkpoint to:
#   --load_run "2026-05-07_23-28-00"
#   --checkpoint "model_40300.pt"
# then:
setsid nohup ~/projects/doga/launch-24h-pyramid.sh </dev/null >/dev/null 2>&1 &
disown
```

#### Step 5 — verify & iterate

After ~1 h of training, check:
```bash
LATEST_RUN=$(ls -td ~/projects/doga/logs/rsl_rl/unitree_go2_velocity/*/ | head -1)
# inspect feet_clearance and Curriculum/terrain_levels via TB Python snippet (§7)
```

If `feet_clearance > 0.5` AND `terrain_levels > 0.3` after 5 k iters of new
training, reward shaping is working. Otherwise, escalate to step 3 or
re-think.

### Alternatives if step 2 doesn't work

| Approach | Effort | Why it might work |
|---|---|---|
| **Add forward-progress reward** | Trivial. Compute `(root_pos_w - last_root_pos_w).x` per step. | Forces actual movement, not just velocity tracking. Defeats sliding. |
| **Heightfield-gradient observation in critic only** | ~50 LOC. | Privileged-at-train; lets critic distinguish good/bad foot placements. Could improve training even without slope penalty. |
| **Reduce action_rate penalty** | 1 line. | -0.1 → -0.01 might unlock faster gait |
| **Increase episode length on harder terrain** | Curriculum hook | More time to traverse 4 m → easier to advance terrain_levels |

### Things explicitly NOT to do (per user constraints)

- Add `height_scanner` to policy obs — forbidden (sim-only, not on real Go2).
- Add foot contact sensors to policy obs — forbidden (Go2 has no foot contact sensors).
- Add scandots / privileged terrain to policy obs — forbidden.
- Skip the explicit slope penalty in favor of "more compute" on the same setup — already plateaued.

---

## 12. Backups

`~/projects/doga/checkpoints_backup/` contains an `rsync` snapshot of all
`logs/rsl_rl/` runs taken on 2026-05-04. Run 10 (current pyramid) is NOT yet
in there. To refresh:

```bash
rsync -a ~/projects/doga/logs/rsl_rl/ ~/projects/doga/checkpoints_backup/
```

---

## 13. The deploy story (for context — NOT next step)

The Jetson deployment guide is in `JETSON-DEPLOY.md`. Key points:
- `unitree_rl_lab/deploy/robots/go2/main.cpp` is the C++ deployment skeleton.
- Bundled ONNX Runtime is x86 — must swap for aarch64 build before Jetson cross.
- Joint SDK ordering is the #1 cause of Day-1 failures. Verify against `unitree.py:joint_sdk_names`.
- `play.py` exports `policy.pt` and `policy.onnx` to `<ckpt_dir>/exported/` after running once.
- Foothold task requires an upstream foothold planner on the Jetson — not solved.
- Latency / staleness domain randomisation NOT in training; add before any sim2real.

For just locomotion (no foothold): `model_40300.pt` (trapezoidal) or earlier
`model_25600.pt` (rough-terrain) is deploy-ready as-is. Pyramid-trained
checkpoints should NOT deploy until the shuffle pathology is fixed.

---

## 14. Conversation memory

The user prefers brief responses (small terminal pane). Only write long
content into files (like this doc). End-of-turn summaries should be 1-2
sentences. Never mention the gentle reminders about TaskCreate.

The user has given direct instructions about:
- Ultrathink for hard reasoning turns (when keyword "ultrathink" appears)
- Backup before destructive operations
- Don't make assumptions; cite primary sources
- They set the rules; do not propose forbidden things
- When given a directive, execute and document — don't editorialise

If a previous exchange referenced an earlier doc (like `HIGH-LEVEL-RESEARCH.md` §14 or `JETSON-DEPLOY.md` §6), trust it; the user has already read and approved those.

---

## 15. Stage-3 surface task — gym id `Unitree-Go2-Surface`

Successor to the virtual-foothold task (`Unitree-Go2-Footholds`). The old task remains registered and runnable — its file, its MDP helpers (`foothold_state.update_claims`, `mdp.foothold_landing`, etc.) and the v2 checkpoint at `model_62600.pt` are all preserved untouched. The surface task is built additively on top.

**Files added / modified (additive only):**

- New: `mdp/surface_labels.py`, `mdp/surface_rewards.py`, `mdp/surface_stats.py`, `mdp/tests/test_surface_labels.py`
- New: `robots/go2/velocity_surface_env_cfg.py`
- New: `terrains/trapezoidal_bumps.py:TRAPEZOIDAL_BUMPS_SURFACE_CFG` (labels-emitting variant; `TRAPEZOIDAL_BUMPS_CFG` unchanged)
- New: `launch-24h-surface-3a.sh` (30 h, GPU 1, `--task Unitree-Go2-Surface`, no resume)
- Edited (additive): `mdp/foothold_state.py` (appended `update_landing_events` at bottom with its own `_landing_*` state), `mdp/__init__.py`, `terrains/__init__.py`, `robots/go2/__init__.py`
- Edited (two-line bug fix): `robots/go2/velocity_env_cfg.py` — registered `ang_vel_cmd_levels` in `CurriculumCfg`, changed `CommandsCfg.base_velocity.ranges.ang_vel_z` to `(0.0, 0.0)`. Affects the baseline task too; intentional per FOOTHOLDS-YAW-ANALYSIS §6.1.

### 15.1 Surface task metric decoding

`mdp/surface_stats.py` emits five TensorBoard metrics through the standard `Episode_Reward/<term>` path:

```
Episode_Reward/landing_fraction_flat
Episode_Reward/landing_fraction_lipped
Episode_Reward/landing_fraction_unsafe
Episode_Reward/landing_fraction_out_of_bounds
Episode_Reward/landings_per_episode
```

These are zero-weight-in-spirit but use `weight=1e-6`, not `0.0`. Isaac Lab's `RewardManager.compute()` (reward_manager.py:144-148) short-circuits zero-weight terms — the func is never called and the per-episode sum stays at zero. To keep the standard logging path, the weight is set to `1e-6` and reward contribution is eight orders of magnitude below the smallest training reward (`track_lin_vel_xy_exp ≈ 25 / ep`).

**Decoding the TB values:** Isaac Lab logs `_episode_sums[term] / max_episode_length_s`, where `_episode_sums += value × weight × dt`. With weight=1e-6, dt=0.02 s, max_episode_length_s=20 s:

```
Episode_Reward/<term> = count × 1e-6 × 0.02 / 20 = count × 1e-9
```

So `count_per_episode = Episode_Reward/<term> × 1e9` (the `DECODE_FACTOR_LANDINGS` constant in `surface_stats.py`). Don't panic when the TB plot shows numbers around 1e-7 — that's 60–120 landings/episode after decoding.

The "fraction" naming is historical; the values are class-conditional COUNTS. Divide by `Episode_Reward/landings_per_episode` to get the actual class fraction (decode constants cancel).

### 15.2 Surface task weight calibration

The original prompt's prescribed Stage-3 reward weights were `r_flat=+2.0`, `p_lipped=-0.5`, `p_pocket=-0.5`, `p_unsafe=-2.0`. Per-episode arithmetic with ~120 landings/episode and ~50 % FLAT fraction at random policy:

```
+2.0 × 60 = 120 weighted/ep   vs   track_lin_vel_xy_exp ≈ 25 weighted/ep
                           ratio = 472 %
```

That would have dominated velocity tracking — same failure mode as the deleted `forward_progress` term (`velocity_env_cfg.py:358-365`) that produced 100 % `bad_orientation` at Stage-2.5. Scaled 50× down to land in the 5–15 % target band:

```
Stage 3a: r_flat=+0.04                                                     ≈ 9 % ratio
Stage 3b: r_flat=+0.04, p_unsafe=-0.04                                     bounded ±9 %
Stage 3c: r_flat=+0.04, p_lipped=-0.01, p_pocket=-0.01, p_unsafe=-0.04     bounded ±9 %
```

If Stage 3a shows `landing_fraction_flat` does not rise above the ~0.5 random baseline, step up to `r_flat=0.06` (15 % band) before deciding the design itself doesn't work.

### 15.3 Surface labelling: corner-smoothing note

The detector uses a 3×3 Sobel gradient (`compute_labels` in `mdp/surface_labels.py`). At a 45° ramp-to-plateau corner, the central-difference slope reading at the FIRST plateau pixel adjacent to the ramp is ≈26.6° — above the 15° UNSAFE threshold. So the **UNSAFE class extends 1 pixel into the plateau on each side** of every ramp, eating into the FLAT strip.

Net effect on trapezoidal bumps at h_scale=0.05 m (per-tile interior):

| Plateau | Geometry | Pure FLAT pixels | Pure FLAT width |
|---|---|---|---|
| Low (between bumps) | 6 px (30 cm) | 5 px (pixels 0..4) | 25 cm |
| Top (on bump) | 6 px (30 cm) | 3 px (pixels 12..14) | **15 cm** |

The top plateau loses 1 pixel of FLAT on each side because both neighbours are ramps; the low plateau loses only one side per period because the next-low-plateau geometry is symmetric end-to-end (the central-diff at the boundary reads 0 since both neighbours are at z=0). This is half the 20 cm of FLAT that Step 2 §2 of the design predicted.

If Stage 3a data shows `landing_fraction_flat` plateaus low because the policy cannot find a wide enough FLAT target on the top plateau, the calibration to retune is either:

1. Raise `SLOPE_UNSAFE_DEG` from 15° to ~35° (only true mid-ramp pixels at 45° qualify as UNSAFE). This restores ~25 cm FLAT on the top plateau.
2. Or widen `LIPPED_DILATION_PIXELS` from 1 to 2 (catches the corner-smoothed pixel as LIPPED instead of UNSAFE, which removes its penalty in Stage 3b).

Decision waits for Stage 3a data per the no-pre-emptive-tuning rule.

**Update 2026-05-31** — Stage 3a (warm-start) and Stage 3b (warm-start) both plateaued at `flat_fraction ≈ 0.34–0.36` after 17k iters, well below the ~0.50 random baseline at the threshold-15° labelling. The trajectory of Stage 3b showed early shaping (unsafe 0.514 → 0.422 in 4k iters) then a plateau — diagnostic of the labelling-as-bottleneck hypothesis (the gait *can* avoid SOME ramps, but can't consistently hit the narrow 15 cm FLAT band). Escalating to option **(1)** above: `SLOPE_UNSAFE_DEG = 35.0`. Note that widening FLAT raises the random-policy baseline from ~0.35 to ~0.55–0.60 at difficulty 0.7, so the new success criterion is `flat_fraction > 0.60 with rising trajectory`, not > 0.50. Launcher: `launch-24h-surface-3b-flat35-warmstart.sh`.

**Update 2026-06-02 — widened-FLAT 3b verdict: NEUTRAL.** After 16,898 iters (run `2026-05-31_18-13-53`), `flat_fraction = 0.562` (vs predicted random baseline ~0.55–0.60), trajectory flat-line from step 733 onward. Locomotion fully preserved (terrain 0.70, err_vel_xy 0.35, bad_orient 16.4%). Combined with the 3a vs 3b result (50× reward swing → ~0.02 fraction shift), this confirms the broader conclusion: **per-footfall surface reward of this form at these magnitudes is not enough to steer Go2 foot placement materially above area-proportional random on this terrain/gait**. Gait dynamics dominate footfall placement. To get a positive result the next experiment needs a different mechanism (much larger reward magnitude, a foothold-planner head, or reactive control gating). The Stage 3 surface reward as currently designed is documented as a null finding, not a positive demonstration of surface preference.

### 15.6 Shutdown signal-handler bug — fixed at the launcher level (2026-06-02)

Two of the three completed surface runs (`2026-05-21_02-51-23` from-scratch 3a, `2026-05-31_18-13-53` warm-start 3b-flat35) failed to terminate cleanly at the 30 h `timeout`. The from-scratch 3a hit the `carb tasking Mutex "Recursion not allowed"` abort during shutdown; the 3b-flat35 entered a zombie state (process alive, GPU at 1 % util, iterations ticking at 25% normal speed) and had to be killed manually 14 hours after the budget expired.

**Root cause:** Isaac Lab installs a custom SIGTERM handler in `~/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py:986` that only calls `self._app.close()` and **returns** — no `os._exit()`. After the handler returns, Python resumes the rsl-rl learning loop with a half-closed SimulationApp, behavior undefined (race between in-flight kit threads and the close call). Sometimes it crashes (3a), sometimes it limps on (3b-flat35).

**Fix (launcher-side, not upstream):** add `--kill-after=120` to `timeout` on all surface launchers. After SIGTERM, `timeout` waits 120 s; if the child is still alive it sends SIGKILL which cannot be caught. Pattern in all four surface launchers as of 2026-06-02:

```
exec timeout --preserve-status --signal=TERM --kill-after=120 108000 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py ...
```

The older launchers (footholds, baseline, pyramid, trapezoidal) were left unchanged — they have completed cleanly historically and the change is only applied where the failure was observed. If a future run on any of those wedges similarly, add `--kill-after=120` there too.

The upstream Isaac Lab handler should also call `os._exit(143)` after `_app.close()` for true graceful shutdown, but that's not our patch to make.

### 15.4 Active stage selection

`mdp/surface_rewards.py` defines `SurfaceStage` (IntEnum: 3A/3B/3C), `STAGE_WEIGHTS` (dict of per-stage weights), and `ACTIVE_STAGE` (default: `STAGE_3A`). Switching stage is a one-line edit; the env config reads `STAGE_WEIGHTS[ACTIVE_STAGE]` at construction time and forwards those into the `surface_aware_landing` RewTerm's params.

### 15.5 Implicit signals already present in Stage-2 rewards

Stage 3a is `r_flat=+0.04, p_unsafe=0`, advertised as "positive-only". It is NOT a pure positive baseline — the inherited Stage-2 reward `feet_on_slope_penalty (weight=-0.5, cos_threshold=0.95 ≈ 18°)` already penalises foot-ground contact on tilted surfaces during stance (i.e., it acts as a coarse UNSAFE-deterrent every step the foot is on a ramp). The new `surface_aware_landing` reward adds a sharper, event-based, class-resolved version. Don't interpret Stage 3a as "the policy learns surface preference purely from r_flat alone."
