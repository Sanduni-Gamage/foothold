# Run record — `Unitree-Go2-Velocity-Lidar` (proprio + L1)

## Configuration

- Task: `Unitree-Go2-Velocity-Lidar`
- Code: `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_lidar_env_cfg.py`
- Sensors fed to policy:
  - proprioception: `base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos_rel`, `joint_vel_rel`, `last_action`
  - **`lidar`** (2304-dim log-distance, σ=2 cm Gaussian noise, 10% per-ray Bernoulli dropout, clip 0.05–30 m, log1p compressed)
- L1 raycaster: `LidarPatternCfg(channels=32, vfov=(0,90), hfov=(-180,180), hres=5°)`, mounted upright at `(0.15, 0, 0.12)` on base, `update_period = 1/11 s`
- Terrain: rough mix (random_rough / slopes / boxes / pyramid_stairs) at curriculum level 0
- Network: actor/critic MLP `[2364 → 512 → 256 → 128 → 12/1]`, ELU, asymmetric (critic gets clean lidar)
- Algo: rsl_rl PPO, `desired_kl=0.01`, initial LR `1e-3` adaptive
- Hardware: GPU 1 (RTX 5060 Ti 16 GB), `num_envs=2048`, seed 42

## Timeline

| Phase | Duration | Iter range | Notes |
|---|---|---|---|
| Phase 1 (initial) | ~12 h budgeted, killed/restarted twice in first hour | 0 → 7300 | log dirs `…/2026-05-02_00-33-21/`, `…/00-37-15/`, `…/00-39-44/`. Run dir `…/00-39-44/` was the long uninterrupted phase-1 stretch (~8.5 h). |
| Pause | — | — | Stopped at iter 7300 to record flat play video |
| Phase 2 (resume) | 24 h budgeted, killed at 6:39 h | 7300 → 12300 | log dir `lidar_train_20260502_100014/`, run dir `…/2026-05-02_10-00-20/`, resumed from `model_7300.pt` |

## Final metrics at kill (iter 12 300, 6 h 39 m into phase-2 resume)

| Metric | Value |
|---|---|
| Mean reward | +3.59 |
| Mean episode length | 506 / 1000 |
| `bad_orientation` termination | 79.2 % |
| `time_out` termination | 20.8 % |
| `track_lin_vel_xy` reward | +0.57 |
| `track_ang_vel_z` reward | +0.21 |
| `Curriculum/terrain_levels` | 0.0 (never advanced) |
| `Curriculum/lin_vel_cmd_levels` | 0.5 |
| Iteration time | 4.4 s |
| Adaptive LR | 1e-5 (pinned at floor for entire run) |
| Mean policy noise std | ~0.30 |
| GPU 1 VRAM | ~10 GB / 16 GB |

## Final checkpoint

```
/home/anyone/projects/doga/logs/rsl_rl/unitree_go2_velocity_lidar/2026-05-02_10-00-20/model_12300.pt
```

Earlier phase-1 checkpoints (every 100 iters) under
`/home/anyone/projects/doga/logs/rsl_rl/unitree_go2_velocity_lidar/2026-05-02_00-39-44/`.

## Artefacts

- Train logs: `~/projects/doga/logs/lidar_train_*/train.log`
- Play video (60 s, flat terrain, iter 7300 checkpoint):
  `…/2026-05-02_00-39-44/videos/play/rl-video-step-0.mp4`
- TensorBoard: `~/projects/doga/logs/rsl_rl/unitree_go2_velocity_lidar/*/events.out.tfevents.*`

## Observations (factual, no interpretation)

- LR pinned at `1e-5` (rsl_rl floor) from iter ~7300 onward — verified from TensorBoard `Loss/learning_rate`.
- Terrain curriculum (`terrain_levels`) never advanced past 0.0 across the entire run.
- `bad_orientation` termination rate stayed in the 75–90 % band; one snapshot at iter 8160 read 0 % momentarily but reverted.
- `track_lin_vel_xy` reward grew from +0.14 (iter ~30) → +0.57 (iter 12 300).
- Velocity-command curriculum advanced 0.10 → 0.50.
- A 60 s flat-terrain play recording at iter 7300 shows the robot taking a few steps and falling.

## Why stopped

Reverted to the standard `Unitree-Go2-Velocity` (proprio-only) recipe per user instruction.
