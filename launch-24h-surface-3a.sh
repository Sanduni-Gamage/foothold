#!/usr/bin/env bash
# Stage-3-surface (Stage 3a) fresh training: surface-class-aware landing
# reward on trapezoidal-bumps heightfield, with downward height-scanner
# perception. GPU 1, 30h budget (108000s) to absorb height-scanner step-time
# overhead vs the proprio Stage-2 baseline.
#
# Fresh init, no warm-start: obs shape change (added height_scan 187 dims +
# foot_positions 12, removed nearest_footholds 8 -> total 244-dim policy obs)
# means no v2 checkpoint loads cleanly. Confirmed acceptable.
#
# Active reward stage lives in
#   ~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/surface_rewards.py
# (ACTIVE_STAGE = SurfaceStage.STAGE_3A by default). One-line edit to advance.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/dogb/logs/surface_train_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 30h surface (stage 3a) training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Surface"
  echo "  seed: 42"
  echo "  fresh init (no warm-start; obs shape change incompatible with --resume)"
  echo "  obs dim: 244 (policy), 259 (critic); height scan 187 rays; surface stage 3a"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$LOG_DIR/meta.txt"
exec timeout --preserve-status --signal=TERM --kill-after=120 108000 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-Surface \
    --headless \
    --num_envs 4096 \
    --max_iterations 200000 \
    --seed 42 \
  >"$LOG_DIR/train.log" 2>&1
