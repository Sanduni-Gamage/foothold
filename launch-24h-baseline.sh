#!/usr/bin/env bash
# Standard unitree_rl_lab Unitree-Go2-Velocity training (proprio-only, no lidar).
# 24h budget, pinned to GPU 1, detached via setsid+nohup. rsl_rl saves
# checkpoints every 100 iters; SIGTERM at 24h leaves the last one intact.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/baseline_train_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h baseline run at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Velocity"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$LOG_DIR/meta.txt"
exec timeout --preserve-status --signal=TERM 86400 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-Velocity \
    --headless \
    --num_envs 4096 \
    --max_iterations 200000 \
    --seed 42 \
  >"$LOG_DIR/train.log" 2>&1
