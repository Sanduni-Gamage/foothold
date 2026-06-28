#!/usr/bin/env bash
# Resume Unitree-Go2-Velocity-Lidar training from model_7300.pt for 24 hours,
# pinned to GPU 1. rsl_rl saves checkpoints every 100 iters; SIGTERM at 24 h
# leaves the last checkpoint intact.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/lidar_train_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h resume run at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 2048"
  echo "  task: Unitree-Go2-Velocity-Lidar"
  echo "  resume from: 2026-05-02_00-39-44/model_7300.pt"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$LOG_DIR/meta.txt"
# 24h = 86400s
exec timeout --preserve-status --signal=TERM 86400 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-Velocity-Lidar \
    --headless \
    --num_envs 2048 \
    --max_iterations 200000 \
    --seed 42 \
    --resume \
    --load_run "2026-05-02_00-39-44" \
    --checkpoint "model_7300.pt" \
  >"$LOG_DIR/train.log" 2>&1
