#!/usr/bin/env bash
# Launches a 12-hour training run of Unitree-Go2-Velocity-Lidar pinned to GPU 1.
# rsl_rl saves checkpoints periodically, so the timeout-kill at 12h leaves a
# trained policy at the last checkpoint.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/lidar_train_${RUN_TS}"
mkdir -p "$LOG_DIR"
echo "starting 12h run at $(date)" | tee "$LOG_DIR/meta.txt"
echo "  GPU: 1" | tee -a "$LOG_DIR/meta.txt"
echo "  num_envs: 2048" | tee -a "$LOG_DIR/meta.txt"
echo "  task: Unitree-Go2-Velocity-Lidar" | tee -a "$LOG_DIR/meta.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG_DIR/meta.txt"
# 12h = 43200s; leave 60s margin so timeout sends SIGTERM cleanly first
exec timeout --preserve-status --signal=TERM 43200 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-Velocity-Lidar \
    --headless \
    --num_envs 2048 \
    --max_iterations 100000 \
    --seed 42 \
  >"$LOG_DIR/train.log" 2>&1
