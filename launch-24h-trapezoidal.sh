#!/usr/bin/env bash
# 24h training on the trapezoidal-bumps terrain (45-deg slopes, sharp peaks).
# Warm-start from rough-terrain stage-2 walker (model_25600.pt). GPU 1, detached.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/trapezoidal_train_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h trapezoidal-bumps training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Velocity (terrain swapped to TRAPEZOIDAL_BUMPS_CFG)"
  echo "  warm-start: 2026-05-03_18-37-15/model_25600.pt (rough-terrain stage-2)"
  echo "  bumps: 45-deg slopes, height 5-15cm, flat low/top 30cm each"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$LOG_DIR/meta.txt"
exec timeout --preserve-status --signal=TERM 86400 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-Velocity \
    --headless \
    --num_envs 4096 \
    --max_iterations 200000 \
    --seed 42 \
    --resume \
    --load_run "2026-05-03_18-37-15" \
    --checkpoint "model_25600.pt" \
  >"$LOG_DIR/train.log" 2>&1
