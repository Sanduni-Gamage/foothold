#!/usr/bin/env bash
# 24h training on the pyramid-ridged-stairs terrain (45-deg up-ramp + small
# 45-deg down-ramp + flat tread, 5x extended in both X and Y, μ=0.75).
# Warm-start from the trapezoidal-bumps walker (model_40300.pt). GPU 1, detached.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/pyramid_train_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h pyramid-ridged-stairs training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Velocity (terrain = PYRAMID_RIDGED_STAIRS_CFG)"
  echo "  warm-start: 2026-05-07_23-28-00/model_40300.pt (trapezoidal walker)"
  echo "  geometry: 20 steps/side, μ=0.75, tread=0.20m, ridge=0.05m, tile 20x40m"
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
    --load_run "2026-05-07_23-28-00" \
    --checkpoint "model_40300.pt" \
  >"$LOG_DIR/train.log" 2>&1
