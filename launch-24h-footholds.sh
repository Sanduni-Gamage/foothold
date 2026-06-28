#!/usr/bin/env bash
# Stage-3 fresh training on Unitree-Go2-Footholds (permutation-invariant
# foothold-reaching). 24h budget, GPU 1, detached.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/footholds_train_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h foothold training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Footholds"
  echo "  seed: 42"
  echo "  fresh init (no warm-start; obs shape change incompatible with --resume)"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$LOG_DIR/meta.txt"
exec timeout --preserve-status --signal=TERM 86400 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-Footholds \
    --headless \
    --num_envs 4096 \
    --max_iterations 200000 \
    --seed 42 \
  >"$LOG_DIR/train.log" 2>&1
