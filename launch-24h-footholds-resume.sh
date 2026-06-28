#!/usr/bin/env bash
# Resume foothold training from model_62600.pt with the bug-fixed v2 cfg.
# 24h budget, GPU 1, detached.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/footholds_resume_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h foothold resume at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Footholds"
  echo "  resume from: 2026-05-05_21-37-06/model_62600.pt"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$LOG_DIR/meta.txt"
exec timeout --preserve-status --signal=TERM 86400 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-Footholds \
    --headless \
    --num_envs 4096 \
    --max_iterations 200000 \
    --seed 42 \
    --resume \
    --load_run "2026-05-05_21-37-06" \
    --checkpoint "model_62600.pt" \
  >"$LOG_DIR/train.log" 2>&1
