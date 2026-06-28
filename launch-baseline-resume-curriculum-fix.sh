#!/usr/bin/env bash
# Resume Unitree-Go2-Velocity from model_2600.pt with the curriculum bug fix
# applied (lin_vel_cmd_levels and ang_vel_cmd_levels now actually advance).
# 24 h budget, GPU 1, detached.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/doga/logs/baseline_currfix_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h resume run with curriculum fix at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Velocity"
  echo "  resume from: 2026-05-03_14-58-34/model_13400.pt"
  echo "  stage 2: terrain swapped to ROUGH_TERRAINS_CFG (was COBBLESTONE flat-only)"
  echo "  stage 2: foot_clearance_reward added (target_height=0.08m, weight=1.0)"
  echo "  curriculum fix: lin_vel/ang_vel _cmd_levels global-step gate replaced with per-env rate limiter"
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
    --load_run "2026-05-03_14-58-34" \
    --checkpoint "model_13400.pt" \
  >"$LOG_DIR/train.log" 2>&1
