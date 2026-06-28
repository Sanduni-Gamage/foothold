#!/usr/bin/env bash
# Stage-3-surface (Stage 3a) training, WARM-STARTED from the Stage-2 rough
# walker (model_25600.pt) via the partial-transfer checkpoint built by
# make_surface_warmstart.py.
#
# Why warm-start: the from-scratch Stage 3a run (2026-05-21_02-51-23) hit three
# kill criteria — terrain_levels collapsed to 0.01, lin-vel reward below the
# 0.7x floor, flat-landing fraction below random. Root cause was the policy
# couldn't bootstrap velocity tracking on the bumps, so the terrain curriculum
# demoted it to difficulty ~0 and never recovered. Warm-starting from a
# rough-competent walker keeps velocity tracking alive so the surface reward
# can actually shape footfall placement.
#
# The warm-start checkpoint zero-inits the new obs columns (height_scan 187 +
# foot_positions 12), so the policy is function-identical to the rough walker
# at iter 0 and learns to use perception from there. Optimizer state cleared,
# iter reset to 0 (true warm-start, not a resume of the rough run).
#
# Prereq (run once, already done):
#   python ~/projects/dogb/make_surface_warmstart.py
# which writes:
#   logs/rsl_rl/unitree_go2_surface/warmstart_rough_25600/model_25600_surface.pt
#
# GPU 1, 30h budget. Active reward stage = STAGE_3A (surface_rewards.py).
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
# IMPORTANT: cwd must be dogb so rsl_rl resolves logs/rsl_rl/... under dogb.
cd /home/anyone/projects/dogb
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/dogb/logs/surface_warmstart_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 30h surface (stage 3a) WARM-START training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Surface"
  echo "  seed: 42"
  echo "  warm-start: warmstart_rough_25600/model_25600_surface.pt (from rough model_25600)"
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
    --resume \
    --load_run warmstart_rough_25600 \
    --checkpoint model_25600_surface.pt \
  >"$LOG_DIR/train.log" 2>&1
