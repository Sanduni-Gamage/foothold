#!/usr/bin/env bash
# Stage-3-surface (Stage 3B) re-run with WIDENED FLAT class.
# Warm-started from the same Stage-2 rough walker checkpoint.
#
# Differs from launch-24h-surface-3b-warmstart.sh ONLY in:
#   - mdp/surface_labels.py: SLOPE_UNSAFE_DEG = 15.0 -> 35.0
#   - log dir prefix surface_warmstart_3b_flat35_*
# Same warm-start checkpoint, same ACTIVE_STAGE=STAGE_3B, same task,
# num_envs, seed, timeout. Clean ablation: 3b vs 3b-flat35 isolates the
# effect of the labelling threshold.
#
# Why this run: the original 3a/3b runs plateaued at flat_fraction ~0.34-0.36
# across 17k iters. Diagnosis (HANDOFF §15.3): with the 15 deg threshold,
# 3x3 Sobel corner-smoothing classifies the first plateau pixel adjacent to
# each ramp as UNSAFE, leaving only 15 cm of pure FLAT on a 30 cm top
# plateau. With Go2 foot pads ~4-5 cm and realistic landing noise of similar
# magnitude, that target is too narrow to hit consistently — the gait can't
# do it regardless of reward weight. Threshold 35 deg restores 25-30 cm of
# pure FLAT.
#
# IMPORTANT — interpretation note: widening FLAT raises the RANDOM-policy
# flat baseline. At difficulty ~0.7 (where the warm-start sits) the random
# FLAT fraction goes from ~0.35 (old labels) to ~0.55-0.60 (new labels).
# So success now means flat_fraction > 0.60 with rising trajectory and a
# clear separation from 1 - (unsafe + lipped) area, NOT > 0.50.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
cd /home/anyone/projects/dogb
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/dogb/logs/surface_warmstart_3b_flat35_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 30h surface (stage 3B, flat35) WARM-START training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Surface"
  echo "  seed: 42"
  echo "  warm-start: warmstart_rough_25600/model_25600_surface.pt"
  echo "  active stage: STAGE_3B   weights: r_flat=+0.04, p_unsafe=-0.04"
  echo "  SLOPE_UNSAFE_DEG: 35.0  (widened from 15.0 — see HANDOFF §15.3)"
  echo "  obs dim: 244 (policy), 259 (critic); height scan 187 rays"
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
