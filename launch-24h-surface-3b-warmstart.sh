#!/usr/bin/env bash
# Stage-3-surface (Stage 3B) training, WARM-STARTED from the Stage-2 rough
# walker (model_25600.pt) via the partial-transfer checkpoint.
#
# Differs from launch-24h-surface-3a-warmstart.sh ONLY in:
#   - ACTIVE_STAGE in mdp/surface_rewards.py is now STAGE_3B
#   - log dir prefix surface_warmstart_3b_*
# Same warm-start checkpoint, same task, same num_envs/seed/timeout. This is
# a clean ablation: 3A vs 3B isolates the effect of adding p_unsafe=-0.04.
#
# Stage 3B weights (from mdp/surface_rewards.py STAGE_WEIGHTS):
#   r_flat=+0.04, p_lipped=0.0, p_pocket=0.0, p_unsafe=-0.04
# Hypothesis (per SURFACE-3A-TB-CHECKLIST.md B2): a same-magnitude unsafe
# penalty is enough to push landing_fraction_flat above the random baseline,
# where positive-only (3A) was not. Stage-3A baseline at iter 17051:
#   flat=0.34 / unsafe=0.41 / lipped=0.23 / oob=0.026
# Kill criterion for 3B (checklist B3 forward note): unsafe fraction must
# drop below ~0.05 by iter 5000.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
# cwd must be dogb so rsl_rl writes logs/rsl_rl/... under dogb.
cd /home/anyone/projects/dogb
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/dogb/logs/surface_warmstart_3b_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 30h surface (stage 3B) WARM-START training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-Surface"
  echo "  seed: 42"
  echo "  warm-start: warmstart_rough_25600/model_25600_surface.pt (from rough model_25600)"
  echo "  active stage: STAGE_3B   weights: r_flat=+0.04, p_unsafe=-0.04, p_lipped=0, p_pocket=0"
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
