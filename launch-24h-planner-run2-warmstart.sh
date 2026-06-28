#!/usr/bin/env bash
# Stage-3 FOOTHOLD-PLANNER Run 2: resume Run 1 with tightened landing
# precision rewards.
#
# Run 1 result (2026-06-02_18-27-32, iter 13795):
#   - planner picked FLAT (surface_aware_target 0.169, vs ~0 at warm-start)
#   - controller tracked targets (foot_to_target_tracking 1.76 / max 2.0)
#   - 86% of landings within 8 cm of target (landed_near_target 0.064/s)
#   - BUT actual flat_fraction = 0.579, vs area-baseline 0.562 → +0.017,
#     statistically indistinguishable from no improvement.
#
# Diagnosis: per-event landing reward at weight 0.5 was negligible
# (~0.06/s) vs per-step tracking at 3.5/s. Policy optimized "be NEAR
# target during swing" not "LAND ON target". The 8 cm tolerance also
# exceeds the 5 cm LIPPED-band width, so "on-target" landings spill into
# LIPPED/UNSAFE.
#
# Run 2 fix (single-variable test, hold everything else constant):
#   w_landed:  0.5 → 5.0   (10× — match per-step authority on ~150 landings/ep)
#   tolerance: 0.08 → 0.04 m   (sub-LIPPED-band width, forces real precision)
#
# Same warm-start chain, this time resuming from Run 1's final checkpoint
# (the planner+controller already work mechanically; this run tightens the
# precision constraint on top).
#
# Predictions:
#   - If precision-bonus is the missing mechanism: flat_fraction climbs
#     toward 0.70+ within 5k iters (the planner already picks FLAT, the
#     controller already tracks, this just makes landings hit the target).
#   - If it doesn't work either: the planner mechanism as designed can't
#     steer landings even with per-landing pressure. That'd close the
#     "explicit foothold targeting" branch of the design space.
#
# 24h budget; --kill-after=120 in case of Isaac Lab shutdown bug
# (HANDOFF §15.6).
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
cd /home/anyone/projects/dogb
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/dogb/logs/planner_run2_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h foothold-planner RUN 2 (precision-bonus) at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-FootholdPlanner"
  echo "  seed: 42"
  echo "  resume from: 2026-06-02_18-27-32/model_13700.pt  (Run 1 final-ish)"
  echo "  CHANGES vs Run 1: w_landed 0.5→5.0, landed_tolerance 0.08→0.04"
  echo "  (single-variable test; surface_aware_landing & planner reward weights unchanged)"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$LOG_DIR/meta.txt"
exec timeout --preserve-status --signal=TERM --kill-after=120 86400 \
  python ~/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2-FootholdPlanner \
    --headless \
    --num_envs 4096 \
    --max_iterations 200000 \
    --seed 42 \
    --resume \
    --load_run 2026-06-02_18-27-32 \
    --checkpoint model_13700.pt \
  >"$LOG_DIR/train.log" 2>&1
