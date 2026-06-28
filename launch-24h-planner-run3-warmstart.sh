#!/usr/bin/env bash
# Stage-3 FOOTHOLD-PLANNER Run 3: middle-ground precision-bonus weight
# (back off from Run 2's reward-gamed value, up from Run 1's noise floor).
#
# Run 1 (2026-06-02): w_landed=0.5  tol=0.08 → flat_frac 0.579 (no movement)
# Run 2 (2026-06-08): w_landed=5.0  tol=0.04 → landings_on_target dominated;
#                     gait collapsed (track_lin_vel_xy 0.43, terrain_levels 0).
# Run 3 (2026-06-08): w_landed=1.5  tol=0.04 (this run)
#                     3× Run 1's weight, with the tighter tolerance kept.
#                     Target authority: ~6% of velocity tracking, in the
#                     5-15% shaping band that worked elsewhere.
#
# Same warm-start chain. Resumes from Run 1's model_13700.pt (the LAST
# working planner+controller checkpoint -- Run 2's checkpoints are in a
# degenerate gait basin and not worth restarting from).
#
# Hypothesis: at this intermediate weight the precision bonus has enough
# authority to shift landings (the per-event mechanism we know works from
# Run 2's behavior) without dominating velocity tracking (the Run 2 failure
# mode). If flat_fraction rises above ~0.62 while velocity reward stays
# ≥0.70 of Stage-2 baseline, the planner mechanism is validated.
#
# Kill criteria (locked in advance):
#   track_lin_vel_xy < 0.55 at iter ~15500 → too aggressive, drop to 1.0
#   flat_fraction still at ~0.58 by iter ~17000 → 1.5 too weak; mechanism
#     has a narrower sweet spot than (1.5, 5.0) → consider 2.5
#   terrain_levels < 0.4 → curriculum demoted, gait broken
#
# 24h budget, --kill-after=120 shutdown bug fix.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
cd /home/anyone/projects/dogb
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/dogb/logs/planner_run3_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h foothold-planner RUN 3 (precision-bonus moderated) at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-FootholdPlanner"
  echo "  seed: 42"
  echo "  resume from: 2026-06-02_18-27-32/model_13700.pt  (Run 1's last working ckpt)"
  echo "  CHANGES vs Run 2: w_landed 5.0→1.5  (tol stays at 0.04)"
  echo "  CHANGES vs Run 1: w_landed 0.5→1.5, tol 0.08→0.04"
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
