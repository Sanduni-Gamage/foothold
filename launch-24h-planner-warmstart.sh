#!/usr/bin/env bash
# Stage-3 FOOTHOLD-PLANNER (hierarchical, with target output head) training,
# WARM-STARTED from the surface task's WS-3b-flat35 final checkpoint
# (model_16800.pt) via make_planner_warmstart.py.
#
# What's new vs the surface task: the policy now has a per-foot xy target
# output head (8 dims; 12 joint + 8 target = 20 actions total). Two new
# rewards: surface_aware_target (planner picks FLAT targets, per-step) and
# foot_to_target_tracking (controller follows the targets during swing).
#
# Why: three surface runs (3a, 3b, 3b-flat35) demonstrated per-footfall
# reward at any reasonable magnitude/labelling can't move flat fraction
# above area-proportional random. Per the design analysis (HANDOFF §15),
# this is because foot placement is downstream of gait dynamics and a
# per-event reward lacks authority. The planner head makes the foothold
# decision an explicit policy output, sidestepping that authority gap.
#
# Why the warm-start works: the planner's new action head is zero-init'd,
# so at iter 0 it outputs target = nominal hip xy = roughly where the
# Stage-2 + surface walker was already placing feet. So initial behavior
# is gait-identical to the source. Training pulls targets toward FLAT and
# the tracking term makes the controller follow.
#
# Initial weights (will tune per Run 1 data):
#   r_target_flat   = +0.10   (per-foot per-step on FLAT target)
#   p_target_unsafe = -0.10
#   w_tracking      =  2.0
#   w_landed        =  0.5
#
# Time budget: 24h (86400s) -- surface runs converged by ~iter 10k.
# Time saved vs 30h lets us turn over a second run if Run 1 needs tweaks.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset CUDA_HOME
export CUDA_VISIBLE_DEVICES=1
# cwd must be dogb so rsl_rl writes logs/rsl_rl/... under dogb.
cd /home/anyone/projects/dogb
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/anyone/projects/dogb/logs/planner_warmstart_${RUN_TS}"
mkdir -p "$LOG_DIR"
{
  echo "starting 24h foothold-planner WARM-START training at $(date)"
  echo "  GPU: 1"
  echo "  num_envs: 4096"
  echo "  task: Unitree-Go2-FootholdPlanner"
  echo "  seed: 42"
  echo "  warm-start: warmstart_surface_3b_flat35_16800/model_planner_warmstart.pt"
  echo "  action dim: 12 joint + 8 foot-target = 20"
  echo "  obs dim:    260 (policy = 244 surface + 8 last_action growth + 8 foot_targets), 275 (critic = 259+16)"
  echo "  surface_labels SLOPE_UNSAFE_DEG: 35.0 (widened, inherited from surface task)"
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
    --load_run warmstart_surface_3b_flat35_16800 \
    --checkpoint model_planner_warmstart.pt \
  >"$LOG_DIR/train.log" 2>&1
