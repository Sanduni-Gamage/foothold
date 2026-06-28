"""Go2 task with hierarchical FOOTHOLD PLANNER head (Stage-3 planner).

Inherits the surface task (terrain, height_scanner, surface labels) and
adds a planner head: the policy outputs per-foot xy target offsets each
step, those become world-frame targets via the base pose, and two new
rewards shape the loop:

  * ``surface_aware_target`` — reward the TARGET's surface class
    (FLAT/UNSAFE etc.). This is where the planner learns to PICK flat
    targets. Per-step, every step (not just landings).
  * ``foot_to_target_tracking`` — Gaussian reward for the SWING foot
    being close to its target. This is where the controller learns to
    HIT the planner's targets.

Plus a small landing-on-target bonus (`landed_near_target`) so successful
target-hits are reinforced as discrete events.

The existing per-footfall ``surface_aware_landing`` is kept at a tiny
weight as auxiliary shaping — once the planner is good, landings should
naturally end up on FLAT and this term will go positive on its own.

Obs deltas vs surface task:
  policy : add foot_targets (8) -> 244 + 8 = 252
  critic : add foot_targets_clean (8) -> 259 + 8 = 267

Action deltas vs surface task:
  add FootTargetAction (8 dim) -> 12 joint + 8 target = 20 actions

Warm-start: ``make_planner_warmstart.py`` extends a surface-task checkpoint
(244-in / 12-out actor) to planner shape (252-in / 20-out) with the new
input cols zero-init'd and the new action head zero-init'd. With FootTarget
offsets scaled by 0.15 and clipped to ±0.30 around nominal hip, the
zero-init action means "target = nominal hip xy", which is roughly where
the gait was placing feet anyway. So warm-start is function-near-identical
at iteration 0, and training pulls targets toward FLAT from there.
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import surface_rewards as _surf_rew

from .velocity_env_cfg import ActionsCfg as BaseActionsCfg
from .velocity_surface_env_cfg import (
    RobotSurfaceEnvCfg,
    RobotSurfacePlayEnvCfg,
    SurfaceObsCfg as BaseSurfaceObsCfg,
    SurfaceRewardsCfg as BaseSurfaceRewardsCfg,
)


# ---------------------------------------------------------------------------
# Actions: existing 12-DOF joint targets + new 8-dim foot-target head.
# ---------------------------------------------------------------------------
@configclass
class PlannerActionsCfg(BaseActionsCfg):
    """Joint position action (inherited) + planner foot-target head."""

    FootTarget = mdp.FootTargetActionCfg(
        asset_name="robot",
        foot_body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
    )


# ---------------------------------------------------------------------------
# Observations: existing surface set + foot_targets (8 dim) on both groups.
# ---------------------------------------------------------------------------
@configclass
class PlannerObsCfg(BaseSurfaceObsCfg):
    @configclass
    class PolicyCfg(BaseSurfaceObsCfg.PolicyCfg):
        foot_targets = ObsTerm(
            func=mdp.foot_targets_robot_frame,
            clip=(-2.0, 2.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(BaseSurfaceObsCfg.CriticCfg):
        foot_targets_clean = ObsTerm(
            func=mdp.foot_targets_robot_frame,
            clip=(-2.0, 2.0),
        )

    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards: inherit Stage-2 + surface task rewards; add the planner triplet.
# ---------------------------------------------------------------------------
# Initial planner reward weights (Run 1 — minimal-scope per design doc).
# r_flat / p_unsafe at ±0.10 give per-step contribution targets in the
# 5-15% band of track_lin_vel_xy (1.27/s baseline). Tracking weight tuned
# so the swing-foot gradient is comparable.
PLANNER_R_TARGET_FLAT = +0.10
PLANNER_P_TARGET_UNSAFE = -0.10
PLANNER_W_TRACKING = 2.0
# Calibration history:
#   Run 1 (2026-06-02): w_landed=0.5, tol=0.08. Contributed ~0.06/s --
#     negligible vs tracking's 3.5/s -- so policy optimized "be near
#     target during swing" not "land on target". Result: flat_frac 0.579
#     (vs 0.562 baseline, no movement).
#   Run 2 (2026-06-08): bumped 0.5 -> 5.0 (10x) and tightened tol 0.08 -> 0.04.
#     Worked TOO well: landed_near_target dwarfed velocity reward (160/s mean
#     reward, vs Run 1's 49). Policy gamed it by standing roughly still and
#     tapping precise landings on tiny targets, ignoring forward velocity.
#     track_lin_vel_xy collapsed to 0.43 (kill criterion < 0.55), terrain_levels
#     dropped to 0.0 (curriculum demoted), error_vel_xy 1.02. Killed at iter 15800.
#   Run 3 (2026-06-08): w_landed=1.5 (3x original, not 10x), keep tol=0.04.
#     Arithmetic: 50 landings-on-target/ep × 1.5 × 0.02 dt = 1.5/ep weighted,
#     /20 = 0.075/s = ~6% of track_lin_vel_xy ~1.27/s. In the 5-15% shaping
#     band -- gives per-event authority comparable to other shaping terms
#     without dominating velocity reward.
PLANNER_W_LANDED = 1.5
PLANNER_LANDED_TOLERANCE = 0.04


@configclass
class PlannerRewardsCfg(BaseSurfaceRewardsCfg):
    """Adds the three planner rewards on top of the inherited surface set.

    The inherited ``surface_aware_landing`` stays at its Stage-3B weights
    (r_flat=+0.04, p_unsafe=-0.04) as small per-landing shaping. Once the
    planner works, actual landings should track FLAT for free.
    """

    surface_aware_target = RewTerm(
        func=mdp.surface_aware_target,
        weight=1.0,
        params={
            "r_flat": PLANNER_R_TARGET_FLAT,
            "p_lipped": 0.0,
            "p_pocket": 0.0,
            "p_unsafe": PLANNER_P_TARGET_UNSAFE,
        },
    )
    foot_to_target_tracking = RewTerm(
        func=mdp.foot_to_target_tracking,
        weight=PLANNER_W_TRACKING,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
            "std": 0.10,
        },
    )
    landed_near_target = RewTerm(
        func=mdp.landed_near_target,
        weight=PLANNER_W_LANDED,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
            "tolerance": PLANNER_LANDED_TOLERANCE,
        },
    )


# ---------------------------------------------------------------------------
# Env cfgs
# ---------------------------------------------------------------------------
@configclass
class RobotPlannerEnvCfg(RobotSurfaceEnvCfg):
    """Stage-3 foothold-planner task (inherits surface task)."""

    actions: PlannerActionsCfg = PlannerActionsCfg()
    observations: PlannerObsCfg = PlannerObsCfg()
    rewards: PlannerRewardsCfg = PlannerRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Sanity print so a launcher can grep for it.
        print(
            "[planner-task] action dim: 12 joint + 8 foot-target = 20  | "
            f"r_flat={PLANNER_R_TARGET_FLAT}  p_unsafe={PLANNER_P_TARGET_UNSAFE}  "
            f"w_track={PLANNER_W_TRACKING}  "
            f"w_landed={PLANNER_W_LANDED}  landed_tol={PLANNER_LANDED_TOLERANCE}  | "
            f"surface stage (auxiliary): {_surf_rew.ACTIVE_STAGE.name}"
        )


@configclass
class RobotPlannerPlayEnvCfg(RobotPlannerEnvCfg, RobotSurfacePlayEnvCfg):
    """Play variant — 8 envs + viewer config inherited from RobotSurfacePlayEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
