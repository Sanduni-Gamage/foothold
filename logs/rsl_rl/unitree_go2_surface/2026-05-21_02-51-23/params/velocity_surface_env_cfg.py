"""Go2 task with surface-class-aware foothold reward (Stage-3 surface).

Successor to the virtual-foothold task (``Unitree-Go2-Footholds``). Same
swing->stance gating, but the reward is gated on the per-tile surface label
under the foot at the moment of landing (FLAT vs LIPPED vs UNSAFE), with the
labels computed once at terrain build time. The policy gets a downward
height-scan as perception (stock Isaac Lab Go2 rough config, verbatim).

Key differences vs ``Unitree-Go2-Footholds``:

* Terrain: ``TRAPEZOIDAL_BUMPS_SURFACE_CFG`` (heightfield, labels emitted).
* Observations: drop ``nearest_unclaimed_footholds``; add ``height_scan``
  (187 dims, same params as stock rough). Keep ``foot_positions_robot_frame``.
* Rewards: replace the four foothold-targeting terms with one
  ``surface_aware_landing`` term + five low-weight stats terms emitting
  TensorBoard counts via the ``Episode_Reward/`` plumbing.
* No warm-start: obs shape changes (65 -> 244 dims) so no v2 checkpoint
  loads cleanly. Fresh run is the intended workflow.

Stage selection lives in ``mdp.surface_rewards.ACTIVE_STAGE`` — one line.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import surface_rewards as _surf_rew
from unitree_rl_lab.terrains import TRAPEZOIDAL_BUMPS_SURFACE_CFG

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg as BaseEventCfg,
    ObservationsCfg as BaseObsCfg,
    RewardsCfg as BaseRewardsCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
    RobotSceneCfg as BaseSceneCfg,
    TerminationsCfg,
    ROBOT_CFG,
)


# ---------------------------------------------------------------------------
# Scene: trapezoidal bumps (label-emitting variant), height scanner restored.
# ---------------------------------------------------------------------------
@configclass
class SurfaceSceneCfg(BaseSceneCfg):
    """Heightfield bumps + the stock downward height-scanner."""

    def __post_init__(self):
        # Swap the inherited terrain (was PYRAMID_RIDGED_STAIRS_CFG) for the
        # labels-emitting trapezoidal-bumps generator. Friction matches the
        # Stage-2 value (0.75) so we can reason about transfer from the
        # rough-terrain walker.
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TRAPEZOIDAL_BUMPS_SURFACE_CFG,
            max_init_terrain_level=1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.75,
                dynamic_friction=0.75,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=(
                    f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                    f"TilesMarbleSpiderWhiteBrickBondHoned/"
                    f"TilesMarbleSpiderWhiteBrickBondHoned.mdl"
                ),
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        # height_scanner stays AS-IS from BaseSceneCfg (RayCasterCfg at
        # /Robot/base, 17x11 grid, yaw-aligned). FootholdSceneCfg null'd it;
        # SurfaceSceneCfg restores it implicitly by not overriding here.


# ---------------------------------------------------------------------------
# Observations: drop nearest_unclaimed_footholds, add height_scan.
# ---------------------------------------------------------------------------
@configclass
class SurfaceObsCfg(BaseObsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        # Stage-2 proprio set, verbatim.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        # NEW: vertical height scan (187 dims, stock Isaac Lab Go2 rough).
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        # Kept from FootholdObsCfg.
        foot_positions = ObsTerm(
            func=mdp.foot_positions_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # Privileged (no corruption): adds base_lin_vel, joint_effort,
        # plus a clean height_scan and foot_positions.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        height_scan_clean = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )
        foot_positions_clean = ObsTerm(
            func=mdp.foot_positions_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
            clip=(-2.0, 2.0),
        )

    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards: keep base velocity terms; add surface-aware landing + 5 stats.
# ---------------------------------------------------------------------------
_STAGE_W = _surf_rew.STAGE_WEIGHTS[_surf_rew.ACTIVE_STAGE]


@configclass
class SurfaceRewardsCfg(BaseRewardsCfg):
    """Inherits the full Stage-2 reward block unchanged; adds one task
    reward (``surface_aware_landing``) and five tiny-weight stats terms
    that surface to TensorBoard via the standard Episode_Reward/ path.
    """

    surface_aware_landing = RewTerm(
        func=mdp.surface_aware_landing,
        weight=1.0,  # class-specific weights live in params, not here.
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
            "r_flat": _STAGE_W["r_flat"],
            "p_lipped": _STAGE_W["p_lipped"],
            "p_pocket": _STAGE_W["p_pocket"],
            "p_unsafe": _STAGE_W["p_unsafe"],
        },
    )

    # ---- TensorBoard-only counts. weight=1e-6 documented in surface_stats.
    landing_fraction_flat = RewTerm(
        func=mdp.landing_class_fraction,
        weight=mdp.STATS_TERM_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
            "class_id": int(mdp.surface_labels.FLAT),
        },
    )
    landing_fraction_lipped = RewTerm(
        func=mdp.landing_class_fraction,
        weight=mdp.STATS_TERM_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
            "class_id": int(mdp.surface_labels.LIPPED),
        },
    )
    landing_fraction_unsafe = RewTerm(
        func=mdp.landing_class_fraction,
        weight=mdp.STATS_TERM_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
            "class_id": int(mdp.surface_labels.UNSAFE),
        },
    )
    landing_fraction_out_of_bounds = RewTerm(
        func=mdp.landing_class_fraction,
        weight=mdp.STATS_TERM_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
            "class_id": int(mdp.surface_labels.OUT_OF_BOUNDS),
        },
    )
    landings_per_episode = RewTerm(
        func=mdp.landings_per_episode_inc,
        weight=mdp.STATS_TERM_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_threshold": 5.0,
        },
    )


# ---------------------------------------------------------------------------
# Env cfgs
# ---------------------------------------------------------------------------
@configclass
class RobotSurfaceEnvCfg(RobotEnvCfg):
    """Go2 surface-aware foothold task (Stage-3 surface)."""

    scene: SurfaceSceneCfg = SurfaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: SurfaceObsCfg = SurfaceObsCfg()
    rewards: SurfaceRewardsCfg = SurfaceRewardsCfg()

    def __post_init__(self):
        # Run BaseSceneCfg's __post_init__ first (sets terrain via our override).
        if hasattr(self.scene, "__post_init__"):
            self.scene.__post_init__()
        # Replicate RobotEnvCfg.__post_init__ inline (height_scanner is kept
        # this time, so we do call its update_period set).
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Terrain curriculum on (set by base when terrain_levels is registered).
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True

        # Sanity print of the active stage + obs dim, evaluated lazily after
        # the observation manager is built (so we don't recompute it here).
        # ManagerBasedRLEnv prints obs dims on creation; the active stage is
        # printed via the import-time module log line below.
        print(
            f"[surface-task] active stage: {_surf_rew.ACTIVE_STAGE.name}"
            f"  weights: {_STAGE_W}"
        )


@configclass
class RobotSurfacePlayEnvCfg(RobotSurfaceEnvCfg):
    scene: SurfaceSceneCfg = SurfaceSceneCfg(num_envs=8, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Side-on viewer (matches the Stage-2 play viewer for parity).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.0, 4.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)
