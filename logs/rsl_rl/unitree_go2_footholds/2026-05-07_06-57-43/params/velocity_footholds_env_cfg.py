"""Go2 task with permutation-invariant foothold reaching.

Stage-3 of the Unitree pipeline. The robot must step on a forward chain of
stepping-stone targets. Any of the 4 feet may step on any of the footholds —
assignment is implicit, governed by the symmetric reward function.

Design choices (drawn from HIGH-LEVEL-RESEARCH.md §14):
* **Virtual footholds on flat terrain (v1)**: no actual stepping stones in
  the scene; instead a per-env buffer of (x, y) target positions, with a
  reward that fires when any foot lands within tolerance ε of an unclaimed
  target. Avoids the "heightfield doesn't know which cells are valid
  footholds" problem at the cost of physical realism. Future v2 should
  add real platform meshes.
* **Permutation invariance lives in the reward**, not the architecture.
  Reward function is symmetric in foot index. Network is a vanilla MLP
  with extra obs dims for the K nearest unclaimed footholds.
* **Warm-start from the rough-terrain stage-2 walker** (model_25600.pt) by
  default — the rough walker is a stronger initialization than scratch.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import foothold_state

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


@configclass
class FootholdSceneCfg(BaseSceneCfg):
    """Scene: rough terrain mix + virtual foothold targets (visualized via debug
    markers in `foothold_state`). The robot walks on rough terrain and is
    rewarded for stepping on the virtual targets — combining Stage-2 rough
    locomotion with Stage-3 foothold-aware stepping.
    """

    def __post_init__(self):
        # rough terrain mix (random_rough, slopes, boxes, pyramid_stairs)
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        # remove height_scanner — not used in foothold task
        self.height_scanner = None


@configclass
class FootholdEventCfg(BaseEventCfg):
    """Inherit base events; add a reset event that regenerates footholds."""
    # NOTE: must run AFTER reset_base (defined in BaseEventCfg) — that's why
    # this term is added here as the LAST reset event in FootholdEventCfg.
    # Order in EventManager is by declaration (last-declared = last-applied).
    reset_footholds = EventTerm(
        func=foothold_state.reset_footholds,
        mode="reset",
        params={
            "n_stones": 20,
            "spacing": 0.50,   # 0.30 too tight — Go2 stance is 0.40m fore-aft
            "jitter": 0.15,
        },
    )


@configclass
class FootholdObsCfg(BaseObsCfg):
    """Extends base obs with foothold-set view + per-foot positions."""

    @configclass
    class PolicyCfg(ObsGroup):
        # standard proprio (matches base PolicyCfg)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        # NEW: foothold-set view (K=4 nearest unclaimed in robot xy frame)
        nearest_footholds = ObsTerm(
            func=mdp.nearest_unclaimed_footholds,
            params={"asset_cfg": SceneEntityCfg("robot"), "k": 4},
            clip=(-15.0, 15.0),
        )
        # NEW: own foot xyz in robot frame
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # privileged: clean foothold obs
        nearest_footholds_clean = ObsTerm(
            func=mdp.nearest_unclaimed_footholds,
            params={"asset_cfg": SceneEntityCfg("robot"), "k": 4},
            clip=(-15.0, 15.0),
        )
        foot_positions_clean = ObsTerm(
            func=mdp.foot_positions_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
            clip=(-2.0, 2.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class FootholdRewardsCfg(BaseRewardsCfg):
    """Extends base rewards with foothold-targeting terms.

    NB: `foothold_landing` MUST be the first foothold-related term — it drives
    the per-step claim-state update that other terms read. The base rewards
    are unchanged; foothold terms are added on top.
    """
    # event-based, MUST come before other foothold terms (drives claim update)
    foothold_landing = RewTerm(
        func=mdp.foothold_landing,
        weight=9.4,  # Choi/Raibo κ_ts1
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tolerance": 0.10,
            "sigma": 0.05,
            "contact_threshold": 5.0,
        },
    )
    foothold_progress = RewTerm(
        func=mdp.foothold_progress,
        weight=0.30,  # Choi/Raibo k_td
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    off_foothold_landing = RewTerm(
        func=mdp.off_foothold_landing,
        # Reduced from 1.0 → 0.3 after v1 run: at 1.0 with continuous-contact
        # firing this term dominated all positive rewards. Even at 0.3 with
        # event-based + smooth ramp it should be a soft pressure, not a wall.
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tolerance": 0.15,
            "contact_threshold": 5.0,
        },
    )
    # claim_advance removed — was redundant with foothold_landing (both
    # returned a per-step claim count). Keeping foothold_landing only.


@configclass
class RobotFootholdEnvCfg(RobotEnvCfg):
    """Go2 permutation-invariant foothold-reaching task (Stage-3)."""

    scene: FootholdSceneCfg = FootholdSceneCfg(num_envs=4096, env_spacing=2.5)
    events: FootholdEventCfg = FootholdEventCfg()
    observations: FootholdObsCfg = FootholdObsCfg()
    rewards: FootholdRewardsCfg = FootholdRewardsCfg()

    def __post_init__(self):
        if hasattr(self.scene, "__post_init__"):
            self.scene.__post_init__()
        # replicate RobotEnvCfg.__post_init__ but skip height_scanner update
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        # rough terrain — terrain_levels curriculum re-enabled (inherited from
        # base CurriculumCfg). Will advance/demote based on distance traversed.
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True


@configclass
class RobotFootholdPlayEnvCfg(RobotFootholdEnvCfg):
    scene: FootholdSceneCfg = FootholdSceneCfg(num_envs=8, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
