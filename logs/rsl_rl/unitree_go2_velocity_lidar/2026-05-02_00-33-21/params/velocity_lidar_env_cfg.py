"""Go2 velocity-tracking task with onboard L1 LiDAR over rough terrain.

Differences from the base ``velocity_env_cfg``:

* The downward-looking ``height_scanner`` (``GridPatternCfg``) is removed.
  No foot-level terrain perception is provided; the policy must rely on
  proprioception + the L1 LiDAR.
* A new ``lidar_scanner`` is added: a ``RayCasterCfg`` using
  ``LidarPatternCfg`` configured to match the Unitree 4D LiDAR-L1 (User
  Manual v1.1, June 2024). 360° H × 90° V upper hemisphere, 11 Hz frame
  rate, ~30 m range, mounted upright on the Go2 base.
* ``COBBLESTONE_ROAD_CFG`` is replaced with a rough terrain mix
  (random_rough, slopes, boxes, pyramid stairs) to give the L1
  meaningful obstacle structure.
* Policy obs adds a 1024-d log-distance LiDAR vector with per-ray noise
  (σ=2 cm, matching L1 ±2 cm spec) and 10% dropout. Critic gets clean
  distances as a privileged signal.

Source for the L1 specs: Unitree 4D LiDAR-L1 User Manual v1.1 (2024.06),
Parameter Specifications page (FOV 360°×90°, horizontal scan 11 Hz,
±2.0 cm accuracy, 0.05–30 m range).
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
    RobotSceneCfg,
    TerminationsCfg,
    ROBOT_CFG,
)


# Rough terrain mix: enables the random_rough / slope / boxes / stairs
# subterrains that Unitree shipped commented-out in the base config.
ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


# Unitree L1 ray pattern (User Manual v1.1, 2024.06).
# Vertical FOV (0, 90) = upper hemisphere, sensor frame: 0 deg = horizontal,
# 90 deg = straight up. The L1 housing occludes the lower hemisphere.
# Channels=32 is a moderate-density approximation of the L1's vertical scan
# (the real sensor sweeps 180 Hz vertically × 11 Hz horizontally; we sample
# a static 32-beam fan per frame and tick at 11 Hz).
L1_LIDAR_PATTERN = patterns.LidarPatternCfg(
    channels=32,
    vertical_fov_range=(0.0, 90.0),
    horizontal_fov_range=(-180.0, 180.0),
    horizontal_res=5.0,  # 72 azimuth steps × 32 channels = 2304 rays
)


@configclass
class LidarSceneCfg(RobotSceneCfg):
    """Scene with the L1 lidar replacing the height_scanner."""

    def __post_init__(self):
        # swap terrain to the rough mix
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAIN_CFG,
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

        # remove the foot-level height scanner — policy gets no foot perception
        self.height_scanner = None

        # add the L1 LiDAR raycaster
        self.lidar_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            # mount the L1 ~12 cm above the base origin; approximate Go2 head
            # mounting plate. Replace with the canonical mount transform when
            # verified against the Go2 USD's L1 prim.
            offset=RayCasterCfg.OffsetCfg(pos=(0.15, 0.0, 0.12)),
            ray_alignment="yaw",
            pattern_cfg=L1_LIDAR_PATTERN,
            max_distance=30.0,
            attach_yaw_only=False,
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


@configclass
class LidarObservationsCfg(ObservationsCfg):
    """Observations with proprioception + L1 lidar; no height scan."""

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
        # L1 lidar: 2304-dim log-distance vector with per-ray noise + dropout
        lidar = ObsTerm(
            func=mdp.lidar_distances,
            params={
                "sensor_cfg": SceneEntityCfg("lidar_scanner"),
                "max_range": 30.0,
                "min_range": 0.05,
                "noise_std": 0.02,
                "dropout_prob": 0.10,
                "log_scale": True,
            },
            clip=(0.0, 5.0),  # log1p(30) ≈ 3.43, leave headroom
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # privileged: full state + clean lidar
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
        lidar_clean = ObsTerm(
            func=mdp.lidar_distances,
            params={
                "sensor_cfg": SceneEntityCfg("lidar_scanner"),
                "max_range": 30.0,
                "min_range": 0.05,
                "noise_std": 0.0,
                "dropout_prob": 0.0,
                "log_scale": True,
            },
            clip=(0.0, 5.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RobotLidarEnvCfg(RobotEnvCfg):
    """Go2 velocity task over rough terrain with L1 LiDAR perception."""

    scene: LidarSceneCfg = LidarSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: LidarObservationsCfg = LidarObservationsCfg()

    def __post_init__(self):
        # apply LidarSceneCfg overrides (terrain swap + sensor swap)
        if hasattr(self.scene, "__post_init__"):
            self.scene.__post_init__()

        # replicate RobotEnvCfg.__post_init__ but skip the height_scanner
        # update_period line, since we removed that sensor.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        # tick the L1 at the real 11 Hz rate
        self.scene.lidar_scanner.update_period = 1.0 / 11.0

        # terrain-curriculum gating, copied from base
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotLidarPlayEnvCfg(RobotLidarEnvCfg):
    """Play variant with fewer envs and easier terrain."""

    scene: LidarSceneCfg = LidarSceneCfg(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
