# Go2 + LiDAR + Rough-Terrain RL — Research Findings

Goal: train a Unitree Go2 to navigate rough terrain using its onboard 3D LiDAR
(Unitree L1) as a policy input, on the Isaac Lab v2.3.2 + Isaac Sim 5.1.0 stack
already installed on this machine.

Every claim below is backed by a primary source — official repo, README, or
local file path.

---

## 1. Unitree's own repos (exhaustive)

Verified by listing the org's 30+ repos and reading each candidate's README.

| Repo | Trains RL? | LiDAR in policy obs? | Source |
|---|---|---|---|
| [`unitree_rl_lab`][url-rllab] (installed) | ✅ Go2/H1/G1 RL | ❌ uses `RayCasterCfg` as a `GridPattern` height-scan only | [local file][file-rllab-go2] line 96–101 |
| [`unitree_rl_mjlab`][url-mjlab] | ✅ Go2/H1/G1/A2/R1 RL | ❌ proprio + IMU only | repo README |
| [`unitree_sim_isaaclab`][url-simlab] | ❌ data-collection / teleop only | ❌ G1/H1 manipulation tasks; no Go2 | repo README |
| [`unitree_rl_gym`][url-rlgym] | ✅ legacy Isaac Gym RL | ❌ proprio only; **deprecated path on Blackwell** | repo README |
| [`unitree_lerobot`][url-lerobot] | imitation/manipulation | ❌ G1-focused | repo README |
| [`point_lio_unilidar`][url-pointlio] | ❌ runtime SLAM/odometry | n/a — operates on real LiDAR data, not training | repo README |
| [`unilidar_sdk2`][url-unilidar] | ❌ hardware SDK | n/a — for L2 device I/O | repo README |

[url-rllab]: https://github.com/unitreerobotics/unitree_rl_lab
[url-mjlab]: https://github.com/unitreerobotics/unitree_rl_mjlab
[url-simlab]: https://github.com/unitreerobotics/unitree_sim_isaaclab
[url-rlgym]: https://github.com/unitreerobotics/unitree_rl_gym
[url-lerobot]: https://github.com/unitreerobotics/unitree_lerobot
[url-pointlio]: https://github.com/unitreerobotics/point_lio_unilidar
[url-unilidar]: https://github.com/unitreerobotics/unilidar_sdk2
[file-rllab-go2]: ~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py

**Conclusion (Unitree):** Unitree publishes the LiDAR **hardware SDK** and a
**SLAM frontend** (Point-LIO), but **no RL recipe** that consumes the L1 in a
training policy.

---

## 2. Community / academic repos (Isaac Lab 2.x)

Restricted to current repos (active 2025-2026) that target Isaac Lab v2.x +
Isaac Sim 5.x. Anything Isaac-Gym-only is excluded — Isaac Gym Preview 4 is
unsupported on Blackwell sm_120 ([source: PyTorch issue #145949 thread][src-blackwell]).

[src-blackwell]: https://github.com/pytorch/pytorch/issues/145949

### Top 3 candidates ranked

#### 1. [`fan-ziqi/robot_lab`][rl-rl] — best stack match, no LiDAR yet

- **Stack match (verified):** README "Version Dependency" table pins
  `v2.3.2` of robot_lab to **Isaac Lab v2.3.2 + Isaac Sim 4.5 / 5.0 / 5.1**.
  Latest release Feb 3, 2026 (one day after Isaac Lab v2.3.2 was tagged).
- Registered Go2 task: `RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0` (verbatim
  from README quadruped table).
- **1.7k ★, 195 commits, 190 forks** — battle-tested community fork.
- **No LiDAR sensor wired in** — README has no mention of `LidarPattern`,
  `BpearlPattern`, `lidar`, or `ray_caster`. Adding `LidarPatternCfg` is the
  smallest delta of any candidate.
- Cons: no sim-to-real LiDAR story; you author the sensor config yourself.

Source: [robot_lab README — Version Dependency table & Quadruped task table][rl-rl].

[rl-rl]: https://github.com/fan-ziqi/robot_lab

#### 2. [`leggedrobotics/fdm`][rl-fdm] — best engineering, ANYmal-default

- **Stack: Isaac Lab 2.1.1** (commit `19b24c7`) — close to our 2.3.2; the
  2.1→2.3 migration is documented and small.
- Roth et al., RSS 2025. Forward-Dynamics-Model-based locomotion over rough
  terrain with **elevation-map abstraction** (sensor-agnostic — LiDAR or depth
  feeds the same map). Real sim-to-real with rosbag fine-tuning.
- **ANYmal D is the primary asset; Go2 not first-class.** You'd swap the USD
  + URDF and add a `LidarPatternCfg` raycaster feeding the elevation map.
- ~309 ★, active through 2025.

[rl-fdm]: https://github.com/leggedrobotics/fdm

#### 3. [`yang-zj1026/NaVILA-Bench`][rl-navila] + [`legged-loco`][rl-llc] — Go2 in Isaac Lab, but **proprio-only locomotion** (downgraded after verification)

- **Correction from initial draft:** the `legged-loco` README has *no mention*
  of LiDAR, point cloud, RayCaster, or perception sensors — verified by
  WebFetch. The low-level locomotion policy is **proprio-only**. NaVILA's
  high-level policy uses VLM + RGB(D) on Matterport meshes for **indoor
  visual-language navigation**, not LiDAR-based rough-terrain locomotion.
- Both repos are pinned to **Isaac Lab 1.1.0** (verbatim: *"This codebase was
  tested with Isaac Lab 1.1.0 and may not be compatible with newer versions"*)
  — porting to 2.3.2 is non-trivial because of the 1.x → 2.x API break.
- ~310 ★ / ~424 ★, MIT.
- **Net value here is low** for our LiDAR + rough-terrain goal. Demoted to
  reference-only.

[rl-navila]: https://github.com/yang-zj1026/NaVILA-Bench
[rl-llc]: https://github.com/yang-zj1026/legged-loco
[src-il2]: https://github.com/isaac-sim/IsaacLab/releases

### Notable near-misses (rejected with reason)

| Repo | Reason rejected |
|---|---|
| [`CAI23sbP/Isaaclab_Parkour`](https://github.com/CAI23sbP/Isaaclab_Parkour) | Go2 + parkour on Isaac Lab, but uses **depth camera**, not LiDAR |
| [`leggedrobotics/viplanner`](https://github.com/leggedrobotics/viplanner) | ANYmal + depth/semantic; no LiDAR |
| [`leggedrobotics/navigation_template`](https://github.com/leggedrobotics/navigation_template) | **Archived May 2025**, Isaac Lab 1.2, ZED camera, ANYmal |
| [`abizovnuralem/go2_omniverse`](https://github.com/abizovnuralem/go2_omniverse) | **Mislabelled, NOT an L1 model.** Ships `Isaac_sim/Unitree/Unitree_L1.json` filename, but the JSON's internal `name` is `"OS0 REV7 128 10hz @ 1024 resolution"` and `comment1` cites the Ouster OS0 datasheet. It's a 128-channel Ouster OS0 preset (75 m range, 100 Hz, ±45° elevation) renamed — none of the L1's actual specs (single-beam dual-motor 11/180 Hz, 30 m range, 21,600 pts/s) appear. Verified 2026-05-02 by fetching the JSON. ~1k ★, pinned to Isaac Sim 2023.1.1 + Orbit 0.3.0. **Do not use as an L1 parameter source.** |
| [`Charlescai123/isaac-wild-go2`](https://github.com/Charlescai123/isaac-wild-go2) | Isaac **Gym** (deprecated for Blackwell) |
| [`LeCAR-Lab/ABS`](https://github.com/LeCAR-Lab/ABS) | Isaac Gym Preview 4 + ZED depth, Go1 |
| [`chengxuxin/extreme-parkour`](https://github.com/chengxuxin/extreme-parkour) | Isaac Gym, depth |
| [`ZiwenZhuang/parkour`](https://github.com/ZiwenZhuang/parkour) | Isaac Gym, scandot |
| [`NtagkasAlex/phase_guided_terrain_traversal`](https://github.com/NtagkasAlex/phase_guided_terrain_traversal) | Go2 + real L1 LiDAR sim, but **MuJoCo MJX, not Isaac Lab** |
| `AME-2` (ETH, Science Robotics 2025) | No public code release found |

**Bottom line (community), revised after verification:** no public repo ships
*Go2 + simulated LiDAR + rough terrain + Isaac Lab 2.3 + Sim 5.1* end-to-end.
`robot_lab` is the cleanest base to bolt a LiDAR onto (matches our stack
exactly, 1.7k ★). `fdm` is the engineering gold standard (RSS 2025) but
ANYmal-default and uses elevation maps rather than raw LiDAR. `go2_omniverse`
isn't training-grade infrastructure but it ships an L1 sensor JSON that's
directly useful as a parameter source for our `LidarPatternCfg`. The
`NaVILA-Bench` claim of "Go2 + simulated LiDAR" did not survive verification
— its low-level locomotion is proprio-only.

---

## 3. DIY path inside the existing install

Isaac Lab v2.3.2 already ships everything needed — no new external deps.

### Available LiDAR ray patterns

From `~/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns_cfg.py`:

- **`LidarPatternCfg`** — fully configurable spinning LiDAR: `channels`,
  `vertical_fov_range`, `horizontal_fov_range`, `horizontal_res`. Suitable for
  modeling the L1's 360°×~90° hemispherical pattern.
- **`BpearlPatternCfg`** — preset for Robosense RS-Bpearl: 360°×90°, 32 beams,
  10° horizontal resolution. Geometrically similar to the L1 (which is a
  hemispherical 4D LiDAR with ~360°×90° FOV) and could be used out-of-the-box
  as a stand-in.
- **`GridPatternCfg`** — what the Go2 task currently uses for height-scan.

Source: local file `patterns_cfg.py` (`class LidarPatternCfg`, `class BpearlPatternCfg`).

### Reference: Unitree L1 4D LiDAR specs (verified, primary source)

Verified against the [Unitree 4D LiDAR-L1 User Manual v1.1 (2024.06)][l1-manual]
(official Unitree CDN PDF). Numbers are taken verbatim from pp. 2, 6, 8.

| Spec | Value | Manual page |
|---|---|---|
| FOV | 360° horizontal × 90° vertical (hemispherical) | p. 2, p. 6 |
| Effective sampling rate | 21,600 points/s | p. 2 |
| Azimuthal scanning frequency (low-speed motor, full 360° rotation) | 11 Hz | p. 2 |
| Vertical scanning frequency (high-speed reflector, sweeps 180°) | 180 Hz | p. 2 |
| Range | 0.05 m – 30 m @ 90% reflectivity | p. 2 |
| Built-in IMU | 3-axis accel + 3-axis gyro, 250 Hz push frequency | p. 2 |
| Eye safety | IEC-60825 Class 1 | p. 2 |
| Operating temperature | −10 °C to +60 °C | p. 2 |
| Size / weight | 75 × 75 × 65 mm / 230 g | p. 8 |

**The L1 is not a fixed-channel multi-beam LiDAR.** Per the *Working Principle*
section (p. 3) and *Effective FOV Range* section (p. 6), it is a **single
beam mechanically dual-scanned**: a fast reflector sweeps the beam 180°
vertically at 180 Hz while a slow motor rotates the whole assembly 360° at
11 Hz. The manual (p. 6) explicitly states *"the point cloud density of the
L1 varies in different FOV areas, with higher density near the center"* and
prints a rosette / Lissajous-like pattern, not a regular angular grid.

Derived geometry per full azimuth rotation (1/11 s ≈ 90.9 ms):

- 21,600 / 11 ≈ **1,964 points per full 360° turn**
- 21,600 / 180 = **120 points per vertical sweep**
- 180 / 11 ≈ **16.4 vertical sweeps per full azimuth turn**

[l1-manual]: https://oss-global-cdn.unitree.com/static/52b72f707b304d229d4321eea223738f.pdf

#### How Unitree's own SDK represents the L1

From [`unitreerobotics/unilidar_sdk`][unilidar-sdk] —
`unitree_lidar_sdk/include/unitree_lidar_sdk.h`:

```c++
struct PointUnitree {
    float x, y, z;
    float intensity;
    float time;
    uint32_t ring;
};

struct PointCloudUnitree {
    double stamp;
    uint32_t id;
    uint32_t ringNum;
    std::vector<PointUnitree> points;
};

struct ScanUnitree {
    double stamp;
    uint32_t id;
    uint32_t validPointsNum;
    PointUnitree points[120];   // exactly one fast-axis vertical sweep
};

struct IMUUnitree {
    double stamp;
    uint32_t id;
    float quaternion[4];
    float angular_velocity[3];
    float linear_acceleration[3];
};

enum MessageType { NONE, IMU, POINTCLOUD, RANGE, AUXILIARY, VERSION, TIMESYNC };
enum LidarWorkingMode { NORMAL, STANDBY };
```

Two structural facts the SDK confirms about the manual's specs:

- `ScanUnitree::points[120]` is a fixed 120-point buffer, exactly matching
  21,600 pts/s ÷ 180 Hz vertical sweep = 120 points per sweep. One `Scan`
  packet = one fast-axis pass.
- There is no fixed-channel-count concept. `ringNum` is a property of an
  *accumulated* `PointCloudUnitree` (how many sweeps got merged), not a
  hardware constant. A "ring" here indexes a vertical sweep within the
  current rotation, not a Velodyne-style fixed laser. This is the
  fingerprint of a single-beam dual-motor scanner; a multi-beam LiDAR
  would expose `ring ∈ [0, N_lasers)` as a hardware constant.
- `MessageType::RANGE` exists alongside `POINTCLOUD`, so raw range returns
  are queryable separately from the projected (x, y, z) cloud — useful
  if a sim front-end ever needs to mimic the wire protocol.

[unilidar-sdk]: https://github.com/unitreerobotics/unilidar_sdk

#### Does Unitree simulate the L1?

**No.** Verified across every Unitree-org repo that could plausibly ship a
sim model:

| Repo | L1 simulation? | Evidence |
|---|---|---|
| [`unilidar_sdk`][unilidar-sdk] | ❌ Hardware-only runtime driver. README describes obtaining "point cloud data and IMU data measured in our lidar"; no sim mode, no virtual sensor, no raycaster. | README + header file |
| [`unitree_mujoco`][um-mj] | ❌ Supported message types are `LowCmd`, `LowState`, `SportModeState`, `IMUState` only. No LiDAR / point cloud / raycast. | README feature list |
| [`unitree_sim_isaaclab`][um-sil] | ❌ G1 / H1-2 manipulation tasks with cameras. No L1, lidar, point cloud, or 4D sensor. | README |
| [`unitree_rl_lab`][um-rllab] | ❌ Uses `RayCasterCfg` only as a `GridPatternCfg` height scanner (different sensor entirely). No L1 / lidar references in upstream code. | `tasks/locomotion/robots/go2/velocity_env_cfg.py` lines 96–101 (verified locally) |
| [`unitree_rl_mjlab`][um-rlmj] | ❌ Proprio + IMU only. | README (per §1 of this doc) |

So the question "does the Unitree code simulate the L1 accurately?" has no
accuracy answer — there is no Unitree-authored sim model to evaluate.

#### Community sim attempts (none model the L1 architecture)

| Asset | Status |
|---|---|
| [`Unitree_L1.json` in `abizovnuralem/go2_omniverse`][go2omni] | **Mislabelled — actually an Ouster OS0 preset.** Verified 2026-05-02 by fetching the JSON; internal name is `"OS0 REV7 128 10hz @ 1024 resolution"`, parameters cite the Ouster OS0 datasheet. See §5c for full details. Not an L1 model. |
| [`Zhefan-Xu/isaac-go2-ros2`][igr2] | Publishes `/unitree_go2/lidar/point_cloud` topic but README does not document the sensor config. Would need source-code dive to determine whether it models L1 specs or uses a generic Isaac Sim built-in (e.g. Velodyne preset). Pinned to Isaac Sim 4.5 / Isaac Lab 2.1 (off-stack from our 2.3.2). |
| [`NtagkasAlex/phase_guided_terrain_traversal`][pgtt] | Uses the real L1 only at deployment. Sim is MuJoCo MJX with a heightmap sensor, not a LiDAR sim. |
| Our own `velocity_lidar_env_cfg.py` | Uses `LidarPatternCfg(channels=32, hres=5°)` regular grid. Wrong architectural shape (multi-beam approximation of a dual-motor scanner) but at least the total ray count (2,304) is in the same ballpark as the real 1,964/turn. See §3 implication subsection. |

**Bottom line:** as of 2026-05-02, no public sim — Unitree-authored or
community — actually models the L1's single-beam dual-motor rosette. The
closest thing is our own regular-grid approximation, which has correct
bulk ray budget but wrong density distribution.

[pgtt]: https://github.com/NtagkasAlex/phase_guided_terrain_traversal

[um-mj]: https://github.com/unitreerobotics/unitree_mujoco
[um-sil]: https://github.com/unitreerobotics/unitree_sim_isaaclab
[um-rllab]: https://github.com/unitreerobotics/unitree_rl_lab
[um-rlmj]: https://github.com/unitreerobotics/unitree_rl_mjlab
[go2omni]: https://github.com/abizovnuralem/go2_omniverse

#### Implication for the current `velocity_lidar_env_cfg.py`

`L1_LIDAR_PATTERN = patterns.LidarPatternCfg(channels=32, vfov=(0,90), hfov=(-180,180), hres=5.0)`
emits a regular 32 × 72 = 2,304-ray grid per sweep — total point count is in
the right ballpark (vs. ~1,964 real), but the **distribution is wrong**:

- Real L1 oversamples the band near the centre of the FOV (where the rosette
  is dense); regular-grid sim spreads rays evenly over solid angle.
- Real L1 undersamples the zenith and edges; regular-grid sim oversamples
  them.
- For close-range (<1 m) foothold work, the dense band is exactly the
  forward-look region the policy needs to read — so the regular-grid
  approximation is plausibly *worse* than the real sensor close in, not
  better. Sim-to-real distribution shift in the wrong direction.

Mitigations, in increasing order of effort:
1. Re-derive `(channels, hres)` from the real ray budget so total ≈ 1,964/turn
   instead of 2,304, and bias the elevation distribution toward the high-
   density band rather than uniform vfov.
2. Replace `LidarPatternCfg` with a custom pattern that emits the actual
   dual-motor rosette (`LidarPatternCfg` is a regular-grid abstraction; a
   `from-functions`-style pattern is needed for the true sweep geometry).
3. ~~Use `Unitree_L1.json` from `abizovnuralem/go2_omniverse` as a parameter
   source for option 2.~~ **Withdrawn:** that file is a relabelled Ouster OS0
   preset, not an L1 model — see §5c. There is no public L1-accurate sim
   asset; option 2 would require reverse-engineering the sweep pattern from
   real L1 captures.

### Files to modify in `unitree_rl_lab`

The Go2 task is one self-contained config file:
`~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`

Three changes, ~30–60 LOC total:

1. **Replace the height-scanner sensor (lines 96–101 of that file)** —
   currently:
   ```python
   height_scanner = RayCasterCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base",
       offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
       ray_alignment="yaw",
       pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
       mesh_prim_paths=["/World/ground"],
   )
   ```
   Add (don't replace — keep the height scan for the critic) a separate
   `lidar_scanner = RayCasterCfg(...)` with `pattern_cfg=patterns.LidarPatternCfg(channels=...)`
   mounted at the actual L1 mount-point on the Go2 (~base, slightly forward,
   ~10 cm above the body — verify against Unitree's URDF).

2. **Re-enable richer terrains in `COBBLESTONE_ROAD_CFG` (lines 23–67)** — the
   current config has the rough subterrains commented out (`random_rough`,
   `pyramid_stairs`, `boxes`, slopes). Uncomment them so the policy actually
   sees rough terrain to navigate.

3. **Add a LiDAR observation term to `PolicyCfg` (around line 198)** — e.g.
   ```python
   lidar = ObsTerm(
       func=mdp.height_scan,   # or a custom obs func returning raw distances
       params={"sensor_cfg": SceneEntityCfg("lidar_scanner")},
       clip=(-1.0, 30.0),
   )
   ```
   Source for `mdp.height_scan` and the obs-term machinery: same file lines 178–204.

**Caveat — TiledCamera Blackwell hang (issue #4951):** raycasters are *not*
TiledCameras and are unaffected. Only relevant if you later switch to RGBD/
depth-camera input. Source: [IsaacLab issue #4951][src-tc].

[src-tc]: https://github.com/isaac-sim/IsaacLab/issues/4951

---

## 4. Recommendation (revised)

Path B was demoted after verification (legged-loco is proprio-only, not a
LiDAR recipe). The viable paths are:

| Path | Effort | What you get |
|---|---|---|
| **A. DIY in `unitree_rl_lab`** | ~½ day | Go2 + L1-pattern raycast + rough terrain on the stack we already have. Smallest delta, but you inherit `unitree_rl_lab`'s recipe (Unitree-tuned for sim-to-real) with no community battle-testing of the LiDAR addition. |
| **A′. DIY in `fan-ziqi/robot_lab`** | ~1 day | Same delta as A but on a 1.7k-★ codebase that's actively maintained against Isaac Lab v2.3.2. Slight install overhead (clone + extension install) but better long-term base. |
| **C. Adapt `leggedrobotics/fdm`** | 2–4 days | RSS 2025 engineering, elevation-map abstraction (sensor-swappable for sim-to-real). Cost: ANYmal → Go2 USD swap + LiDAR raycaster wiring + 2.1.1 → 2.3.2 minor port. |

In all paths, mine `abizovnuralem/go2_omniverse`'s `Unitree_L1.json` for the
LiDAR ray-pattern parameters and translate to `LidarPatternCfg`.

My revised recommendation: **A′ (robot_lab)** for fastest iteration with
better community footing than A; **C (fdm)** if sim-to-real on real Go2
hardware is the actual end goal — the elevation-map abstraction is the
single biggest sim-to-real win in this whole space.

---

## 5b. The "closest to Unitree" answer (deep synthesis)

After a deeper pass through Unitree's issue tracker, forks, and deprecated
asset repos, the picture sharpens significantly:

### Evidence that the community wants this and Unitree hasn't shipped it

- [`unitree_rl_lab` issue #45][i45] (Sep 2 2025, *closed without resolution*):
  *"Could you kindly share some insights or experiences on integrating visual
  inputs, such as **LiDAR data or depth images**, into the Go2 or G1
  workflows?"* — no maintainer response visible.
- [`unitree_rl_lab` issue #67][i67] (open): *"how should I configure
  RayCasterCfg in the simulation?"* — no maintainer response.
- [`unitree_rl_lab` issue #88][i88] (Nov 21 2025, open): another RayCasterCfg
  configuration problem, unresolved.
- **Zero PRs** in `unitree_rl_lab` matching `lidar`, `perception`, `vision`,
  `raycast`, `sensor`, or `point cloud`.
- **Zero notable forks** of `unitree_rl_lab` adding LiDAR or perception.
- [`unitree_model`][um] (the canonical Go2 USD asset repo) is
  **deprecated** — README points users to **Hugging Face** for future Unitree
  asset releases.

[i45]: https://github.com/unitreerobotics/unitree_rl_lab/issues/45
[i67]: https://github.com/unitreerobotics/unitree_rl_lab/issues/67
[i88]: https://github.com/unitreerobotics/unitree_rl_lab/issues/88
[um]: https://github.com/unitreerobotics/unitree_model

So "closest to Unitree" doesn't mean "find the existing Unitree solution" —
it means **author the solution inside `unitree_rl_lab`** and align with their
conventions. There is no existing Unitree solution.

### One additional credible artifact: `Zhefan-Xu/isaac-go2-ros2`

[`Zhefan-Xu/isaac-go2-ros2`][igr2] (518 ★, Isaac Sim 4.5 + Isaac Lab 2.1)
**publishes a `/unitree_go2/lidar/point_cloud` ROS2 topic** for the simulated
Go2 — the only Go2 LiDAR simulation we found that's actively maintained and
wired to a usable interface. Integrates with the **NavRL** RL agent for
navigation. It's a *deployment* platform (ROS2 in/out), not a training
recipe, and not pinned to our stack — but it's the cleanest reference for
"how to model a Go2 LiDAR in Isaac Sim" if you need a sanity check for sensor
placement and topic schema.

[igr2]: https://github.com/Zhefan-Xu/isaac-go2-ros2

### The synthesized recommendation

**Build inside `unitree_rl_lab`** (= maximally close to Unitree, can be
PR'd back to close issues #45/#67/#88), composed of four ingredients each
sourced from the strongest available reference:

| Ingredient | Source | Why this source |
|---|---|---|
| Base task config (Go2 velocity env, MDP rewards, terrain framework, USD asset, joint setup) | `unitree_rl_lab/.../go2/velocity_env_cfg.py` (Unitree) | Canonical Unitree-tuned recipe; sim-to-real-ready against their SDK |
| Rough sub-terrains | Already present in `COBBLESTONE_ROAD_CFG` of the same file, **just commented out** (random_rough, slopes, boxes, pyramid_stairs) | Unitree wrote them; uncommenting is the lowest-friction path |
| LiDAR sensor primitive | `LidarPatternCfg` in Isaac Lab `~/IsaacLab/.../patterns_cfg.py` | NVIDIA-supported, raycaster-based (no Blackwell TiledCamera issue) |
| Real L1 ray-pattern parameters (FOV, scan rates, range, IMU) | [Unitree L1 User Manual v1.1, 2024.06][l1-manual] (primary source, see §3) | Verified 2026-05-02. The previous `Unitree_L1.json` recommendation was wrong — that file is a relabelled Ouster OS0 preset, see §5c. No L1-accurate sim asset is publicly available; the manual gives bulk specs only, not the per-sweep ray pattern. |
| Sensor → policy projection (turn ~10k raw distances into something an MLP can ingest) | Elevation-map abstraction from [Roth et al., **RSS 2025** (`leggedrobotics/fdm`)][fdm] | The current academic SOTA for sim-to-real perceptive locomotion; sensor-agnostic |

[fdm]: https://github.com/leggedrobotics/fdm

This is the intersection of:
- **Maximally Unitree-proximate** — lives in their repo, uses their asset, follows their conventions, closes their open issues
- **Highest-quality engineering** — Isaac Lab raycaster + RSS 2025 elevation-map abstraction
- **Sim-to-real-aligned** — elevation map decouples policy from sensor specifics, so the same policy can run on real Go2 with the actual L1 (or any later Unitree LiDAR like the L2)
- **Buildable on the stack we already installed** — zero new external deps

### Concrete delta against the current `unitree_rl_lab/.../go2/velocity_env_cfg.py`

1. **Uncomment lines ~28–67** to re-enable `random_rough`, `pyramid_stairs`,
   `boxes`, slopes in `COBBLESTONE_ROAD_CFG`. (Already written by Unitree.)
2. **Add an `lidar_scanner = RayCasterCfg(...)`** alongside the existing
   `height_scanner`, with `pattern_cfg=patterns.LidarPatternCfg(channels=N,
   vertical_fov_range=..., horizontal_fov_range=(-180,180), horizontal_res=...)`,
   parameters lifted from `Unitree_L1.json`. Mount offset from the Go2's
   actual L1 dock position (verify against the Hugging Face USD or Unitree's
   URDF).
3. **Implement an MDP obs function** that projects the LiDAR distances onto a
   robot-centered 2.5D elevation grid (the `fdm` pattern) instead of feeding
   raw distances to the MLP. Add it under `mdp/` next to the existing
   `height_scan` function.
4. **Add the projected elevation map as a term in `PolicyCfg`** with
   appropriate scale/clip/noise.
5. **Tune** — initial run will likely need adjustments to `feet_air_time`
   weight, terrain curriculum bounds, and obs scales.

Realistic effort: 1–3 days for a working policy, +1–2 weeks for sim-to-real
hyperparameter sweeps. Ship as a PR to `unitree_rl_lab`.

### Why not `fan-ziqi/robot_lab` (revising A′)

`robot_lab` is excellent infrastructure (1.7k ★, exact stack match), but
crucially: it **diverges from Unitree's task naming and asset paths** —
`RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0` vs Unitree's
`Unitree-Go2-Velocity`, separate `source/robot_lab/assets/unitree.py` vs
Unitree's `unitree_rl_lab.assets.robots.unitree`. Building inside it gives
you better community footing but makes the result **less PR-able back to
Unitree** and **slightly less sim-to-real-aligned** (Unitree's SDK assumes
their own task/observation conventions). For "closest to Unitree", the
direct path through `unitree_rl_lab` wins.

`robot_lab` is still the right second choice if Unitree's repo proves too
unstable or if you want a drop-in 2.3.2-compatible base.

---

## 5c. Fidelity gaps & required augmentations

The synthesized recipe in §5b is a **kinematic/geometric** representation of
the real Go2. The body and terrain geometry are sim-to-real-grade. The LiDAR
is not. This section enumerates every gap I'm aware of so they can be either
addressed in the implementation or knowingly deferred.

### Accurate enough for sim-to-real (no augmentation needed)

| Component | Status |
|---|---|
| Go2 mass / inertia / joint limits / torques (from `UNITREE_GO2_CFG`) | ✓ Unitree-published, validated on their own hardware |
| Action scaling (0.25) + decimation (4 → 50 Hz policy on 200 Hz sim) | ✓ Matches Unitree's deployed sim-to-real recipe |
| Reward shaping (track_lin_vel_xy, feet_air_time, energy, joint_pos_limits, etc.) | ✓ Same recipe Unitree uses for shipped policies |
| Terrain *geometry* (random_rough, slopes, boxes, stairs in the commented `COBBLESTONE_ROAD_CFG`) | ✓ Realistic for what a LiDAR sees |
| Friction / restitution / mass / push-velocity domain randomization | ✓ Already in `EventCfg` |

### LiDAR abstractions that will bite on real hardware

What the **real Unitree L1** has vs. what `LidarPatternCfg` simulates:

| Phenomenon | Real L1 | Our raycast | Augmentation needed |
|---|---|---|---|
| Range noise | Gaussian + range-dependent σ | Zero noise | Add per-ray Gaussian noise (σ ≈ 2–5 cm at typical ranges) |
| Reflectance / dropouts | Black/wet/specular surfaces miss returns | Always hits if ray meets mesh | Random per-ray dropout (~5–15%) at training time |
| Motion distortion | Mirror sweep during base motion | Instantaneous snapshot | Either accept the gap or compose rays from base poses across the sweep window |
| Update rate | ~10–15 Hz | Whatever the sim ticks at | Set `update_period = 1/12 s` on the `RayCasterCfg`, hold last frame between ticks |
| Intensity / reflectance | 4D (3D + I) | 1D (range only) | Not modeled — the elevation-map projection abstracts away intensity anyway |
| Multi-path / specular returns | Occasional spurious hits | None | Acceptable to ignore for a first pass |
| Mount-point uncertainty | Real L1 mounted with mm/° calibration error | Perfect transform | Randomize sensor offset (±1 cm position, ±2° rotation) at `reset` |

### Other realism gaps in the base task (not LiDAR-specific)

| Gap | Status | Plan |
|---|---|---|
| IMU bias drift | Not modeled | Acceptable for first run; common to add later |
| Motor backlash / temperature drift | Not modeled | Acceptable for first run |
| Material variation (grass / mud / wet pavement) | Friction randomized, but no compliance/deformability | Acceptable; the L1 sees geometry, not material |
| Real-world terrain (foliage, glass, water) | Mesh-based — all "solid" to the raycaster | Adds a known sim-to-real gap; mitigate with random dropouts |

### `Unitree_L1.json` is mislabelled (correction)

Bulk L1 specs (FOV, sampling rate, scan frequencies, range, IMU rate) are
now verified against the [Unitree L1 User Manual v1.1, 2024.06][l1-manual]
— see §3 "Reference: Unitree L1 4D LiDAR specs" above for verbatim values
and page citations.

**Earlier passes of this doc recommended mining
`abizovnuralem/go2_omniverse/Isaac_sim/Unitree/Unitree_L1.json` for the L1
ray-pattern parameters. That recommendation is wrong** — verified by
fetching the file 2026-05-02:

```json
"name": "OS0 REV7 128 10hz @ 1024 resolution",
"comment3": "ouster OS0 REV7 128 channels @ 10Hz 1024 horizontal resolution",
"comment1": "parameters obtained from https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p0-os0.pdf"
```

The file is an Ouster OS0 datasheet preset (128 fixed `numberOfEmitters`,
`farRangeM: 75.0`, `scanRateBaseHz: 100.0`, ±45° `elevationDeg` array,
`wavelengthNm: 865`) merely renamed to `Unitree_L1.json`. None of the L1's
actual specs (single beam, 11 Hz azimuth, 180 Hz vertical sweep, 30 m
range, 21,600 pts/s, hemispherical 360°×90°) appear anywhere in the JSON.
**Do not use as an L1 parameter source.**

**Outstanding gap:** no public sim asset encodes the L1's actual single-
beam dual-motor scan trajectory. The manual gives bulk frequencies but
not the closed-form rosette pattern. To build a faithful L1 sim, the
options are: (1) reverse-engineer the rosette from real L1 captures via
the SDK's `ScanUnitree::points[120]` packets, or (2) accept the
distribution shift and use a regular-grid `LidarPatternCfg` with the
right total point budget (~1,964 rays/turn) and accept that
forward-looking density will be lower than the real device.

### Mount-point caveat

The Unitree-published Go2 USD (now hosted on Hugging Face since
`unitree_model` was deprecated) likely does **not** include the L1 as a
named prim. The implementation will add the LiDAR via an explicit
`OffsetCfg(pos=..., rot=...)` transform on the `RayCasterCfg`. The transform
parameters need to come from the Go2 EDU/PRO mounting plate dimensions —
also unverified at this revision; treat as a calibration parameter to
randomize during training.

### Elevation-map projection caveat (sim-to-real)

If the real-world deployment uses
[`elevation_mapping_cupy`](https://github.com/leggedrobotics/elevation_mapping_cupy)
(ETH RSL, the de-facto standard) on the robot, the in-sim projection should
**match its algorithm** — recursive Bayesian filter on a robot-centered XY
grid with motion compensation — rather than a naïve per-frame projection.
Otherwise the input distribution shifts between training and deployment.
For the first training run, a naïve projection is acceptable; matching the
real-world algorithm is a known follow-up.

### Augmentation budget

These augmentations are an additive **~50–150 LOC** on top of the §5b
recipe (per-ray noise, per-ray dropout, mount-offset randomization, update
rate gating). No new external dependencies — they're `torch.randn` /
`torch.bernoulli` calls inside the obs function. They should be applied
**only on the policy obs** (LiDAR), not on the privileged critic obs
(which can keep clean ranges as ground truth).

---

## 5d. Sim-to-real philosophy: envelope coverage, not fidelity

**Decision (2026-05-02):** the sim-side L1 work is *not* aimed at faithful
reproduction of the device — it is aimed at sim-to-real transfer. These
are different goals and the doc should not conflate them.

### Why this reframe

§3 establishes that the real L1 is a single-beam dual-motor scanner
producing a non-uniform rosette, and that **no public sim asset
reproduces it** (§5c, "Community sim attempts"). Reverse-engineering the
rosette from real device captures is technically possible but expensive,
and our existing `LidarPatternCfg(channels=32, hres=5°)` regular grid is
the wrong architectural shape (§3 implication subsection).

A faithful-sim mindset says "the regular grid is wrong, fix it." A
sim-to-real mindset says **"make the policy robust to a wide enough
envelope that whatever the real L1 produces is inside it."** The second
goal is what we actually need.

### Implication: the regular-grid sim is fine as a starting shape

The work moves from "match the L1's geometry" to "widen the
randomisation envelope until the real device's behaviour is covered."
The structural-shift worry — that the sim systematically over-samples at
some elevations and the policy learns to rely on that — dissolves if we
randomise across density distributions per episode, because the policy
never gets to memorise any single one.

### Concrete domain-randomisation envelope

Per-episode (or per-reset) sampling, layered on top of the existing
`lidar_distances` obs function in
`tasks/locomotion/mdp/observations.py`:

| Axis | Range / set | Source of envelope width |
|---|---|---|
| Total ray count per turn | uniform in [1,500, 2,500] | manual: ~1,964; ±20% covers count uncertainty |
| Elevation density bias | sample one of {uniform, top-biased, bottom-biased, centre-biased} | covers the manual's "denser near centre" + structural-shift unknowns |
| Per-ray azimuth jitter | Gaussian σ ≈ 0.2° | unmodeled mechanical jitter of the slow motor |
| Per-ray elevation jitter | Gaussian σ ≈ 0.3° | unmodeled jitter of the fast reflector |
| Per-ray range noise | Gaussian σ uniform in [1, 5] cm | manual gives no range-σ spec; bracket plausible values |
| Per-ray dropout | Bernoulli p uniform in [0%, 15%] | reflectance / specular / wet / glass returns |
| Mount offset | ±1 cm position, ±2° rotation | mm/° calibration error on real hardware |
| Update period | 1/11 s × uniform [0.85, 1.20] | covers azimuth-rate jitter + IMU-vs-cloud-stamp drift |
| Range clip | [0.05, 30] m | manual; held fixed |

Density-bias sampling is the load-bearing piece. As long as
`{uniform, top, bottom, centre}` is the envelope, the real centre-biased
rosette is by construction inside it, and the policy cannot lean on any
single distribution.

### What this is *not*

- Not a claim that the sim looks like the L1. It doesn't.
- Not a substitute for an on-robot test. The real device is the only
  honest validator; the DR envelope is the bet, the on-robot test is
  the settlement.
- Not a reason to skip sim-to-real best practice (privileged critic,
  observation noise schedules, asymmetric obs) — those still apply.

### Falsification

The bet fails if the on-robot evaluation shows the policy degrading in a
way that traces back to a *sensor distribution* the DR didn't cover
(e.g. a real-world dropout mode beyond 15%, or a density bias outside
the four sampled distributions). Mitigation in that case: widen the
relevant axis and retrain, *not* try to faithfully model the L1.
Faithful modelling is only justified if widening DR doesn't close the
gap — and historically (Tobin 2017, OpenAI 2019, Lee 2020) it usually
does.

---

## 5. Audit log — corrections applied during verification

The first draft of this document was assembled from a research-agent summary.
After re-verification against primary sources (each repo's actual README via
WebFetch), the following claims were corrected:

| Original claim | Status after verification | Action |
|---|---|---|
| NaVILA-Bench reports "88% sim-to-real success" | Not in README. Source unverifiable. | Removed. |
| `legged-loco` "trains a Go2 policy with simulated LiDAR-derived input" | README has no LiDAR / RayCaster / perception references; policy is proprio-only. | Demoted from Top-3 to reference only. |
| `robot_lab` star count vague | Verified: 1.7k ★, 195 commits, 190 forks; release v2.3.2 dated Feb 3 2026. | Numbers added; promoted in recommendation. |
| `go2_omniverse` "teleop demo, no RL training" | Roadmap lists PPO RL training as complete and ships `Unitree_L1.json` LiDAR config. | Re-classified as a useful artifact (sensor JSON), not a viable training base (Orbit 0.3.0 is too old). |
| Unitree L1 specs (360°×90°, 21,600 pts/s, 0.18°, 0.05–30 m) | `shop.unitree.com/products/unitree-4d-lidar-l1` returned 404; specs unverifiable from official source at original revision. | **Resolved (2026-05-02):** verified against [Unitree L1 User Manual v1.1, 2024.06][l1-manual] (oss-global-cdn.unitree.com PDF). Verbatim values now in §3. Architecture corrected: L1 is single-beam dual-motor (180 Hz vertical / 11 Hz azimuth), not multi-beam — the regular-grid `LidarPatternCfg` in `velocity_lidar_env_cfg.py` is a wrong-shape approximation, see §3. |
| Path B effort estimate (`NaVILA-Bench` port) "3–5 days, gives Go2-with-LiDAR" | Path B doesn't actually give you LiDAR (legged-loco is proprio-only). | Path B removed from the recommendation table. |

**Open items** (acknowledged unknowns, not yet resolved):

- The `½ day` (Path A) and `1 day` (Path A′) effort estimates assume a working
  obs-function exists for raw LiDAR distances. In practice an MLP policy can't
  ingest ~10k raw distances directly — you need either a downsampling step or
  an elevation-map projection front-end. Realistic effort is closer to **50–
  200 LOC + a new MDP obs function + non-trivial RL hyperparameter tuning**.
  This was understated in the original draft.
- `fdm` is pinned to Isaac Lab 2.1.1; the 2.1 → 2.3 port may surface its own
  Blackwell / sm_120 concerns separate from the ones we already handled in
  the main install. Verify with the same `torch.cuda.get_arch_list()` smoke
  test after any environment change.
- The Unitree L1 hardware exists on Go2 EDU / PRO variants but **not on every
  Go2 SKU**. Before committing to L1 specs, confirm which variant the target
  hardware actually is.

---

## Source index

### Unitree
- [unitreerobotics organization repo list](https://github.com/orgs/unitreerobotics/repositories)
- All Unitree repo READMEs cited in §1
- [Unitree L1 4D LiDAR product page](https://shop.unitree.com/products/unitree-4d-lidar-l1)
- [Unitree 4D LiDAR-L1 User Manual v1.1, 2024.06 (PDF)](https://oss-global-cdn.unitree.com/static/52b72f707b304d229d4321eea223738f.pdf) — **primary source for all L1 specs cited in §3**
- [Unitree LiDAR product overview](https://www.unitree.com/LiDAR/)
- [`unitreerobotics/unilidar_sdk`](https://github.com/unitreerobotics/unilidar_sdk) — primary source for the L1 wire-protocol data structures cited in §3
- [`unitreerobotics/unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco) — confirms no L1 sim model
- [`unitreerobotics/unitree_sim_isaaclab`](https://github.com/unitreerobotics/unitree_sim_isaaclab) — confirms no L1 sim model

### NVIDIA
- Local: `~/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns_cfg.py`
- [Isaac Lab Ray Caster docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.RayCasterCfg)
- [Isaac Lab v2.3.2 release tag](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.2)
- [Isaac Lab issue #4951 — TiledCamera Blackwell hang](https://github.com/isaac-sim/IsaacLab/issues/4951)

### Community / academic
- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)
- [leggedrobotics/fdm](https://github.com/leggedrobotics/fdm) — Roth et al., RSS 2025
- [Zhefan-Xu/isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2) — Go2 LiDAR sim + ROS2 + NavRL integration
- [yang-zj1026/NaVILA-Bench](https://github.com/yang-zj1026/NaVILA-Bench)
- [yang-zj1026/legged-loco](https://github.com/yang-zj1026/legged-loco)
- [CAI23sbP/Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour)
- [abizovnuralem/go2_omniverse](https://github.com/abizovnuralem/go2_omniverse)
- [NtagkasAlex/phase_guided_terrain_traversal](https://github.com/NtagkasAlex/phase_guided_terrain_traversal)
- [LeCAR-Lab/ABS](https://github.com/LeCAR-Lab/ABS)
- [chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour)
- [ZiwenZhuang/parkour](https://github.com/ZiwenZhuang/parkour)
- [leggedrobotics/viplanner](https://github.com/leggedrobotics/viplanner)
- [leggedrobotics/navigation_template (archived)](https://github.com/leggedrobotics/navigation_template)
- [Charlescai123/isaac-wild-go2](https://github.com/Charlescai123/isaac-wild-go2)
- [arxiv 2511.04831](https://arxiv.org/abs/2511.04831) — *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning* (canonical Isaac Lab paper, Nov 2025)
