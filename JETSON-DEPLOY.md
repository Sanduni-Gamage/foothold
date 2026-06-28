# Jetson Orin Deployment Guide — Unitree-Go2-Footholds policy

What a coder needs to know to run a policy trained in this repo on a Go2's
on-board Jetson Orin. Written for the foothold task; the velocity-only and
lidar variants share the same plumbing with smaller obs vectors.

This doc assumes the deployer is comfortable with C++/Python, ROS-ish robot
software, and has a Go2 to test on. None of this is theoretical — Unitree
ships a working C++ deployment skeleton in this exact repo.

---

## 1. What gets deployed

Three artefacts:

| Artefact | Where it lives | Format |
|---|---|---|
| Policy weights (the actual neural net) | `~/projects/doga/logs/rsl_rl/unitree_go2_footholds/<run-ts>/exported/policy.pt` and `policy.onnx` | TorchScript JIT and ONNX |
| Joint SDK ordering | `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` → `UNITREE_GO2_CFG.joint_sdk_names` | Python list, copy verbatim |
| Actuator gains | same file → `UnitreeActuatorCfg_Go2HV(stiffness=25.0, damping=0.5, friction=0.01)` | Apply to PD on real motors |

The exported `.pt` and `.onnx` are produced **automatically** by `rsl_rl`'s
`play.py` when you load any checkpoint. They drop into a `exported/`
subdirectory of the run log dir. If you don't see them, run `play.py` once
against the chosen `model_X.pt` to materialize them.

```bash
CUDA_VISIBLE_DEVICES=1 python ~/unitree_rl_lab/scripts/rsl_rl/play.py \
  --task Unitree-Go2-Footholds --num_envs 1 --headless \
  --checkpoint <path-to-model_X.pt>
# → produces .../exported/policy.pt and .../exported/policy.onnx
```

The Jetson runs ONNX (via the bundled ONNX Runtime — also in this repo at
`~/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/`).

---

## 2. Existing reference deployment (use this, don't re-invent)

Unitree ships a complete C++ deployment skeleton in
`~/unitree_rl_lab/deploy/`:

- `deploy/robots/go2/main.cpp` — entry point, Unitree DDS init, FSM start
- `deploy/include/FSM/` — finite state machine: `Passive`, `FixStand`,
  `RLBase` (the RL policy state)
- `deploy/include/unitree_articulation.h` — joint state ↔ obs vector packing
- `deploy/include/param.h` — YAML loader for runtime config
- `deploy/thirdparty/onnxruntime-linux-x64-1.22.0/` — ONNX runtime bundled

Existing `State_RLBase` does:

1. Read sensor state from Unitree DDS (`LowState_t`)
2. Pack obs vector in the order Isaac Lab expects (see §4)
3. Run ONNX policy inference
4. Unpack 12-dim action → joint position targets via PD
5. Write `LowCmd_t` back to robot via DDS

For the foothold task, the existing skeleton **needs additions** for:
- `nearest_footholds` obs source (8 dims)
- `foot_positions` obs source (12 dims via FK)
- Foothold planner subscription (whatever produces footholds)

Build:

```bash
cd ~/unitree_rl_lab/deploy
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
# → produces deploy/build/robots/go2/go2
```

Cross-compile for Jetson aarch64 if building on x86; native build on the
Jetson itself is simpler.

---

## 3. Hardware / sensor sources on Go2

Map every obs term to its actual on-Jetson source:

| Obs term (dim) | Source on Go2 |
|---|---|
| `base_ang_vel` (3) | IMU gyro via DDS `LowState_t.imu_state.gyroscope[0..2]` |
| `projected_gravity` (3) | IMU quaternion → rotate (0,0,−1) into body frame. `LowState_t.imu_state.quaternion` |
| `velocity_commands` (3) | Joystick or external publisher (`vx, vy, ωz`). DSL keyboard helper at `deploy/include/unitree_joystick_dsl.hpp` |
| `joint_pos_rel` (12) | `LowState_t.motor_state[i].q` minus `default_pos[i]`. Order via `joint_sdk_names`. |
| `joint_vel_rel` (12) | `LowState_t.motor_state[i].dq` minus `default_vel[i]`. Same order. |
| `last_action` (12) | Internal — the policy's previous output. |
| `nearest_footholds` (8 = 4×2) | **External foothold planner** (see §6). Robot-frame xy of next 4 unclaimed stones. |
| `foot_positions` (12 = 4×3) | **Forward kinematics** on joint state. Use `pinocchio` or hand-rolled FK. Robot-frame xyz. |

**Joint SDK ordering — DO NOT GET WRONG**: from `assets/robots/unitree.py`:

```python
joint_sdk_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
```

Isaac Lab internally uses **alphabetical** joint ordering, but
`UnitreeArticulationCfg.joint_sdk_names` defines the SDK-side ordering and
the deploy-time C++ does the swap. Get this wrong and you'll see twitchy
limbs swapping randomly — it's the #1 cause of Day-1 deploy failures.

Default joint positions (used to compute `joint_pos_rel`):

```python
joint_pos = {
    ".*R_hip_joint": -0.1, ".*L_hip_joint": 0.1,
    "F[L,R]_thigh_joint": 0.8, "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
}
```

In SDK order: `[-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]`.

---

## 4. Obs vector layout (in the EXACT order the policy expects)

Isaac Lab concatenates `policy` ObsGroup terms in **declaration order**
inside `FootholdObsCfg.PolicyCfg`. For the foothold task this is:

```
[base_ang_vel(3),
 projected_gravity(3),
 velocity_commands(3),
 joint_pos_rel(12),
 joint_vel_rel(12),
 last_action(12),
 nearest_footholds(8),
 foot_positions(12)]   →  total = 65 dims
```

Pre-policy scaling and noise (applied at training time, must NOT be applied
at deploy because the sensor data is already real):

| Term | Train-time scale | Train-time noise (Unoise) | Clip |
|---|---|---|---|
| `base_ang_vel` | × 0.2 | ±0.2 | (−100, 100) |
| `projected_gravity` | — | ±0.05 | (−100, 100) |
| `velocity_commands` | — | — | (−100, 100) |
| `joint_pos_rel` | — | ±0.01 | (−100, 100) |
| `joint_vel_rel` | × 0.05 | ±1.5 | (−100, 100) |
| `last_action` | — | — | (−100, 100) |
| `nearest_footholds` | — | — | (−15, 15) |
| `foot_positions` | — | — | (−2, 2) |

**At deploy: apply the *scales* but NOT the noise.** Noise was a
training-time domain-randomization signal. The scales are part of the
input normalization the policy learned with.

Output: 12-dim action. Decode as joint position targets:

```cpp
target_q[i] = default_pos[i] + action[i] * 0.25;   // action_scale = 0.25
torque[i]   = stiffness * (target_q[i] - q[i]) + damping * (0 - dq[i]);  // PD
```

`stiffness=25.0`, `damping=0.5` from the actuator cfg. Friction `0.01` is
applied internally to the actuator model — don't re-add at deploy.

Policy runs at **50 Hz** (decimation=4 of 200 Hz physics; deploy-side just
runs the policy at 50 Hz). Motor command can be issued at higher rate by
re-applying the same target_q until the next policy step.

---

## 5. Python skeleton (for prototyping, not production)

If you want a quick Python prototype before committing to the C++ pipeline:

```python
import torch, numpy as np
import onnxruntime as ort

policy = ort.InferenceSession(
    "exported/policy.onnx",
    providers=["CUDAExecutionProvider"],  # Jetson Orin uses CUDA EP
)

# Cached state
last_action = np.zeros(12, dtype=np.float32)
default_q   = np.array([-0.1, 0.8, -1.5,  0.1, 0.8, -1.5,
                        -0.1, 1.0, -1.5,  0.1, 1.0, -1.5], dtype=np.float32)

def step(low_state, vx, vy, wz, footholds_xy_4x2, foot_xyz_4x3):
    # Read sensors (numpy on host or via Jetson sensor driver)
    base_ang_vel = np.array(low_state.imu.gyro)               # (3,)
    proj_grav    = quat_rotate_inv(low_state.imu.q, [0, 0, -1])  # (3,)
    cmd          = np.array([vx, vy, wz])                     # (3,)
    q  = sdk_reorder(low_state.motor_q)                       # (12,)
    dq = sdk_reorder(low_state.motor_dq)                      # (12,)

    # Build obs in declared order
    obs = np.concatenate([
        base_ang_vel * 0.2,
        proj_grav,
        cmd,
        (q - default_q),
        (dq * 0.05),
        last_action,
        footholds_xy_4x2.flatten(),
        foot_xyz_4x3.flatten(),
    ]).astype(np.float32).reshape(1, -1)  # (1, 65)

    # Inference
    action = policy.run(None, {"obs": obs})[0][0]  # (12,)

    # Decode
    target_q = default_q + action * 0.25
    return target_q
```

Production: do this in C++ with ONNX Runtime to hit the 50 Hz cadence with
margin. The bundled `onnxruntime-linux-x64-1.22.0/` is x86; for Jetson aarch64
download the matching ARM64 build from
`https://github.com/microsoft/onnxruntime/releases` and replace the
`thirdparty/` folder. Or use TensorRT for lower latency (see §7).

---

## 6. The foothold-planner gap (READ THIS)

The trained policy expects `nearest_footholds` to come from somewhere. **At
training time it's perfect ground truth.** At deploy, the Jetson must
produce 4 nearest unclaimed foothold xy positions in robot frame, every 20 ms.

Options the deployer must choose:

| Option | What it costs |
|---|---|
| **(a) Build a foothold planner from L1 lidar** | L1 → SLAM → elevation map → footstep selection. Substantial — ETH RSL's TAMOLS is internal. Open: `elevation_mapping_cupy` (RSL, on GitHub) for the map; the foothold-selection layer is custom. ~2–4 weeks of work. |
| **(b) Use a hand-defined foothold sequence** | Mocap or manual waypoint placement. OK for benchmarks/demos, useless in the wild. |
| **(c) Have a high-level human emit footholds via teleop** | Joystick → mouse → foothold targets relayed at the policy rate. Useful for testing. |
| **(d) Skip the foothold task at deploy and run the velocity-only policy instead** | Use `model_25600.pt` from the Stage-2 rough-terrain run; no foothold input needed. Loses the foothold capability but is deploy-ready as-is. |

Until (a) exists in your stack, the practical deploy story is (d) for
locomotion + (c) for foothold-aware navigation.

The training also did **not** simulate planner staleness, latency, or noise
(see HIGH-LEVEL-RESEARCH.md §15b). When you do build the planner, expect to
go back and re-train the foothold policy with domain-randomization on those
factors. ~30 LOC change in `mdp/observations.py`.

---

## 7. Performance / TensorRT

For the foothold task the actor MLP is `[65 → 512 → 256 → 128 → 12]`,
~150k parameters, ~600 KB FP32. At 50 Hz this is sub-millisecond on Orin
even on CPU. ONNX Runtime + CUDA EP is plenty.

If you want lower latency or to share the GPU with the perception pipeline:

```bash
trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16
```

Load with `tensorrt` Python or C++. Expect ~3× speedup, but already so fast
it's unlikely to matter — the Jetson budget will go to perception, not
policy inference.

---

## 8. Sim-to-real gotchas (verified pitfalls)

1. **Joint ordering** — covered in §3. Worth repeating because it bites
   everyone exactly once.
2. **Default joint positions** — must match `init_state.joint_pos` from
   training cfg exactly. Subtract these from raw `q` to get `joint_pos_rel`.
3. **Action scale** — `0.25`. Train-time `action_scale = 0.25`. Apply on
   deploy. Don't double-apply.
4. **Use_default_offset** — `True`. Means `target_q = default_q + action *
   scale`, not `target_q = action * scale`. Easy to get wrong.
5. **Body order in `foot_positions`** — must match the order Isaac Lab
   used. Check `body_names=".*_foot"` regex match order on the deployed
   URDF; verify by comparing shapes.
6. **Quaternion convention** — Isaac Lab is `(w, x, y, z)` order in
   `root_quat_w`. Unitree DDS may use `(x, y, z, w)`. Fix at the boundary.
7. **`projected_gravity`** — not the raw accelerometer reading. It's the
   gravity vector rotated INTO the body frame. Compute via
   `quat_rotate_inverse(base_quat, [0, 0, -1])`. Train-time it's an exact
   sim signal; deploy-time it requires a stable orientation estimate (the
   Go2's onboard EKF gives this).
8. **No `last_action` initialization at startup** — start with zeros
   (matches reset behaviour in sim).
9. **Foot kinematics** — be sure to use the URDF that matches the trained
   policy's USD. They both come from `unitree_model/Go2/`. Mismatched
   meshes/inertia → silent sim-to-real degradation.
10. **`bad_orientation` tolerance** — at training, episodes terminate at
    base roll/pitch > 0.8 rad ≈ 46°. At deploy, you should monitor this
    and trigger a recovery FSM state if exceeded; don't let the policy
    keep running on a robot that's about to tip.

---

## 9. Minimal deploy checklist

- [ ] Pick checkpoint: `~/projects/doga/logs/rsl_rl/unitree_go2_footholds/<best-run>/model_X.pt`
- [ ] Run `play.py` once with `--checkpoint <ckpt>` to export `.pt` + `.onnx`
- [ ] Copy `policy.onnx` to Jetson at known path
- [ ] Build `unitree_rl_lab/deploy/robots/go2/` C++ skeleton on Jetson
- [ ] Replace x86 ONNX Runtime under `thirdparty/` with aarch64 build
- [ ] Modify `State_RLBase` to:
   - Add foothold-planner subscription (see §6)
   - Compute `foot_positions` via FK (Pinocchio or hand-rolled)
   - Pack obs vector in the exact order from §4
- [ ] Verify `joint_sdk_names` ordering against deployed URDF
- [ ] Test sequence:
   - Hold Go2 in air, run with zero velocity command — verify gait emerges
   - Place on flat ground at low velocity command — verify it walks
   - Place on stepping-stone setup with planner running — verify foothold tracking
- [ ] Add a watchdog: if policy output goes NaN or `bad_orientation` triggers,
   transition FSM → `FixStand` → `Passive`.

---

## 10. References

- `~/unitree_rl_lab/deploy/` — Unitree's reference C++ deployment for Go2/B2/H1/G1
- `~/unitree_rl_lab/source/.../assets/robots/unitree.py` — Go2 actuator + joint mapping
- `~/projects/doga/HIGH-LEVEL-RESEARCH.md` §15b — sim2real gaps to address before fielding
- ONNX Runtime aarch64: <https://github.com/microsoft/onnxruntime/releases>
- TensorRT (for FP16/INT8): <https://developer.nvidia.com/tensorrt>
- Pinocchio (FK library): <https://github.com/stack-of-tasks/pinocchio>
- elevation_mapping_cupy (foothold planner input): <https://github.com/leggedrobotics/elevation_mapping_cupy>
