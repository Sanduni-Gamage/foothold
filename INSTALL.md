# Isaac Lab Install for Unitree Go2 Training

**Target machine:** Ubuntu 24.04.3 LTS, 2× RTX 5060 Ti 16 GB (Blackwell, sm_120),
NVIDIA driver 580.126.09, 64 GB RAM, 376 GB free, gcc 13.3, miniforge3
already installed at `~/miniforge3`.

Every command and version claim below cites a primary source from NVIDIA, the
Isaac Lab project, PyTorch, or Unitree. No third-party guides are used.

---

## 1. Version selection

| Component  | Pinned version | Source |
|------------|----------------|--------|
| Isaac Lab  | **v2.3.2** (released 2026-02-02; latest non-beta on the 2.x line) | [Isaac Lab releases][rel] |
| Isaac Sim  | **5.1.0** (required by Isaac Lab 2.3.x) | [Isaac Lab pip install docs][pip] |
| Python     | **3.11** (required for Isaac Sim 5.x) | [Isaac Lab pip install docs][pip] |
| PyTorch    | **2.7.0 + cu128** (Blackwell-capable wheels) | [Isaac Lab pip install docs][pip], [PyTorch 2.7 release][torch27] |

Isaac Lab v3.0.0-beta (2026-03-17) targets Isaac Sim 6.0 and Python 3.12, but is
flagged by NVIDIA as a beta with breaking changes — not used here.
[Source: Isaac Lab releases][rel]

[rel]: https://github.com/isaac-sim/IsaacLab/releases
[pip]: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
[torch27]: https://pytorch.org/blog/pytorch-2-7/

---

## 2. Verify hardware/OS prerequisites

NVIDIA's official Isaac Sim 5.1 system-requirements page lists:

| Requirement              | Spec                              | This machine | OK? |
|--------------------------|-----------------------------------|--------------|-----|
| OS                       | Ubuntu 22.04 / 24.04 or Win 10/11 | Ubuntu 24.04 | ✅  |
| NVIDIA driver (Linux)    | ≥ 580.65.06                       | 580.126.09   | ✅  |
| GPU                      | RT-Core GPU; "Minimum: RTX 4080"  | RTX 5060 Ti  | ⚠ see note |
| GPU VRAM                 | ≥ 16 GB                           | 16 GB        | ✅  |
| System RAM               | ≥ 32 GB                           | 64 GB        | ✅  |
| Disk                     | ≥ 50 GB SSD                       | 376 GB free  | ✅  |
| GLIBC                    | ≥ 2.35 (`manylinux_2_35_x86_64`)  | 2.39         | ✅  |
| Vulkan loader            | required (Sim is a Vulkan renderer) | `libvulkan1` present | ✅ |

**Sources:**
- [Isaac Sim 5.1 — System Requirements][sysreq]
- [Isaac Sim 5.1 — Python Environment Install][siminstpy] (GLIBC ≥ 2.35 / `manylinux_2_35_x86_64`)

[sysreq]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
[siminstpy]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html

> Note on the GPU row. NVIDIA's table lists "GeForce RTX 4080" as **Minimum**
> and "GeForce RTX 5080" under **Good**; the RTX 5060 Ti is not enumerated.
> The 5060 Ti has RT cores and 16 GB VRAM (matching the minimum), and shares
> the Blackwell architecture explicitly listed on the same page (RTX 5080,
> RTX PRO 6000 Blackwell). It satisfies the documented minimum criteria but is
> not a tier NVIDIA explicitly tests against. Source: same page above.

Pre-flight verification:

```bash
ldd --version | head -1                          # glibc ≥ 2.35
ls /usr/share/vulkan/icd.d/ | grep -i nvidia     # NVIDIA Vulkan ICD present
nvidia-smi --query-gpu=driver_version --format=csv,noheader  # driver
```

---

## 3. APT system dependencies

The pip-install docs require:

```bash
sudo apt update
sudo apt install -y cmake build-essential
# vulkan-tools is optional but gives us `vulkaninfo` for diagnostics
sudo apt install -y vulkan-tools
```

**Source:** [Isaac Lab pip install — Linux requirements][pip]

---

## 4. Create the conda environment

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install --upgrade pip

# Avoid the system CUDA_HOME=/usr leaking into pip-built extensions:
unset CUDA_HOME
# Accept the Omniverse EULA non-interactively (required to launch isaacsim):
export OMNI_KIT_ACCEPT_EULA=YES
```

**Sources:**
- [Isaac Lab pip install docs][pip] (env name + Python version, verbatim)
- [Isaac Sim 5.1 Python install — EULA env var][siminstpy] (`OMNI_KIT_ACCEPT_EULA=YES`)

> If you want `OMNI_KIT_ACCEPT_EULA=YES` to persist in this conda env, store it
> via conda's per-env activate hook: `conda env config vars set OMNI_KIT_ACCEPT_EULA=YES -n env_isaaclab` then re-activate.

---

## 5. Install Isaac Sim 5.1.0 from NVIDIA's PyPI

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

**Source (verbatim):** [Isaac Lab pip install docs][pip]

The `[all,extscache]` extras pull in the full Isaac Sim runtime plus the
extension cache; `https://pypi.nvidia.com` is NVIDIA's official wheel index.

---

## 6. Pin PyTorch 2.7.0 with CUDA 12.8 wheels

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

**Source (verbatim Linux x86_64 command):** [Isaac Lab pip install docs][pip]

PyTorch 2.7 explicitly states: *"PyTorch 2.7 introduces support for NVIDIA's
new Blackwell GPU architecture and ships pre-built wheels for CUDA 12.8."*
The `cu128` wheel index is the one tagged for Blackwell.
**Source:** [PyTorch 2.7 release announcement][torch27]

> ⚠ **Validate this immediately after install.** PyTorch 2.7's release notes
> announce "Blackwell support" but don't enumerate sm_120 (consumer Blackwell)
> verbatim, and tracking issue
> [pytorch/pytorch #164342](https://github.com/pytorch/pytorch/issues/164342)
> ("Official support for sm_120 in stable PyTorch builds") shows real users
> on RTX 50-series have hit gaps. NVIDIA's CUDA 12.8 release notes *do*
> explicitly add compiler support for `SM_100`, `SM_101`, `SM_120`
> ([CUDA 12.8 archive][cuda128]), so the toolchain is capable; whether
> the prebuilt cu128 wheel ships sm_120 kernels needs to be verified
> empirically — see §6.1.

[cuda128]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html

### 6.1 Post-PyTorch validation (do this before installing Isaac Lab)

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device 0:", torch.cuda.get_device_name(0))
print("arch list:", torch.cuda.get_arch_list())
print("capability(0):", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
y = (x @ x).sum().item()
print("matmul OK, sum=", y)
PY
```

Pass criteria:
- `arch list` contains `sm_120` (or equivalent like `sm_120a`)
- `capability(0)` is `(12, 0)`
- the matmul prints a number without a `no kernel image is available` error

**Fallback if sm_120 is absent:** install a PyTorch nightly cu128 wheel:
```bash
pip install --pre --upgrade torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```
(Nightly cu128 historically ships sm_120; see PyTorch issue #164342 thread.)

---

## 7. Clone Isaac Lab at v2.3.2 and run the installer

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh --install
```

`--install` (with no framework argument) installs Isaac Lab plus all RL
frameworks (rsl_rl, skrl, rl_games, sb3). To install only one:
`./isaaclab.sh --install rsl_rl`.

**Source:** [Isaac Lab pip install docs][pip] (install command);
[Isaac Lab releases][rel] (v2.3.2 tag).

---

## 8. Smoke test

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

This is the standard verification step from the Isaac Lab quickstart.
**Source:** [Isaac Lab installation index][instidx]

[instidx]: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

---

## 9. Train Unitree Go2 — option A: built-in Isaac Lab tasks

Isaac Lab ships four registered Go2 tasks. Verbatim IDs from the official
environments page:

- `Isaac-Velocity-Flat-Unitree-Go2-v0`
- `Isaac-Velocity-Flat-Unitree-Go2-Play-v0`
- `Isaac-Velocity-Rough-Unitree-Go2-v0`
- `Isaac-Velocity-Rough-Unitree-Go2-Play-v0`

All four support **rsl_rl (PPO)** and **skrl (PPO)** configs.

**Source:** [Isaac Lab — Environments][envs]

[envs]: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html

Train (rough terrain, rsl_rl, single-GPU):

```bash
export CUDA_VISIBLE_DEVICES=0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless
```

Replay a trained policy:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 --num_envs 32
```

The training script and flags follow the structure documented in the same
environments page (CLI flags `--task`, `--headless`, `--num_envs` are listed).

---

## 10. Train Unitree Go2 — option B: Unitree's own extension

Unitree maintains [`unitreerobotics/unitree_rl_lab`][urll], an Isaac Lab
extension with Unitree-tuned configs and sim-to-real assets. The repo's own
README states:

- *"a set of reinforcement learning environments for Unitree robots, built on
  top of IsaacLab"*
- Compatible with **IsaacSim 5.1.0** and **IsaacLab 2.3.0** (works with v2.3.2)
- Currently supports **Go2, H1, and G1-29dof**

[urll]: https://github.com/unitreerobotics/unitree_rl_lab

Install (after step 7):

```bash
conda activate env_isaaclab
cd ~
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab
./unitree_rl_lab.sh -i
```

For URDF-based robot models (recommended for Isaac Sim ≥ 5.0):

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git
```

Train Go2:

```bash
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity
# equivalent direct invocation
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Velocity
```

Play:

```bash
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity
```

**Sources (all verbatim from the repo README):**
[unitree_rl_lab README][urll-readme],
[unitree_ros repo][uros]

[urll-readme]: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/README.md
[uros]: https://github.com/unitreerobotics/unitree_ros

> The Unitree README explicitly documents end-to-end commands only for the G1
> robot; the Go2 task name (`Unitree-Go2-Velocity`) follows the pattern shown
> for G1/H1 and matches the Go2 entries in the repo's task registry.
> Verify the task name with `python scripts/rsl_rl/train.py --help` before
> running, in case Unitree renames it.

---

## 11. Known Blackwell caveats (documented issues)

These are filed against `isaac-sim/IsaacLab` and are not closed as of the
date of writing. None blocks Go2 velocity locomotion training (no camera
sensors, no perception), but worth knowing:

| Issue | Symptom | Affects Go2 velocity training? |
|-------|---------|--------------------------------|
| [#4951][i4951] | TiledCamera sensor hangs on sm_120 | No — Go2 velocity tasks use no cameras |
| [#3612 (discussion)][d3612] | PhysX GPU pipeline reportedly falls back to CPU on Blackwell on some setups | Watch the training log; if you see "PhysX falling back to CPU", training will be ~10× slower |
| [#2483][i2483] | Older torch wheels lack sm_120 kernels | Resolved by the cu128 wheel pinned in Step 6 |

[i4951]: https://github.com/isaac-sim/IsaacLab/issues/4951
[d3612]: https://github.com/isaac-sim/IsaacLab/discussions/3612
[i2483]: https://github.com/isaac-sim/IsaacLab/issues/2483

---

## 12. End-to-end command summary

```bash
# pre-flight
ldd --version | head -1                          # glibc ≥ 2.35
ls /usr/share/vulkan/icd.d/ | grep -i nvidia     # NVIDIA Vulkan ICD

# system deps
sudo apt update && sudo apt install -y cmake build-essential vulkan-tools

# conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install --upgrade pip
unset CUDA_HOME
export OMNI_KIT_ACCEPT_EULA=YES
conda env config vars set OMNI_KIT_ACCEPT_EULA=YES -n env_isaaclab

# Isaac Sim + PyTorch
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Validate sm_120 BEFORE proceeding
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability(0)); x=torch.randn(1024,1024,device='cuda'); print((x@x).sum().item())"

# Isaac Lab
cd ~ && git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v2.3.2
./isaaclab.sh --install

# smoke test
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# (optional) Unitree's tuned extension
cd ~ && git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab && ./unitree_rl_lab.sh -i

# train Go2 (built-in task)
export CUDA_VISIBLE_DEVICES=0
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 --headless
```

> Isaac Sim **6.0.0** (released 2026-03-16 on PyPI) and Isaac Lab **v3.0.0-beta**
> (2026-03-17) exist but are intentionally not used here — v3.0.0 is flagged
> beta with breaking changes and Sim 6.0 is "Early Developer Release."
> Sources: [Isaac Lab releases][rel], [isaacsim on PyPI](https://pypi.org/project/isaacsim/).

---

## Source index

- NVIDIA — [Isaac Sim 5.1 system requirements][sysreq]
- NVIDIA — [Isaac Lab pip installation][pip]
- NVIDIA — [Isaac Lab installation index][instidx]
- NVIDIA — [Isaac Lab environments (registered tasks)][envs]
- NVIDIA — [Isaac Lab release tags][rel]
- PyTorch — [PyTorch 2.7 release announcement][torch27]
- Unitree — [unitree_rl_lab repository / README][urll-readme]
- Unitree — [unitree_ros (URDF assets)][uros]
- GitHub issues (Blackwell): [#4951][i4951], [#3612][d3612], [#2483][i2483]
