# Foothold task: yaw tracking analysis

Notes on the `Unitree-Go2-Footholds` run (started 2026-05-05_21-37-06, GPU 1)
during iteration ~48 300. The proprio-only `Unitree-Go2-Velocity` baseline
(`2026-05-03_18-37-15`, ended at iter 25 600) is used as the reference.

---

## 1. Symptom

| Metric                            | Proprio baseline @ iter 25 600 | Foothold run @ iter 48 300 |
|-----------------------------------|-------------------------------:|---------------------------:|
| `Metrics/.../error_vel_xy`        |                          0.33 |                       0.24 |
| `Metrics/.../error_vel_yaw`       |                          0.38 |                       **0.57** |
| `Episode_Reward/track_lin_vel_xy` |                          1.27 |                       1.39 |
| `Episode_Reward/track_ang_vel_z`  |                          0.62 |                       0.63 |

Linear tracking is *better* in the foothold run; yaw tracking is markedly
worse — at ~0.6 rad/s mean abs error on a ±1 rad/s command range, with the
reward kernel (`std = 0.5`) effectively saturated outside the tolerance
window.

These are independent runs from scratch (the foothold launcher does **not**
warm-start — `obs shape change incompatible with --resume`, see
`launch-24h-footholds.sh`). So this isn't a regression in time; it's two
runs that converge to different yaw-tracking equilibria.

---

## 2. Worse than do-nothing

For `cmd ~ U(−1, 1)`, a policy that always outputs `yaw_actual = 0` would
score `mean(|cmd|) = 0.5`. The foothold policy at **0.57 > 0.5** is yawing in
ways uncorrelated with (sometimes opposite to) the command — not just
ignoring it. The proprio baseline's 0.38 < 0.5 confirms a policy *can*
beat the do-nothing floor here. The foothold policy is below it.

---

## 3. Verified mechanism: dense `foothold_progress` gradient is anti-yaw

### 3.1 Sparse landing reward is not the driver

`Episode_Reward/X` is `(per-episode sum of weighted reward) / max_episode_length_s`
(`isaaclab/managers/reward_manager.py:120`). Decoding:

- `foothold_landing = 0.10/s`, weight 9.4, episode length 20 s → ≈ **0.21
  claimed footholds per episode** out of 20 stones. ~1 % claim rate.

The sparse landing reward is not where the gradient is coming from.

### 3.2 The dense `foothold_progress` term is

`foothold_progress` (`mdp/rewards.py:262`):

```python
progress = (foot_vel_w * direction).sum(dim=-1)   # swing-foot vel toward stone
```

It rewards the **world-frame** component of swing-foot velocity pointing at
the nearest unclaimed stone — i.e. *forward in world frame*, where the chain
sits. Yawing the body curves the foot trajectory; the rewarded forward
component shrinks and an unrewarded tangential component grows. So:

> `foothold_progress` actively *penalizes* yawing. It is not merely
> indifferent to the yaw command — its gradient pushes against it every
> swing step on every env.

Weighted reward 0.29/s × 20 s = 5.8 weighted per episode, dominating the
~12.6 weighted per episode from `track_ang_vel_z` only because *
`track_ang_vel_z` is saturated low* (kernel reward at err=0.57 is ~0.27).
The dense yaw-suppressive gradient wins in expectation.

### 3.3 The chain placement makes the conflict structural

`reset_footholds` (`mdp/foothold_state.py:73`) places 20 stones at
`i × 0.50 m` along the **spawn yaw direction**, jittered ±0.15 m
perpendicular, transformed once to world frame, then **frozen for the
20 s episode** (`mode="reset"`).

The velocity command is sampled in body frame, with full range
`ang_vel_z = (−1, 1)` rad/s, resampled every 10 s. Mean |yaw cmd| = 0.5 rad/s,
so the body accumulates ~10 rad of commanded rotation per episode. The
chain is straight; the body is supposed to spiral. They cannot both be
satisfied. PPO arbitrages by suppressing yaw — that is what we observe.

### 3.4 Information is not the bottleneck

`nearest_unclaimed_footholds` (`mdp/observations.py:107`) rotates the chain
positions by `quat_apply_inverse(base_quat_w, …)` into the **robot frame**.
After the body yaws, the policy still observes where the (now-rotated) chain
is. It could go back to it; the gradient just doesn't reward doing so.

---

## 4. New finding: yaw curriculum exists in code but is never wired

`mdp/curriculums.py:62` defines a complete `ang_vel_cmd_levels` function
(structural twin of `lin_vel_cmd_levels`). `CurriculumCfg` in
`velocity_env_cfg.py:373` registers only `lin_vel_cmd_levels`. So **yaw
command range starts at ±1.0 and stays there from iteration 0 in every run
on this machine** — proprio baseline included.

That alone is enough to explain the proprio baseline plateauing at err=0.38:
the kernel `std=0.5` only delivers strong gradient inside ±0.5 rad/s, but
the policy was trained on commands up to ±1.0 from the first iteration with
no graduated signal. For the foothold task it is much worse, because there
is also a competing dense gradient suppressing yaw entirely.

---

## 5. Why the simple knobs don't fix this

| Knob                                    | Why it fails                                          |
|-----------------------------------------|-------------------------------------------------------|
| Bump `track_ang_vel_z` weight 0.75 → 2.0 | Doesn't change `foothold_progress` direction; equilibrium shifts but conflict remains. |
| Tighten kernel `std` from 0.5 → 0.3     | More gradient near err=0.3 but kernel saturates closer to zero — increases pressure but still loses to dense forward-progress. |
| Set yaw range ±0.2 rad/s                | Even ±0.2 × 20 s = 4 rad accumulated drift; chain has 0.15 m jitter — body drifts off-chain in seconds. |
| Halve foothold weights                  | Structural conflict with the chain geometry survives any positive weight. |

The conflict is between the **objective specifications** (body-frame
velocity command vs. world-frame forward chain), not the relative weights.

---

## 6. Recommendations

### 6.1 Minimum viable fix (small code change, big yield)

Two edits, both in `unitree_rl_lab`:

1. Register the existing yaw curriculum:

   `tasks/locomotion/robots/go2/velocity_env_cfg.py` (`CurriculumCfg`):
   ```python
   ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)
   ```

2. Initial yaw range to zero, with a graduated limit:

   Same file, `CommandsCfg.base_velocity`:
   ```python
   ranges       = ... ang_vel_z=(0.0, 0.0)     # initial — no yaw command
   limit_ranges = ... ang_vel_z=(-0.5, 0.5)    # final — half of upstream
   ```

   The curriculum widens the range only when `track_ang_vel_z` reward
   exceeds 80 % of weight (`curriculums.py:88`), gating advance on actual
   tracking competence.

After these, ctrl-C the current foothold run and relaunch fresh. Linear
tracking should remain strong (already at err=0.24); yaw should converge
toward 0.2–0.3 rad/s within a comparable iteration budget. The chain
incompatibility remains structurally present at the upper limit of ±0.5
rad/s, but bounded enough that the policy can hold tracking under a chain
that diverges only modestly over 20 s.

### 6.2 Principled fix (next iteration)

Make the chain follow the velocity command, not the spawn yaw:

- Generate the chain as an **arc** consistent with `(lin_vel_x, ang_vel_z)`
  at episode reset and at every command resample (`resampling_time_range`
  is currently 10 s; trigger off the same event).
- For pure forward (`ω=0`), the arc degenerates to the existing straight
  line. For nonzero `ω`, place stones at body-frame
  `(R sin(s ω / v), R(1 − cos(s ω / v)))` for arc-length `s ∈
  [spacing, …, n_stones·spacing]`, where `R = v / ω`.
- Anchor claimed stones in world frame; only project unclaimed ones onto
  the new arc when the command resamples (otherwise claims aren't stable).

Code changes touch `mdp/foothold_state.py:reset_footholds`, add a new
`resample_footholds` term to `FootholdEventCfg` keyed on cmd resample, and
update `update_claims` to handle the partial reprojection. Probably a day
of work plus a short validation run.

### 6.3 What not to do

Do not just retune weights or tighten the yaw kernel. The conflict between
"track an arbitrary body-frame yaw command" and "step on a fixed
world-frame straight chain" is exactly satisfied for `ang_vel_z = 0` and
exactly violated otherwise. No weight ratio resolves it; the task spec
must change.

---

## 7. Open questions

- **Is `foothold_progress` even helping?** ~0.21 claims/episode. Most of
  what the dense gradient is doing is shaping body posture, not landing.
  Worth ablating with `weight=0` once the yaw issue is decoupled.
- **Is `episode_length_s = 20` plus 20 stones × 0.5 m = 10 m chain
  matched to the achievable forward velocity?** At cmd `lin_vel_x = 1.0`
  and err 0.24 → actual ≈ 0.76 m/s × 20 s = 15 m of travel. The dog can
  outrun the chain. But the claim rate is ~1 %, so it isn't outrunning it
  in practice — it's running roughly forward but not landing precisely.
  Once yaw is decoupled, look at this next.
- **Should `off_foothold_landing` (weight 0.3) be raised** once the policy
  is actually landing on stones? Currently almost no events trigger it; it
  isn't shaping much.

---

## 8. Files referenced

- `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`
- `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_footholds_env_cfg.py`
- `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/foothold_state.py`
- `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`
- `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py`
- `~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`
- `~/projects/doga/launch-24h-footholds.sh`
- `~/projects/doga/logs/rsl_rl/unitree_go2_footholds/2026-05-05_21-37-06/` — current run
- `~/projects/doga/logs/rsl_rl/unitree_go2_velocity/2026-05-03_18-37-15/` — proprio baseline
