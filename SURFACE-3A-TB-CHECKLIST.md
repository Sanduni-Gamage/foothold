# Surface task Stage 3a — TensorBoard checklist

Watchlist for the first ~2 hours of `Unitree-Go2-Surface` training launched via `launch-24h-surface-3a.sh`. Decide kill-or-continue after seeing the trajectories below settle (≈ iter 1000–2000, ~1.5–2 h wall-clock at 4096 envs).

**Stage-2 baseline anchor** — proprio rough-terrain walker, run `2026-05-03_18-37-15`, iter 25 600. Numbers from `FOOTHOLDS-YAW-ANALYSIS.md`:

| Anchor metric | Stage-2 value (per-second weighted) |
|---|---|
| `Episode_Reward/track_lin_vel_xy` | 1.27 |
| `Episode_Reward/track_ang_vel_z` | 0.62 |
| `Episode_Termination/bad_orientation` | ~6 % |
| `Curriculum/terrain_levels` | ~0.85 |
| `Curriculum/lin_vel_cmd_levels` | 1.0 (saturated) |
| `flat_orientation_l2` | TBD — read from the run's TB before Stage 3a starts. Use as comparison anchor for the orientation-conflict check. |

(`flat_orientation_l2` Stage-2 value is the only anchor not in the analysis doc. If you don't have it cached, open the run's TB at iter 25 600 and read the `Episode_Reward/flat_orientation_l2` curve once before launching Stage 3a. Sub-1-minute task.)

The decoder constant for the five surface-stats metrics is `1e9` (multiply TB reading by `1e9` → count per episode). See HANDOFF §15.1.

---

## A. Velocity tracking — must not regress

### A1. `Episode_Reward/track_lin_vel_xy`
- **Expected initial:** 0.0 at iter 0 (random policy).
- **Direction:** rises monotonically through curriculum stages; should approach 0.9–1.3 by ~iter 1500.
- **Kill criterion:** `< 0.89` (= 0.7 × Stage-2 mean of 1.27) at iter 2000. The surface reward is supposed to be shaping, not dominating; if velocity tracking can't reach 70 % of the Stage-2 floor by iter 2000, the calibration is wrong.

### A2. `Episode_Reward/track_ang_vel_z` — yaw curriculum verification
- **Expected initial:** 0.0 (and `Curriculum/ang_vel_cmd_levels = 0.0` initial range; that's by design — yaw command range starts collapsed to `(0, 0)`).
- **Direction:** stays near 0 until policy reaches >80 % of `track_lin_vel_xy_exp` weight (then curriculum starts widening). `Curriculum/ang_vel_cmd_levels` should begin ticking up from 0.0 toward 1.0 around iter 500–1500. Once it does, `track_ang_vel_z` should rise comparably to Stage-2 (target ≥ 0.5 by iter 4000).
- **Kill criterion:** `ang_vel_cmd_levels` still at 0.0 at iter 5000 AND `lin_vel_cmd_levels` saturated. That would mean the yaw-curriculum gate (reward > 0.8 × weight) is unsatisfiable, suggesting a deeper bug.

---

## B. Surface reward — must rise from zero

### B1. `Episode_Reward/surface_aware_landing`
- **Expected initial:** ~0. With 50 % FLAT fraction at random and `r_flat=0.04`, expected ceiling ≈ (60 landings × 0.04) / 20 s = **0.12 per-second weighted** (≈ 2.4 per episode).
- **Direction:** must rise from 0 (visible above noise) by iter 1000. Should asymptote toward ~0.16–0.20 if `fraction_landings_on_flat` climbs from 0.5 → 0.8.
- **Kill criterion:** flat at zero at iter 2000. That means either (a) the policy never lands feet (gait broken — check B5), or (b) the label-map lookup is broken (check B4).

### B2. `Episode_Reward/landing_fraction_flat`  (decode ×1e9 → count/ep)
- **Expected initial:** TB value ≈ 6e-8 (= 60 / 1e9). Decoded count ≈ 60 per episode.
- **Direction:** **must rise above the random ~0.5 fraction**. Decoded count target: 75–100+ per episode by iter 5000 if Stage 3a is working. Compute fraction = `landing_fraction_flat / landings_per_episode` (TB or offline).
- **Kill criterion:** decoded count stays at ≤ 65 per episode (= still random) at iter 5000. With `r_flat=0.04` exerting too little pressure, step up to `r_flat=0.06` for Stage 3b per the calibration note in HANDOFF §15.2.

### B3. `Episode_Reward/landing_fraction_unsafe`  (decode ×1e9 → count/ep)
- **Expected initial:** TB ≈ 4e-8 (≈ 40/ep at ~33 % UNSAFE fraction; ramps + corner-smoothed pixels).
- **Direction (Stage 3a):** any value — the `p_unsafe=0` weight gives no direct pressure to reduce it. Watch only; don't kill.
- **Direction (Stage 3b expectation, for later reference):** must drop below ~5 % fraction (decoded count ≤ 6/ep) by iter 5000 — that's the kill criterion FOR Stage 3b, not 3a.

### B4. `Episode_Reward/landing_fraction_lipped`  (decode ×1e9 → count/ep)
- **Expected initial:** TB ≈ 1.5e-8 (≈ 15/ep at ~12 % LIPPED fraction with 1-px dilation; see HANDOFF §15.3 corner-smoothing note).
- **Direction (Stage 3a):** watch only; the reward has `p_lipped=0`. Useful as a **detector-sanity** metric: if this is permanently zero, the dilation step is broken or the registry is misaligned. If it's permanently 1.0 fraction, the calibration over-classified.
- **Kill criterion (sanity, not policy):** decoded count = 0 across the run. Means LIPPED never fires → label map likely empty for non-edge pixels.

### B5. `Episode_Reward/landings_per_episode`  (decode ×1e9 → total count/ep)
- **Expected initial:** TB ≈ 1.2e-7 (≈ 120 landings/ep at a trotting gait).
- **Direction:** stable ~80–160 per episode through training. Drops as command range expands and ep length effective increases.
- **Kill criterion:** decoded count < 30/ep at iter 1000 → gait collapse, robot isn't stepping (shuffling or falling). Same failure mode as the current `2026-05-09_01-07-01` pyramid run that HANDOFF flagged as degenerate.

---

## C. Conflicts to watch (Stage 2 carryover rewards)

### C1. `Episode_Reward/flat_orientation_l2`
- **Anchor:** TBD; read from run `2026-05-03_18-37-15`. Provisionally expected around `-0.3` per-second weighted at Stage-2 maturity.
- **Direction:** roughly stable, drifting slightly more negative as the policy learns to step onto bumps (body pitches up at step-up).
- **Kill criterion:** value drops **>3× more negative than Stage-2 anchor** by iter 3000. Means the orientation penalty is fighting the surface reward — drop weight from −2.5 to −1.0 for the next stage per HANDOFF §15.2 follow-up note (Decision 4 in the design review).

### C2. `Curriculum/terrain_levels`
- **Expected initial:** 0.10–0.20 (since `max_init_terrain_level=1` clamps starting difficulty).
- **Direction:** rises monotonically once the policy can walk; Stage-2 baseline saturated near 0.85.
- **Kill criterion:** plateau **below 0.4 by iter 5000**. Indicates `foot_clearance_reward (target_height=0.08 m)` is incompatible with the larger bumps. Raise target_height to 0.12 m for Stage 3b per HANDOFF §15.2 / Decision 5.

### C3. `Episode_Reward/feet_air_time`
- **Direction:** small positive (it's the +0.1 weight × air-time excess); fires on the same swing→stance events as `surface_aware_landing`.
- **Kill criterion:** none specific; flagged for the writeup. If `landings_per_episode` is healthy but `feet_air_time` is sharply negative, the policy is taking short hops below the 0.5 s air-time threshold — slow shuffle, same failure mode as terrain_levels stuck low.

---

## D. Termination + perception health

### D1. `Episode_Termination/bad_orientation`
- **Anchor:** Stage-2 baseline ~6 %.
- **Expected initial:** high (50–80 %) at iter 0 while policy is random; falls fast.
- **Kill criterion:** > 30 % at iter 2000 → policy isn't recovering balance on the bumps. Possibly a friction/clearance interaction; investigate before scaling stage.

### D2. `Episode_Termination/base_contact`
- **Kill criterion:** > 5 % sustained. Indicates the robot is belly-flopping; same failure-mode signature as the early lidar runs in HANDOFF §5 row 2.

### D3. Height-scan obs sanity — pseudo-metric
The five surface-stats metrics drive a useful indirect check on the height scanner: if the robot ever falls outside the (num_rows × num_cols) tile grid (off the terrain border), `landing_fraction_out_of_bounds` rises sharply. Specifically:

- **Expected:** decoded count ≤ 1 per episode (rare). Recall that `OUT_OF_BOUNDS=255` is the sentinel for feet outside any tile.
- **Kill criterion:** decoded count > 10/ep sustained for >500 iters → either the env spawns the robot outside the grid (config bug) or the policy is sprinting off the edge. Inspect `env._surface_tile_corner_w` vs spawn poses.

Direct height-scan obs limits (no native TB output, but visible via `tensorboard --inspect` or a play-mode probe):
- Each ray's clipped value is in `[-1.0, +1.0]` m (stock formulation, `mdp.height_scan` with `offset=0.5`).
- If you see all rays pinned to either `-1.0` or `+1.0`: the robot has fallen off the terrain (or the height_scanner mount is misconfigured, e.g., the +20 m base offset isn't applied).

---

## E. Quick decision flow at iter 2000

1. **A1 < 0.89?** → kill, the surface reward is dominating despite the calibration. Re-derive weights from observed landings/ep and FLAT fraction.
2. **B1 still at 0 AND B5 healthy (~120/ep)?** → kill, label-map lookup is broken. First check: `landing_fraction_out_of_bounds` should be near zero. If it isn't, the tile origin offset is wrong.
3. **B5 < 30/ep?** → kill, gait collapse independent of surface reward. Same loop as the pyramid-ridged-stairs run.
4. **All A and B nominal but C1 way more negative than Stage-2?** → keep running, but plan to drop `flat_orientation_l2` for Stage 3b.
5. **All nominal?** → let it run the full 24+ h. Decision on Stage 3b vs Stage 3a-extended waits for `landing_fraction_flat` trajectory to plateau.
