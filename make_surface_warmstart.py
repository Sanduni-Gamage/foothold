#!/usr/bin/env python3
"""Build a warm-start checkpoint for Unitree-Go2-Surface from the Stage-2
rough-terrain walker (model_25600.pt).

The surface task's observation vector is a strict PREFIX-superset of the
Stage-2 walker's:

  policy : [base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel,
            joint_vel_rel, last_action]                                  (45)
           + height_scan(187) + foot_positions(12)                      -> 244
  critic : [base_lin_vel, base_ang_vel, projected_gravity,
            velocity_commands, joint_pos_rel, joint_vel_rel, joint_effort,
            last_action]                                                 (60)
           + height_scan_clean(187) + foot_positions_clean(12)          -> 259

So the transfer is a clean column-prefix copy of the first Linear layer, with
the NEW input columns ZERO-INITIALISED. Zero-init makes the warm-started
policy *function-identical* to the Stage-2 walker at iteration 0 (the new
obs are ignored until training learns to use them) — preserving all the
velocity-tracking competence that the from-scratch surface run lacked.

Everything else (hidden layers, action head, noise std) transfers verbatim;
the parameter COUNT is unchanged (only two weight tensors grow in width), so
the source optimizer's param_groups remain structurally valid. We clear the
optimizer state (drop stale 45/60-input Adam moments) and reset iter to 0,
giving true warm-start semantics under stock `--resume` with no code changes.

Usage:
  python make_surface_warmstart.py            # uses default paths
  python make_surface_warmstart.py --src <model.pt> --out <dir>/<file.pt>
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

# --- surface task obs dims (must match SurfaceObsCfg in velocity_surface_env_cfg.py)
SRC_ACTOR_IN = 45
SRC_CRITIC_IN = 60
TGT_ACTOR_IN = 244
TGT_CRITIC_IN = 259

DEFAULT_SRC = (
    "/home/anyone/projects/dogb/checkpoints_backup/"
    "unitree_go2_velocity/2026-05-03_18-37-15/model_25600.pt"
)
DEFAULT_OUT = (
    "/home/anyone/projects/dogb/logs/rsl_rl/unitree_go2_surface/"
    "warmstart_rough_25600/model_25600_surface.pt"
)


def _forward_mlp(state: dict, prefix: str, x: torch.Tensor) -> torch.Tensor:
    """Manual forward through an rsl-rl ActorCritic MLP (Linear/ELU/.../Linear).

    Layers are indexed 0,2,4,6 (Linear) with ELU between. Matches the
    [512, 256, 128] -> out architecture.
    """
    out = x
    idx = 0
    while f"{prefix}.{idx}.weight" in state:
        w = state[f"{prefix}.{idx}.weight"]
        b = state[f"{prefix}.{idx}.bias"]
        out = F.linear(out, w, b)
        # apply ELU on all but the final linear layer
        if f"{prefix}.{idx + 2}.weight" in state:
            out = F.elu(out)
        idx += 2
    return out


def build_warmstart(src_path: str, out_path: str) -> None:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    src_sd = ck["model_state_dict"]

    # --- sanity: confirm source input dims are what we expect
    a_in = src_sd["actor.0.weight"].shape[1]
    c_in = src_sd["critic.0.weight"].shape[1]
    assert a_in == SRC_ACTOR_IN, f"source actor input {a_in} != {SRC_ACTOR_IN}"
    assert c_in == SRC_CRITIC_IN, f"source critic input {c_in} != {SRC_CRITIC_IN}"

    # --- build target model_state_dict (copy verbatim, widen the two inputs)
    tgt_sd: dict[str, torch.Tensor] = {}
    for k, v in src_sd.items():
        if k == "actor.0.weight":
            w = torch.zeros(v.shape[0], TGT_ACTOR_IN, dtype=v.dtype)
            w[:, :SRC_ACTOR_IN] = v
            tgt_sd[k] = w
        elif k == "critic.0.weight":
            w = torch.zeros(v.shape[0], TGT_CRITIC_IN, dtype=v.dtype)
            w[:, :SRC_CRITIC_IN] = v
            tgt_sd[k] = w
        else:
            tgt_sd[k] = v.clone()

    # --- verify function-preservation: actor/critic outputs identical when the
    #     new obs columns are zero.
    torch.manual_seed(0)
    proprio = torch.randn(8, SRC_ACTOR_IN)
    crit_in = torch.randn(8, SRC_CRITIC_IN)
    a_src = _forward_mlp(src_sd, "actor", proprio)
    a_tgt = _forward_mlp(tgt_sd, "actor", F.pad(proprio, (0, TGT_ACTOR_IN - SRC_ACTOR_IN)))
    c_src = _forward_mlp(src_sd, "critic", crit_in)
    c_tgt = _forward_mlp(tgt_sd, "critic", F.pad(crit_in, (0, TGT_CRITIC_IN - SRC_CRITIC_IN)))
    a_diff = (a_src - a_tgt).abs().max().item()
    c_diff = (c_src - c_tgt).abs().max().item()
    # Tolerance 1e-4: zero-padding the input widens the matmul by 199/199 cols,
    # changing float32 summation order -> ~1e-6 rounding noise is expected and
    # harmless. A real transfer bug (wrong columns) would give diffs O(0.1+).
    assert a_diff < 1e-4, f"actor output not preserved: max|diff|={a_diff}"
    assert c_diff < 1e-4, f"critic output not preserved: max|diff|={c_diff}"

    # --- optimizer: param COUNT unchanged, so source param_groups still valid.
    #     Clear the per-param state (stale moments for the old input width) and
    #     reset iter -> fresh-optimizer warm-start semantics.
    src_opt = ck.get("optimizer_state_dict", {})
    tgt_opt = {"state": {}, "param_groups": src_opt.get("param_groups", [])}
    n_params_groups = len(tgt_opt["param_groups"])
    n_params = sum(len(g.get("params", [])) for g in tgt_opt["param_groups"])

    out = {
        "model_state_dict": tgt_sd,
        "optimizer_state_dict": tgt_opt,
        "iter": 0,
        "infos": ck.get("infos", {}),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)

    # --- report
    print("warm-start checkpoint written:")
    print(f"  src: {src_path}  (iter {ck.get('iter')})")
    print(f"  out: {out_path}")
    print(f"  actor.0.weight : {tuple(src_sd['actor.0.weight'].shape)} -> "
          f"{tuple(tgt_sd['actor.0.weight'].shape)}  (cols 0:{SRC_ACTOR_IN} copied, rest zero)")
    print(f"  critic.0.weight: {tuple(src_sd['critic.0.weight'].shape)} -> "
          f"{tuple(tgt_sd['critic.0.weight'].shape)}  (cols 0:{SRC_CRITIC_IN} copied, rest zero)")
    print(f"  verbatim tensors: {len(src_sd) - 2} (all but the two input weights)")
    print(f"  optimizer: {n_params_groups} group(s), {n_params} params, state cleared")
    print(f"  function-preservation: max|actor diff|={a_diff:.2e}, "
          f"max|critic diff|={c_diff:.2e}  (both < 1e-4; fp32 noise)")
    print(f"  iter reset to 0")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    build_warmstart(args.src, args.out)
