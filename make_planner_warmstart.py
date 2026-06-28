#!/usr/bin/env python3
"""Build a warm-start checkpoint for Unitree-Go2-FootholdPlanner from the
surface task's best checkpoint (WS-3b-flat35 model_16800.pt by default).

Shape transfer:

  Source (surface task)            Target (planner task)
  --------------------------------  --------------------------------
  actor.0.weight  (512, 244)       (512, 252)   8 new cols zero-init
  critic.0.weight (512, 259)       (512, 267)   8 new cols zero-init
  actor.6.weight  ( 12, 128)       ( 20, 128)   8 new action rows zero-init
  actor.6.bias    ( 12,)           ( 20,)       8 new action biases zero-init
  std             ( 12,)           ( 20,)       8 new noise stds = source mean
  (all other tensors copy verbatim)

Zero-init on the new action head means the planner's foot-target offset
output is zero at iter 0. With FootTargetAction's raw_scale=0.15 and clip
±0.30 around nominal hip, that means target = nominal hip xy — roughly
where the gait was placing feet anyway. So the warm-started planner is
function-near-identical to the surface walker at iteration 0; training then
pulls targets toward FLAT and the controller learns to track them.

For the std on the new action dims, we initialize to the MEAN of the
source's std (across the 12 existing dims). This gives the planner
exploration on its target head similar to what the joint head had at
convergence, rather than starting at the much larger init_noise_std=1.0
(which would scatter targets across ±0.30 m randomly).
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

# --- planner task obs/action dims (must match velocity_planner_env_cfg.py)
SRC_ACTOR_IN = 244
SRC_CRITIC_IN = 259
SRC_ACTIONS = 12
TGT_ACTIONS = 20  # +8 foot-target action head
# last_action obs auto-grows with action space: +8. foot_targets obs: +8.
# So the policy obs grows by 16, not 8 as initially estimated.
TGT_ACTOR_IN = 244 + 16   # = 260
TGT_CRITIC_IN = 259 + 16  # = 275

DEFAULT_SRC = (
    "/home/anyone/projects/dogb/logs/rsl_rl/unitree_go2_surface/"
    "2026-05-31_18-13-53/model_16800.pt"
)
DEFAULT_OUT = (
    "/home/anyone/projects/dogb/logs/rsl_rl/unitree_go2_footholdplanner/"
    "warmstart_surface_3b_flat35_16800/model_planner_warmstart.pt"
)


def _forward_mlp(state: dict, prefix: str, x: torch.Tensor) -> torch.Tensor:
    out = x
    idx = 0
    while f"{prefix}.{idx}.weight" in state:
        w = state[f"{prefix}.{idx}.weight"]
        b = state[f"{prefix}.{idx}.bias"]
        out = F.linear(out, w, b)
        if f"{prefix}.{idx + 2}.weight" in state:
            out = F.elu(out)
        idx += 2
    return out


def build_warmstart(src_path: str, out_path: str) -> None:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    src_sd = ck["model_state_dict"]

    # --- sanity: confirm source dims
    a_in = src_sd["actor.0.weight"].shape[1]
    c_in = src_sd["critic.0.weight"].shape[1]
    a_out = src_sd["actor.6.weight"].shape[0]
    assert a_in == SRC_ACTOR_IN, f"src actor in {a_in} != {SRC_ACTOR_IN}"
    assert c_in == SRC_CRITIC_IN, f"src critic in {c_in} != {SRC_CRITIC_IN}"
    assert a_out == SRC_ACTIONS, f"src action dim {a_out} != {SRC_ACTIONS}"

    # --- build target model_state_dict
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
        elif k == "actor.6.weight":
            # (12, 128) -> (20, 128). Action rows [12:20] zero -> target = nominal hip.
            w = torch.zeros(TGT_ACTIONS, v.shape[1], dtype=v.dtype)
            w[:SRC_ACTIONS, :] = v
            tgt_sd[k] = w
        elif k == "actor.6.bias":
            b = torch.zeros(TGT_ACTIONS, dtype=v.dtype)
            b[:SRC_ACTIONS] = v
            tgt_sd[k] = b
        elif k == "std":
            # (12,) -> (20,). New action dims get the *mean* of the converged
            # source std, not init_noise_std=1.0 (which would scatter the
            # planner targets randomly across ±0.30 m).
            s = torch.zeros(TGT_ACTIONS, dtype=v.dtype)
            s[:SRC_ACTIONS] = v
            s[SRC_ACTIONS:] = v.mean()
            tgt_sd[k] = s
        else:
            tgt_sd[k] = v.clone()

    # --- function preservation check: feed proprio+zero padding, confirm
    #     the first 12 action outputs match the source's actions.
    torch.manual_seed(0)
    proprio_src = torch.randn(8, SRC_ACTOR_IN)
    crit_src = torch.randn(8, SRC_CRITIC_IN)
    proprio_tgt = F.pad(proprio_src, (0, TGT_ACTOR_IN - SRC_ACTOR_IN))
    crit_tgt = F.pad(crit_src, (0, TGT_CRITIC_IN - SRC_CRITIC_IN))
    a_src = _forward_mlp(src_sd, "actor", proprio_src)
    a_tgt_full = _forward_mlp(tgt_sd, "actor", proprio_tgt)
    a_tgt_joint = a_tgt_full[:, :SRC_ACTIONS]
    a_tgt_target_head = a_tgt_full[:, SRC_ACTIONS:]
    c_src = _forward_mlp(src_sd, "critic", crit_src)
    c_tgt = _forward_mlp(tgt_sd, "critic", crit_tgt)

    a_joint_diff = (a_src - a_tgt_joint).abs().max().item()
    a_target_head_max = a_tgt_target_head.abs().max().item()
    c_diff = (c_src - c_tgt).abs().max().item()

    assert a_joint_diff < 1e-4, (
        f"joint action head not preserved: max|diff|={a_joint_diff}"
    )
    assert a_target_head_max < 1e-6, (
        f"target action head should be zero at warm-start, got max|val|={a_target_head_max}"
    )
    assert c_diff < 1e-4, f"critic not preserved: max|diff|={c_diff}"

    # --- optimizer state: param count UNCHANGED (only shapes grew). Reuse
    #     source param_groups with cleared state. Same trick as the surface
    #     warm-start; same reasoning.
    src_opt = ck.get("optimizer_state_dict", {})
    tgt_opt = {"state": {}, "param_groups": src_opt.get("param_groups", [])}
    n_groups = len(tgt_opt["param_groups"])
    n_params = sum(len(g.get("params", [])) for g in tgt_opt["param_groups"])

    out = {
        "model_state_dict": tgt_sd,
        "optimizer_state_dict": tgt_opt,
        "iter": 0,
        "infos": ck.get("infos", {}),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)

    print("planner warm-start checkpoint written:")
    print(f"  src: {src_path}  (src iter {ck.get('iter')})")
    print(f"  out: {out_path}")
    print(f"  actor.0.weight  : (512, {SRC_ACTOR_IN}) -> "
          f"(512, {TGT_ACTOR_IN})  [+8 obs cols zero]")
    print(f"  actor.6.weight  : ({SRC_ACTIONS}, 128) -> "
          f"({TGT_ACTIONS}, 128)  [+8 action rows zero -> target=nominal hip]")
    print(f"  critic.0.weight : (512, {SRC_CRITIC_IN}) -> "
          f"(512, {TGT_CRITIC_IN})  [+8 obs cols zero]")
    print(f"  std             : ({SRC_ACTIONS},) -> ({TGT_ACTIONS},)  "
          f"[+8 new = mean(src std) = {tgt_sd['std'][SRC_ACTIONS:].mean().item():.3f}]")
    print(f"  verbatim tensors: {len(src_sd) - 5}")
    print(f"  optimizer       : {n_groups} group(s), {n_params} params, state cleared")
    print(f"  fp check        : joint diff={a_joint_diff:.2e}, "
          f"target head max={a_target_head_max:.2e}, critic diff={c_diff:.2e}")
    print(f"  iter reset to 0")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    build_warmstart(args.src, args.out)
