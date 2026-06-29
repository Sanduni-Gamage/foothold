"""Physics-only rollout of the surface policy (NO renderer -> works headless).

Runs the deterministic policy and logs, for every swing->stance landing across
all envs: (tile_row, tile_col, px_i, px_j, surface_class). Also dumps the real
per-tile label grid. An offline script then plots the foot landings coloured by
FLAT/LIPPED/UNSAFE over the true class map -> the spatial version of shot 3,
without needing the broken offscreen renderer.
"""
import argparse, os, sys
sys.path.insert(0, os.path.expanduser("~/unitree_rl_lab/scripts/rsl_rl"))
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Unitree-Go2-Surface")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--out", default=os.path.expanduser("~/projects/dogb/figures/sim_capture/landings.npz"))
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False                # <-- no renderer, no hang

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.tasks.locomotion.mdp.foothold_state import update_landing_events
from unitree_rl_lab.tasks.locomotion.mdp.surface_labels import surface_class_at


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                            use_fabric=True, entry_point_key="play_env_cfg_entry_point")
    try:
        env_cfg.commands.base_velocity.debug_vis = False
    except Exception:
        pass
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    resume_path = retrieve_file_path(args.checkpoint) if args.checkpoint else \
        get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)),
                            agent_cfg.load_run, agent_cfg.load_checkpoint)
    print("[data] checkpoint:", resume_path, flush=True)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=u.device)

    foot_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot"); foot_cfg.resolve(u.scene)
    robot = u.scene["robot"]; foot_ids = foot_cfg.body_ids

    obs = env.get_observations()
    if isinstance(obs, (tuple, list)): obs = obs[0]

    rows = []  # row,col,pxi,pxj,cls per landing
    for step in range(args.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, (tuple, list)): obs = obs[0]
        just = update_landing_events(u, foot_cfg)                 # (E,F)
        foot_w = robot.data.body_pos_w[:, foot_ids, :]            # (E,F,3)
        cls = surface_class_at(u, foot_w)                          # (E,F); sets _surface_* attrs
        gmin = u._surface_grid_min_w; tx = u._surface_tile_x_m; ty = u._surface_tile_y_m; hs = u._surface_h_scale
        xy = foot_w[..., :2]
        rel_x = xy[..., 0] - gmin[0]; rel_y = xy[..., 1] - gmin[1]
        col = torch.floor(rel_x / tx).long(); row = torch.floor(rel_y / ty).long()
        pxi = torch.floor((rel_x - col * tx) / hs).long()
        pxj = torch.floor((rel_y - row * ty) / hs).long()
        idx = torch.nonzero(just)
        for e, f in idx.tolist():
            rows.append([int(row[e, f]), int(col[e, f]), int(pxi[e, f]), int(pxj[e, f]), int(cls[e, f])])
        if step % 100 == 0:
            print(f"[data] step {step}  landings so far {len(rows)}", flush=True)

    rows = np.array(rows, dtype=np.int64) if rows else np.zeros((0, 5), np.int64)
    labels = u._surface_labels.detach().cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, landings=rows, labels=labels,
             h_scale=float(u._surface_h_scale), tile_x=float(u._surface_tile_x_m),
             tile_y=float(u._surface_tile_y_m))
    # quick class histogram
    if len(rows):
        names = {0: "FLAT", 1: "LIPPED", 2: "POCKET", 3: "UNSAFE", 255: "OOB"}
        uq, ct = np.unique(rows[:, 4], return_counts=True)
        dist = {names.get(int(k), int(k)): f"{v/len(rows)*100:.1f}%" for k, v in zip(uq, ct)}
        print(f"[data] {len(rows)} landings  class dist: {dist}", flush=True)
    print("[data] wrote", args.out, flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
