"""Headless Isaac Sim capture for the Go2 surface (foothold-selection) task.

Adds the surface-class foot-contact visualisation that the project lacks:
on every swing->stance landing it drops a sphere at the foot, coloured by the
FLAT / LIPPED / UNSAFE class under the foot (reusing surface_class_at + the
same update_landing_events the reward uses). Renders the side-on viewport to
high-res PNGs. Deterministic policy (inference mean), headless, GPU-selectable.

Run (GPU 1):
  cd ~/projects/dogb && CUDA_VISIBLE_DEVICES=1 \
    python capture_surface_shots.py --checkpoint <model.pt> --num_steps 450
"""
import argparse, os, sys

# cli_args lives next to play.py
sys.path.insert(0, os.path.expanduser("~/unitree_rl_lab/scripts/rsl_rl"))
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Unitree-Go2-Surface")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=450)
parser.add_argument("--warmup", type=int, default=60, help="steps before saving (let gait settle)")
parser.add_argument("--outdir", default=os.path.expanduser("~/projects/dogb/figures/sim_capture"))
parser.add_argument("--res", type=int, nargs=2, default=[1600, 900])
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs with the sim app live ----
import numpy as np
import torch
from PIL import Image
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.tasks.locomotion.mdp.foothold_state import update_landing_events
from unitree_rl_lab.tasks.locomotion.mdp.surface_labels import (
    surface_class_at, FLAT, LIPPED, UNSAFE,
)

os.makedirs(args.outdir, exist_ok=True)


def save(frame, name):
    if frame is None:
        print(f"[capture] render returned None for {name}"); return
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(os.path.join(args.outdir, name))
    print(f"[capture] wrote {name}  shape={arr.shape}")


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                            use_fabric=True, entry_point_key="play_env_cfg_entry_point")
    try:
        env_cfg.viewer.resolution = (args.res[0], args.res[1])
    except Exception as e:
        print("[capture] could not set viewer resolution:", e)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)

    if args.checkpoint:
        resume_path = retrieve_file_path(args.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print("[capture] checkpoint:", resume_path)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=u.device)

    # ---- NEW: surface-class contact markers (FLAT green / LIPPED orange / UNSAFE red) ----
    markers = None
    try:
        mcfg = VisualizationMarkersCfg(prim_path="/Visuals/SurfaceContacts", markers={
            "flat":   sim_utils.SphereCfg(radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.17, 0.63, 0.17))),
            "lipped": sim_utils.SphereCfg(radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.50, 0.05))),
            "unsafe": sim_utils.SphereCfg(radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.84, 0.15, 0.15))),
        })
        markers = VisualizationMarkers(mcfg)
        print("[capture] surface-class markers created")
    except Exception as e:
        print("[capture] WARN marker creation failed (will still render robot+terrain):", e)

    foot_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot"); foot_cfg.resolve(u.scene)
    robot = u.scene["robot"]
    foot_ids = foot_cfg.body_ids
    CLASS_IDX = {int(FLAT): 0, int(LIPPED): 1, int(UNSAFE): 2}

    acc_pos, acc_idx = [], []

    obs = env.get_observations()
    if isinstance(obs, (tuple, list)):
        obs = obs[0]

    saved = 0
    for step in range(args.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, (tuple, list)):
                obs = obs[0]

        # --- accumulate class-coloured contact markers (env 0) ---
        if markers is not None:
            try:
                just = update_landing_events(u, foot_cfg)            # (E,F) bool
                foot_w = robot.data.body_pos_w[:, foot_ids, :]        # (E,F,3)
                cls = surface_class_at(u, foot_w)                     # (E,F)
                landed = torch.nonzero(just[0]).flatten()
                for f in landed.tolist():
                    c = int(cls[0, f].item())
                    if c in CLASS_IDX:
                        acc_pos.append(foot_w[0, f].detach().clone())
                        acc_idx.append(CLASS_IDX[c])
                if acc_pos:
                    trans = torch.stack(acc_pos[-400:])
                    idxs = torch.tensor(acc_idx[-400:], device=u.device, dtype=torch.long)
                    markers.visualize(translations=trans, marker_indices=idxs)
            except Exception as e:
                if step == args.warmup:
                    print("[capture] WARN marker update failed:", e)
                markers = markers  # keep trying is fine; errors are per-step caught

        # --- save frames in the second half once a trail exists ---
        if step >= args.warmup and (step % 60 == 0 or step == args.num_steps - 1):
            frame = env.render()
            save(frame, f"frame_{step:04d}_contacts{len(acc_pos)}.png")
            saved += 1

    print(f"[capture] done. saved {saved} frames, total landings logged: {len(acc_pos)}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
