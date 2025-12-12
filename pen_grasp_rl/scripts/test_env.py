"""
Test environment setup - verify pen model, collision, and markers
Runs with random actions to check:
1. New pen.usd model is visible
2. Pen collision works (gripper can push pen)
3. Markers (red/green/blue/yellow) are displayed
4. Termination triggers when pen is displaced > 15cm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test pen grasp environment setup")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg, PEN_LENGTH

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def quat_rotate_vector(quat, vec):
    """Rotate vector by quaternion (w, x, y, z format)."""
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    t = 2.0 * torch.stack([
        q_y * vec[:, 2] - q_z * vec[:, 1],
        q_z * vec[:, 0] - q_x * vec[:, 2],
        q_x * vec[:, 1] - q_y * vec[:, 0]
    ], dim=-1)
    result = vec + q_w.unsqueeze(-1) * t + torch.stack([
        q_y * t[:, 2] - q_z * t[:, 1],
        q_z * t[:, 0] - q_x * t[:, 2],
        q_x * t[:, 1] - q_y * t[:, 0]
    ], dim=-1)
    return result


def main():
    # Create environment
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = PenGraspEnv(cfg=env_cfg)

    # Create visual markers (z-axis only)
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PenMarkers",
        markers={
            "pen_axis": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),  # Blue - pen z-axis
            ),
            "gripper_axis": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # Yellow - gripper z-axis
            ),
        }
    )
    pen_markers = VisualizationMarkers(marker_cfg)

    AXIS_POINTS = 5
    AXIS_LENGTH = 0.10

    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    act_dim = env.action_manager.total_action_dim

    print("=" * 60)
    print("Environment Test - Random Actions")
    print("=" * 60)
    print(f"  Obs dim: {obs_dim}")
    print(f"  Act dim: {act_dim}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Pen length: {PEN_LENGTH * 100:.1f} cm")
    print("=" * 60)
    print("Markers:")
    print("  Blue   = Pen z-axis")
    print("  Yellow = Gripper z-axis")
    print("=" * 60)
    print("Press Ctrl+C to exit")
    print()

    obs, _ = env.reset()
    step_count = 0
    episode_count = 0
    termination_count = 0

    try:
        while simulation_app.is_running():
            # Random actions
            actions = torch.randn(args.num_envs, act_dim, device="cuda:0") * 0.5

            step_result = env.step(actions)
            # Handle different return formats
            if len(step_result) == 2:
                obs, extras = step_result
                rewards = extras.get("rewards", torch.zeros(args.num_envs, device="cuda:0"))
                dones = extras.get("dones", torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0"))
            else:
                obs, rewards, dones, truncated, info = step_result
            step_count += 1

            # Count terminations (pen displaced)
            if dones.any():
                termination_count += dones.sum().item()
                episode_count += dones.sum().item()

            # Update markers
            pen_pos = env.scene["pen"].data.root_pos_w
            pen_quat = env.scene["pen"].data.root_quat_w
            robot = env.scene["robot"]

            # Grasp point calculation
            l1_pos = robot.data.body_pos_w[:, 7, :]
            r1_pos = robot.data.body_pos_w[:, 8, :]
            l2_pos = robot.data.body_pos_w[:, 9, :]
            r2_pos = robot.data.body_pos_w[:, 10, :]
            base_center = (l1_pos + r1_pos) / 2.0
            tip_center = (l2_pos + r2_pos) / 2.0
            finger_dir = tip_center - base_center
            finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)
            grasp_point = base_center + finger_dir * 0.02

            # Cap position
            half_len = PEN_LENGTH / 2.0
            pen_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            pen_axis_world = quat_rotate_vector(pen_quat, pen_axis)
            cap_pos = pen_pos - pen_axis_world * half_len

            # Gripper z-axis
            link6_quat = robot.data.body_quat_w[:, 6, :]
            gripper_z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            gripper_z_world = quat_rotate_vector(link6_quat, gripper_z_axis)

            # Pen displacement from initial position
            pen_pos_local = pen_pos - env.scene.env_origins
            init_pos = torch.tensor([0.5, 0.0, 0.3], device=pen_pos.device)
            displacement = torch.norm(pen_pos_local - init_pos, dim=-1)

            # Build marker positions (z-axis markers only)
            num_envs = pen_pos.shape[0]
            markers_per_env = AXIS_POINTS * 2  # pen_axis + gripper_axis
            all_positions = torch.zeros((num_envs * markers_per_env, 3), device=pen_pos.device)
            marker_indices = []

            for i in range(num_envs):
                base_idx = i * markers_per_env

                # Pen z-axis markers (blue)
                for j in range(AXIS_POINTS):
                    t = (j + 1) / AXIS_POINTS * AXIS_LENGTH
                    all_positions[base_idx + j] = pen_pos[i] + pen_axis_world[i] * t
                    marker_indices.append(0)  # pen_axis

                # Gripper z-axis markers (yellow)
                for j in range(AXIS_POINTS):
                    t = (j + 1) / AXIS_POINTS * AXIS_LENGTH
                    all_positions[base_idx + AXIS_POINTS + j] = grasp_point[i] + gripper_z_world[i] * t
                    marker_indices.append(1)  # gripper_axis

            pen_markers.visualize(translations=all_positions, marker_indices=marker_indices)

            # Print status every 100 steps
            if step_count % 100 == 0:
                dist_to_cap = torch.norm(grasp_point - cap_pos, dim=-1)
                gripper_pos = robot.data.joint_pos[:, 6:10].mean(dim=-1)

                print(f"Step {step_count:5d} | Episodes: {episode_count} | Terminations: {termination_count}")
                print(f"  Reward: {rewards.mean().item():+.4f}")
                print(f"  Distance to cap: {dist_to_cap.mean().item()*100:.1f} cm")
                print(f"  Pen displacement: {displacement.mean().item()*100:.1f} cm (max: {displacement.max().item()*100:.1f})")
                print(f"  Gripper state: {gripper_pos.mean().item():.2f}")
                print()

    except KeyboardInterrupt:
        print("\nTest stopped by user")

    print("=" * 60)
    print(f"Final stats: {step_count} steps, {termination_count} terminations")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
