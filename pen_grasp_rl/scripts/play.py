"""
Test trained policy for Pen Grasp RL
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test trained pen grasping policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch with GUI (no --headless)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg, PEN_LENGTH

# Debug visualization
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG


class SimplePolicy(nn.Module):
    """Simple MLP policy for inference."""
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256, 128]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def quat_rotate_vector(quat, vec):
    """Rotate vector by quaternion (w, x, y, z format)."""
    # quat: (N, 4), vec: (N, 3)
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Cross product: q_xyz x vec
    t = 2.0 * torch.stack([
        q_y * vec[:, 2] - q_z * vec[:, 1],
        q_z * vec[:, 0] - q_x * vec[:, 2],
        q_x * vec[:, 1] - q_y * vec[:, 0]
    ], dim=-1)

    # Result: vec + q_w * t + q_xyz x t
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

    # Setup debug markers for pen tip (green) and cap (red)
    tip_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/tip_markers",
        markers={
            "sphere": SPHERE_MARKER_CFG.replace(
                visual_material={"diffuse_color": (0.0, 1.0, 0.0)},  # Green
            ).replace(
                scale=(0.015, 0.015, 0.015),  # 15mm diameter
            ),
        },
    )
    cap_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/cap_markers",
        markers={
            "sphere": SPHERE_MARKER_CFG.replace(
                visual_material={"diffuse_color": (1.0, 0.0, 0.0)},  # Red
            ).replace(
                scale=(0.015, 0.015, 0.015),  # 15mm diameter
            ),
        },
    )
    tip_markers = VisualizationMarkers(tip_marker_cfg)
    cap_markers = VisualizationMarkers(cap_marker_cfg)

    # Get observation and action dimensions
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    act_dim = env.action_manager.total_action_dim

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)

    # Create simple policy network
    policy = SimplePolicy(obs_dim, act_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    # Extract actor weights from checkpoint
    model_state = checkpoint["model_state_dict"]

    # Map weights to our simple policy
    policy_state = {}
    layer_idx = 0
    for key, value in model_state.items():
        if "actor" in key and "weight" in key:
            policy_state[f"network.{layer_idx}.weight"] = value
            layer_idx += 1
        elif "actor" in key and "bias" in key:
            policy_state[f"network.{layer_idx - 1}.bias"] = value

    # Try to load weights, if structure doesn't match, use raw inference
    try:
        policy.load_state_dict(policy_state, strict=False)
        print("Loaded policy weights successfully")
    except Exception as e:
        print(f"Could not load exact weights: {e}")
        print("Using random policy for visualization")

    policy.eval()

    print("=" * 50)
    print("Testing trained policy...")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Act dim: {act_dim}")
    print(f"  Num envs: {args.num_envs}")
    print("=" * 50)
    print("Pen markers:")
    print("  - GREEN sphere: tip (a) - writing end")
    print("  - RED sphere: cap (b) - gripper target")
    print("=" * 50)
    print("Press Ctrl+C to exit")

    # Run test loop
    obs, _ = env.reset()
    policy_obs = obs["policy"]

    step_count = 0
    device = env.device
    half_len = PEN_LENGTH / 2.0

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(policy_obs)
                # Clamp actions to reasonable range
                actions = torch.clamp(actions, -1.0, 1.0)

            obs, rewards, dones, truncated, info = env.step(actions)
            policy_obs = obs["policy"]
            step_count += 1

            # Update pen tip/cap markers
            pen_pos = env.scene["pen"].data.root_pos_w  # (N, 3)
            pen_quat = env.scene["pen"].data.root_quat_w  # (N, 4) w,x,y,z

            # Pen local z-axis offset
            local_tip_offset = torch.tensor([[0.0, 0.0, half_len]], device=device).expand(pen_pos.shape[0], -1)
            local_cap_offset = torch.tensor([[0.0, 0.0, -half_len]], device=device).expand(pen_pos.shape[0], -1)

            # Rotate offsets by pen orientation
            tip_pos = pen_pos + quat_rotate_vector(pen_quat, local_tip_offset)
            cap_pos = pen_pos + quat_rotate_vector(pen_quat, local_cap_offset)

            # Visualize markers
            tip_markers.visualize(tip_pos)
            cap_markers.visualize(cap_pos)

            # Print mean reward occasionally
            if step_count % 100 == 0:
                print(f"Step {step_count}, Mean reward: {rewards.mean().item():.4f}")

    except KeyboardInterrupt:
        print("\nTest stopped by user")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
