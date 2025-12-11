"""
Test script for Pen Grasp Environment
"""
import argparse
import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test pen grasping environment")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")

# Append AppLauncher arguments (includes --headless)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching
import torch
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg


def main():
    """Main test function."""

    # Environment config
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Create environment
    print("=" * 50)
    print("Creating environment...")
    print("=" * 50)
    env = PenGraspEnv(cfg=env_cfg)

    print(f"Environment created with {env.num_envs} envs")
    print(f"Observation dim: {env.observation_manager.group_obs_dim}")
    print(f"Action dim: {env.action_manager.total_action_dim}")

    # Run test loop
    print("\n" + "=" * 50)
    print("Running test loop...")
    print("=" * 50)

    obs, _ = env.reset()
    count = 0

    while simulation_app.is_running() and count < 500:
        with torch.inference_mode():
            # Random actions
            actions = torch.randn(env.num_envs, 7, device=env.device) * 0.1

            # Step environment (Isaac Lab returns obs_dict, extras)
            obs_dict, extras = env.step(actions)

            if count % 50 == 0:
                print(f"Step {count}")

            count += 1

    print("\nTest complete!")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
