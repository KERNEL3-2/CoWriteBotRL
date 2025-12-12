"""
Training script for Pen Grasp RL
"""
import argparse
import os
import sys
from datetime import datetime

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train pen grasping policy")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=3000, help="Max training iterations")

# Append AppLauncher arguments (includes --headless)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching
import torch
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg

# Use RSL-RL for training
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    """Main training function."""

    # Environment config
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Create environment
    env = PenGraspEnv(cfg=env_cfg)

    # PPO configuration
    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=42,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        save_interval=100,
        experiment_name="pen_grasp",
        run_name="ppo_run",
        logger="tensorboard",
        obs_groups={"policy": ["policy"]},
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
            actor_obs_normalization=False,
            critic_obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./logs/pen_grasp/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Train
    print("=" * 50)
    print("Starting training...")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Log dir: {log_dir}")
    print("=" * 50)

    runner.learn(num_learning_iterations=args.max_iterations)

    # Save final model
    runner.save("final_model")
    print("Training complete!")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
