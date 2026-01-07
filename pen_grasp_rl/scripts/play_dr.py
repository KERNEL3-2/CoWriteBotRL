"""
E0509 Domain Randomization 환경 테스트 스크립트

학습된 모델을 시각화하면서 테스트합니다.

사용법:
    # 모델 테스트
    python play_dr.py --checkpoint ./pen_grasp_rl/logs/e0509_dr/model_5000.pt

    # 환경 개수 조절
    python play_dr.py --checkpoint /path/to/model.pt --num_envs 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 DR 환경 테스트")
parser.add_argument("--num_envs", type=int, default=50, help="병렬 환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="테스트할 체크포인트")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.e0509_dr_env import E0509DREnv, E0509DREnvCfg_PLAY

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL 설정 (학습 설정과 동일해야 함)
# =============================================================================
@configclass
class E0509DRPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """E0509 DR 환경 PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "e0509_dr"
    run_name = "pen_grasp_dr"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )


def main():
    """메인 테스트 함수"""

    # =============================================================================
    # 환경 설정 (PLAY 설정 - DR 비활성화)
    # =============================================================================
    env_cfg = E0509DREnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    env = E0509DREnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정 (학습 시와 동일)
    # =============================================================================
    agent_cfg = E0509DRPPORunnerCfg()

    # =============================================================================
    # Runner 생성 및 모델 로드
    # =============================================================================
    log_dir = "./pen_grasp_rl/logs/e0509_dr_play"
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    print(f"체크포인트 로드: {args.checkpoint}")
    runner.load(args.checkpoint)

    # =============================================================================
    # 테스트 실행
    # =============================================================================
    print("=" * 70)
    print("E0509 Domain Randomization 환경 테스트")
    print("=" * 70)
    print(f"  체크포인트: {args.checkpoint}")
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  Domain Randomization: OFF (테스트 모드)")
    print("=" * 70)

    # Evaluation 모드로 실행
    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs, _ = env.get_observations()
    success_count = 0
    total_episodes = 0
    step = 0

    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)

        obs, rewards, dones, infos = env.step(actions)

        # 성공 카운트
        if "log" in infos and "Metrics/success_rate" in infos["log"]:
            success_rate = infos["log"]["Metrics/success_rate"]
            if step % 100 == 0:
                print(f"Step {step}: Success rate = {success_rate:.2%}")

        step += 1

        if step >= 10000:  # 최대 10000 스텝
            break

    print("\n테스트 완료!")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
