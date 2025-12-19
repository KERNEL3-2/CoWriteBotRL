"""
E0509 IK 환경 V3 학습 스크립트

펜 축 기준 접근 방식 학습
- perpendicular_dist: 펜 축에서 벗어난 거리 페널티
- 충돌 페널티
- 단계 체류 페널티

사용법:
    # 기본 실행 (headless 모드)
    python train_ik_v3.py --headless --num_envs 4096

    # GUI 모드로 실행
    python train_ik_v3.py --num_envs 64

    # 체크포인트에서 이어서 학습
    python train_ik_v3.py --headless --num_envs 4096 --checkpoint /path/to/model.pt

주의:
    - 학습은 별도 터미널에서 실행해야 합니다 (Claude 터미널은 타임아웃 있음)
    - GPU 메모리에 따라 num_envs 조절 필요
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V3 환경 학습")
parser.add_argument("--num_envs", type=int, default=4096, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=5000, help="최대 학습 반복 횟수")
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.e0509_ik_env_v3 import E0509IKEnvV3, E0509IKEnvV3Cfg

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL 설정
# =============================================================================
@configclass
class E0509IKV3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """E0509 IK V3 환경 PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "e0509_ik_v3"
    run_name = "pen_axis_approach"

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
    """메인 학습 함수"""

    # =============================================================================
    # 환경 설정
    # =============================================================================
    env_cfg = E0509IKEnvV3Cfg()
    env_cfg.scene.num_envs = args.num_envs

    env = E0509IKEnvV3(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    agent_cfg = E0509IKV3PPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations

    # =============================================================================
    # Runner 생성
    # =============================================================================
    log_dir = "./pen_grasp_rl/logs/e0509_ik_v3"
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"체크포인트 로드: {args.checkpoint}")
        runner.load(args.checkpoint)

    # =============================================================================
    # 학습 시작
    # =============================================================================
    print("=" * 70)
    print("E0509 IK V3 환경 강화학습 시작 (펜 축 기준 접근)")
    print("=" * 70)
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space} (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 70)
    print("V3 핵심 변경사항:")
    print("  - 펜 축 기준 접근 (perpendicular_dist 사용)")
    print("  - 충돌 페널티 추가")
    print("  - 단계 체류 페널티 추가")
    print("=" * 70)
    print("단계 (Phase):")
    print("  1. APPROACH: 펜 축 방향에서 pre-grasp 위치로 접근")
    print("  2. ALIGN: 위치 유지 + 자세 정렬")
    print("  3. DESCEND: 정렬 유지하며 캡으로 하강")
    print("  4. GRASP: 그리퍼 닫기")
    print("=" * 70)
    print("전환 조건:")
    print("  APPROACH → ALIGN: perp_dist < 3cm & axis_dist < 8cm")
    print("  ALIGN → DESCEND: dot < -0.95")
    print("  DESCEND → GRASP: dist_cap < 2cm & dot < -0.95")
    print("  SUCCESS: dist_cap < 1.5cm & dot < -0.95 & gripper_closed")
    print("=" * 70)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n학습 완료!")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
