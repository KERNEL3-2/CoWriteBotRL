"""
E0509 IK 환경 V8 학습 스크립트 (관절 제한 + 수직거리 7cm)

V8 주요 변경사항 (V7 대비):
- SUCCESS_DIST_TO_CAP: 3cm → 7cm
- 관절 제한 페널티 추가 (90% 넘으면 페널티)

사용법:
    # Level 0부터 시작 (펜 수직)
    python train_v8.py --headless --num_envs 4096 --level 0

    # Level 1 (10°) - Level 0 체크포인트에서 이어서
    python train_v8.py --headless --num_envs 4096 --level 1 --checkpoint /path/to/l0_model.pt

    # Fixed LR 모드 (발산 방지)
    python train_v8.py --headless --num_envs 4096 --level 0 --fixed_lr

주의:
    - 학습은 별도 터미널에서 실행해야 합니다 (Claude 터미널은 타임아웃 있음)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V8 환경 학습 (관절 제한 + 7cm)")
parser.add_argument("--num_envs", type=int, default=4096, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=5000, help="최대 학습 반복 횟수")
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트")
parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Curriculum Level (0: 수직, 1: 10°, 2: 20°, 3: 30°)")
parser.add_argument("--fixed_lr", action="store_true", help="Fixed Learning Rate 사용 (발산 방지)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.e0509_ik_env_v8 import (
    E0509IKEnvV8,
    E0509IKEnvV8Cfg,
    E0509IKEnvV8Cfg_L0,
    E0509IKEnvV8Cfg_L1,
    E0509IKEnvV8Cfg_L2,
    E0509IKEnvV8Cfg_L3,
    CURRICULUM_TILT_MAX,
    SUCCESS_DIST_TO_CAP,
)

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL 설정
# =============================================================================
@configclass
class E0509IKV8PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """E0509 IK V8 환경 PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "e0509_ik_v8"
    run_name = "curriculum_l0"

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


@configclass
class E0509IKV8PPORunnerCfg_FixedLR(E0509IKV8PPORunnerCfg):
    """Fixed LR 버전 (발산 방지용)"""
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def get_env_cfg_for_level(level: int):
    """Curriculum Level에 맞는 환경 설정 반환"""
    cfg_map = {
        0: E0509IKEnvV8Cfg_L0,
        1: E0509IKEnvV8Cfg_L1,
        2: E0509IKEnvV8Cfg_L2,
        3: E0509IKEnvV8Cfg_L3,
    }
    return cfg_map.get(level, E0509IKEnvV8Cfg_L0)()


def main():
    """메인 학습 함수"""

    # =============================================================================
    # 환경 설정
    # =============================================================================
    env_cfg = get_env_cfg_for_level(args.level)
    env_cfg.scene.num_envs = args.num_envs

    env = E0509IKEnvV8(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    if args.fixed_lr:
        agent_cfg = E0509IKV8PPORunnerCfg_FixedLR()
        lr_mode = "Fixed (1e-4)"
    else:
        agent_cfg = E0509IKV8PPORunnerCfg()
        lr_mode = "Adaptive (3e-4)"

    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = f"curriculum_l{args.level}"

    # =============================================================================
    # Runner 생성
    # =============================================================================
    log_dir = f"./pen_grasp_rl/logs/e0509_ik_v8_l{args.level}"
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
    tilt_deg = CURRICULUM_TILT_MAX[args.level] * 180 / 3.14159
    print("=" * 70)
    print("E0509 IK V8 환경 강화학습 시작 (관절 제한 + 7cm)")
    print("=" * 70)
    print(f"  Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}°)")
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  Learning Rate: {lr_mode}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space} (Δx, Δy, Δz)")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 70)
    print("V8 핵심 변경사항 (V7 대비):")
    print(f"  - 성공 거리: 3cm → {SUCCESS_DIST_TO_CAP*100:.0f}cm")
    print("  - 관절 제한 페널티 추가 (90% 이상이면 페널티)")
    print("=" * 70)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n학습 완료!")
    print(f"모델 저장 위치: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
