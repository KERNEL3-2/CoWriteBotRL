"""
E0509 IK 환경 V5 학습 스크립트 (Curriculum Learning)

V5 주요 변경사항:
- TCP DESCEND 버그 수정 (월드 Z축 방향 하강)
- Curriculum Learning: 펜 각도 점진적 증가
- 자세 보정 속도 강화

사용법:
    # Level 0부터 시작 (펜 수직)
    python train_ik_v5.py --headless --num_envs 4096 --level 0

    # Level 1 (10°) - Level 0 체크포인트에서 이어서
    python train_ik_v5.py --headless --num_envs 4096 --level 1 --checkpoint /path/to/l0_model.pt

    # Level 2 (20°) - Level 1 체크포인트에서 이어서
    python train_ik_v5.py --headless --num_envs 4096 --level 2 --checkpoint /path/to/l1_model.pt

    # Level 3 (30°) - 최종 목표
    python train_ik_v5.py --headless --num_envs 4096 --level 3 --checkpoint /path/to/l2_model.pt

    # Fixed LR 모드 (발산 방지)
    python train_ik_v5.py --headless --num_envs 4096 --level 0 --fixed_lr

Curriculum Learning 전략:
    Level 0: 펜 수직 (0°)   → 기본 동작 학습
    Level 1: 펜 10° 기울기  → 약간의 변형 적응
    Level 2: 펜 20° 기울기  → 중간 난이도
    Level 3: 펜 30° 기울기  → 최종 목표

주의:
    - 학습은 별도 터미널에서 실행해야 합니다 (Claude 터미널은 타임아웃 있음)
    - GPU 메모리에 따라 num_envs 조절 필요
    - 체크포인트에서 이어서 학습할 때는 같은 level에서는 괜찮지만,
      다음 level로 넘어갈 때는 --fixed_lr 사용 권장
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V5 환경 학습 (Curriculum Learning)")
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
from envs.e0509_ik_env_v5 import (
    E0509IKEnvV5,
    E0509IKEnvV5Cfg,
    E0509IKEnvV5Cfg_L0,
    E0509IKEnvV5Cfg_L1,
    E0509IKEnvV5Cfg_L2,
    E0509IKEnvV5Cfg_L3,
    CURRICULUM_TILT_MAX,
)

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass




# =============================================================================
# RSL-RL 설정
# =============================================================================
@configclass
class E0509IKV5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """E0509 IK V5 환경 PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "e0509_ik_v5"
    run_name = "curriculum_l0"  # 실행 시 동적으로 변경됨

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
class E0509IKV5PPORunnerCfg_FixedLR(E0509IKV5PPORunnerCfg):
    """Fixed LR 버전 (발산 방지용)"""
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,    # 3e-4 → 1e-4 (더 작게)
        schedule="fixed",      # adaptive → fixed
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def get_env_cfg_for_level(level: int):
    """Curriculum Level에 맞는 환경 설정 반환"""
    cfg_map = {
        0: E0509IKEnvV5Cfg_L0,
        1: E0509IKEnvV5Cfg_L1,
        2: E0509IKEnvV5Cfg_L2,
        3: E0509IKEnvV5Cfg_L3,
    }
    return cfg_map.get(level, E0509IKEnvV5Cfg_L0)()


def main():
    """메인 학습 함수"""

    # =============================================================================
    # 환경 설정
    # =============================================================================
    env_cfg = get_env_cfg_for_level(args.level)
    env_cfg.scene.num_envs = args.num_envs

    env = E0509IKEnvV5(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    if args.fixed_lr:
        agent_cfg = E0509IKV5PPORunnerCfg_FixedLR()
        lr_mode = "Fixed (1e-4)"
    else:
        agent_cfg = E0509IKV5PPORunnerCfg()
        lr_mode = "Adaptive (3e-4)"

    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = f"curriculum_l{args.level}"

    # =============================================================================
    # Runner 생성
    # =============================================================================
    log_dir = f"./pen_grasp_rl/logs/e0509_ik_v5_l{args.level}"
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
    print("E0509 IK V5 환경 강화학습 시작 (Curriculum Learning)")
    print("=" * 70)
    print(f"  Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}°)")
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  Learning Rate: {lr_mode}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space} (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 70)
    print("V5 핵심 변경사항:")
    print("  - TCP DESCEND 버그 수정: 월드 Z축 방향 하강")
    print("  - 자세 보정 속도 강화 (0.5 → 1.5)")
    print("  - Curriculum Learning: Level 0(0°) → 1(10°) → 2(20°) → 3(30°)")
    print("=" * 70)
    print("Curriculum Learning 전략:")
    print("  Level 0: 펜 수직 (0°)   → 기본 동작 학습")
    print("  Level 1: 펜 10° 기울기  → 약간의 변형 적응")
    print("  Level 2: 펜 20° 기울기  → 중간 난이도")
    print("  Level 3: 펜 30° 기울기  → 최종 목표")
    print("=" * 70)
    print("단계 (Phase) - 6단계:")
    print("  0. APPROACH: RL - 펜 축 방향에서 접근")
    print("  1. ALIGN: RL - 대략적 자세 정렬 (dot < -0.85)")
    print("  2. FINE_ALIGN: TCP - 정밀 자세 정렬 (dot → -0.98)")
    print("  3. DESCEND: TCP - 월드 Z축 하강 + 자세 유지")
    print("  4. GRASP: TCP - 그리퍼 닫기 (Good Grasp + 30스텝)")
    print("  5. LIFT: TCP - 펜 5cm 들어올리기 → 성공")
    print("=" * 70)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n학습 완료!")
    print(f"모델 저장 위치: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
