"""
E0509 OSC 환경 학습 스크립트 (Operational Space Control)

OSC vs IK 차이점:
- IK: 관절 위치 타겟 → set_joint_position_target()
- OSC: 관절 토크 타겟 → set_joint_effort_target()

OSC 장점:
- Sim2Real gap 감소 (임피던스 제어)
- 접촉 시 부드러운 반응
- 힘 제어 가능

사용법:
    # 기본 학습 (stiffness=150)
    python train_osc.py --headless --num_envs 4096

    # Soft 모드 (stiffness=60, 부드러운 동작)
    python train_osc.py --headless --num_envs 4096 --soft

    # 커스텀 설정 (명령행으로 직접 지정)
    python train_osc.py --headless --num_envs 4096 --stiffness 80 --action_scale 0.04 --hold_steps 15

    # Soft + 일부 오버라이드
    python train_osc.py --headless --num_envs 4096 --soft --hold_steps 20

    # Fixed LR (안정적 학습)
    python train_osc.py --headless --num_envs 4096 --fixed_lr

    # 체크포인트에서 이어서
    python train_osc.py --headless --num_envs 4096 --checkpoint /path/to/model.pt

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
parser = argparse.ArgumentParser(description="E0509 OSC 환경 학습")
parser.add_argument("--num_envs", type=int, default=4096, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=5000, help="최대 학습 반복 횟수")
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트")
parser.add_argument("--fixed_lr", action="store_true", help="Fixed Learning Rate 사용 (발산 방지)")
parser.add_argument("--soft", action="store_true", help="Soft 모드 (stiffness=60, 부드러운 동작)")

# 환경 설정 오버라이드
parser.add_argument("--stiffness", type=float, default=None, help="OSC stiffness (Default: 150, Soft: 60)")
parser.add_argument("--action_scale", type=float, default=None, help="Action scale (Default: 0.05, Soft: 0.03)")
parser.add_argument("--hold_steps", type=int, default=None, help="성공 유지 스텝 (Default: 30, Soft: 10)")
parser.add_argument("--log_dir", type=str, default=None, help="로그 저장 경로 (미지정 시 자동 설정)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.e0509_osc_env import E0509OSCEnv, E0509OSCEnvCfg, E0509OSCEnvCfg_Soft

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL 설정
# =============================================================================
@configclass
class E0509OSCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """E0509 OSC 환경 PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "e0509_osc"
    run_name = "osc_run"

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
class E0509OSCPPORunnerCfg_FixedLR(E0509OSCPPORunnerCfg):
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


def main():
    """메인 학습 함수"""

    # =============================================================================
    # 환경 설정
    # =============================================================================
    if args.soft:
        env_cfg = E0509OSCEnvCfg_Soft()
        env_mode = "Soft"
    else:
        env_cfg = E0509OSCEnvCfg()
        env_mode = "Default"

    env_cfg.scene.num_envs = args.num_envs

    # 명령행 인자로 설정 오버라이드
    if args.stiffness is not None:
        env_cfg.osc_motion_stiffness = args.stiffness
    if args.action_scale is not None:
        env_cfg.action_scale = args.action_scale
    if args.hold_steps is not None:
        env_cfg.success_hold_steps = args.hold_steps

    env = E0509OSCEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    if args.fixed_lr:
        agent_cfg = E0509OSCPPORunnerCfg_FixedLR()
        lr_mode = "Fixed (1e-4)"
    else:
        agent_cfg = E0509OSCPPORunnerCfg()
        lr_mode = "Adaptive (3e-4)"

    agent_cfg.max_iterations = args.max_iterations

    # =============================================================================
    # Runner 생성
    # =============================================================================
    if args.log_dir:
        log_dir = args.log_dir
    elif args.soft:
        log_dir = "./pen_grasp_rl/logs/e0509_osc_soft"
    else:
        log_dir = "./pen_grasp_rl/logs/e0509_osc"
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
    print("E0509 OSC 환경 강화학습 시작 (Operational Space Control)")
    print("=" * 70)
    print(f"  환경 모드: {env_mode}")
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  Learning Rate: {lr_mode}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space} (Δx, Δy, Δz)")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 70)
    print("OSC 핵심 특징:")
    print("  - 토크 제어 (set_joint_effort_target)")
    print(f"  - 임피던스 제어 (stiffness={env_cfg.osc_motion_stiffness}, damping_ratio={env_cfg.osc_motion_damping_ratio})")
    print(f"  - action_scale: {env_cfg.action_scale}")
    print("  - 중력 보상 활성화")
    print("  - 관성 디커플링 활성화")
    print("=" * 70)
    print("관절 제한 (작업 범위 제한):")
    print("  - J1: ±45° | J2: ±95° | J3: ±135°")
    print("  - J4: ±45° | J5: ±135° | J6: ±90°")
    print("=" * 70)
    print("성공 조건:")
    print(f"  - 캡까지 거리 < {env_cfg.success_dist_to_cap*100:.0f}cm")
    print(f"  - 펜 축 정렬 (perp_dist < {env_cfg.success_perp_dist*100:.0f}cm)")
    print("  - 캡 위에 있음")
    print(f"  - {env_cfg.success_hold_steps} 스텝 유지")
    print("=" * 70)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n학습 완료!")
    print(f"모델 저장 위치: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
