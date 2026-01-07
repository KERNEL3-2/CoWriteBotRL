"""
E0509 Domain Randomization 환경 학습 스크립트 (Sim2Real Ready)

=== Domain Randomization 항목 ===
1. 관측 노이즈: 관절 위치/속도, EE 위치, 펜 위치
2. 액션 노이즈 및 지연
3. 로봇 동역학 랜덤화 (stiffness, damping)
4. 펜 위치/기울기 랜덤화 (확장된 범위)
5. 초기 로봇 자세 랜덤화

=== 사용법 ===
    # 학습 (headless 모드)
    python train_dr.py --headless --num_envs 4096

    # 학습 (DR 비활성화 - 디버깅용)
    python train_dr.py --headless --num_envs 4096 --no_dr

    # 체크포인트에서 이어서 학습
    python train_dr.py --headless --num_envs 4096 --checkpoint /path/to/model.pt

    # 시각화 모드 (환경 50개)
    python train_dr.py --num_envs 50

주의:
    - 학습은 별도 터미널에서 실행해야 합니다
    - GPU 메모리에 따라 num_envs 조절 필요 (RTX 3090: 4096, RTX 4090: 8192)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 Domain Randomization 환경 학습")
parser.add_argument("--num_envs", type=int, default=4096, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=5000, help="최대 학습 반복 횟수")
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트")
parser.add_argument("--no_dr", action="store_true", help="Domain Randomization 비활성화")
parser.add_argument("--fixed_lr", action="store_true", help="Fixed Learning Rate 사용")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.e0509_dr_env import E0509DREnv, E0509DREnvCfg, E0509DREnvCfg_PLAY, E0509DREnvCfg_TRAIN

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL 설정
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


@configclass
class E0509DRPPORunnerCfg_FixedLR(E0509DRPPORunnerCfg):
    """Fixed LR 버전 (안정적 학습용)"""
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,    # 더 작은 LR
        schedule="fixed",      # 고정 LR
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
    if args.no_dr:
        # DR 비활성화 (디버깅용)
        env_cfg = E0509DREnvCfg_PLAY()
        dr_mode = "OFF"
    else:
        # DR 활성화 (학습용)
        env_cfg = E0509DREnvCfg_TRAIN()
        dr_mode = "ON"

    env_cfg.scene.num_envs = args.num_envs

    env = E0509DREnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    if args.fixed_lr:
        agent_cfg = E0509DRPPORunnerCfg_FixedLR()
        lr_mode = "Fixed (1e-4)"
    else:
        agent_cfg = E0509DRPPORunnerCfg()
        lr_mode = "Adaptive (3e-4)"

    agent_cfg.max_iterations = args.max_iterations

    # =============================================================================
    # Runner 생성
    # =============================================================================
    log_dir = "./pen_grasp_rl/logs/e0509_dr"
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
    print("E0509 Domain Randomization 환경 강화학습 시작")
    print("=" * 70)
    print(f"  Domain Randomization: {dr_mode}")
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  Learning Rate: {lr_mode}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space} (Δx, Δy, Δz)")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 70)
    print("Domain Randomization 항목:")
    print(f"  - 관측 노이즈: {env_cfg.dr.enable_obs_noise}")
    print(f"    · 관절 위치: ±{env_cfg.dr.obs_noise_joint_pos} rad")
    print(f"    · 관절 속도: ±{env_cfg.dr.obs_noise_joint_vel} rad/s")
    print(f"    · EE 위치: ±{env_cfg.dr.obs_noise_ee_pos} m")
    print(f"    · 펜 위치: ±{env_cfg.dr.obs_noise_pen_pos} m")
    print(f"  - 액션 노이즈: {env_cfg.dr.enable_action_noise}")
    print(f"    · 노이즈 스케일: {env_cfg.dr.action_noise}")
    print(f"    · 지연 확률: {env_cfg.dr.action_delay_prob}")
    print(f"  - 동역학 랜덤화: {env_cfg.dr.enable_dynamics_randomization}")
    print(f"    · Stiffness: {env_cfg.dr.stiffness_range}")
    print(f"    · Damping: {env_cfg.dr.damping_range}")
    print(f"  - 펜 위치 범위:")
    print(f"    · X: {env_cfg.dr.pen_pos_x_range}")
    print(f"    · Y: {env_cfg.dr.pen_pos_y_range}")
    print(f"    · Z: {env_cfg.dr.pen_pos_z_range}")
    print(f"    · Tilt: {env_cfg.dr.pen_tilt_range} rad")
    print(f"  - 초기 관절 노이즈: ±{env_cfg.dr.init_joint_noise} rad")
    print("=" * 70)
    print("성공 조건:")
    print("  - 캡까지 거리 < 3cm")
    print("  - 펜 축에서 수직 거리 < 1cm")
    print("  - 캡 위에 위치 (on_correct_side)")
    print("  - 30 스텝 유지")
    print("=" * 70)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n학습 완료!")
    print(f"모델 저장 위치: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
