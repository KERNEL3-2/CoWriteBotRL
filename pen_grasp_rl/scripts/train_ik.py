"""
E0509 IK 환경 학습 스크립트

Task Space Control (IK 기반) 환경 학습
그리퍼가 위에서 아래로 내려다보는 자세로 접근하도록 학습

사용법:
    # 기본 실행 (headless 모드)
    python train_ik.py --headless --num_envs 4096

    # GUI 모드로 실행
    python train_ik.py --num_envs 64

    # 체크포인트에서 이어서 학습
    python train_ik.py --headless --num_envs 4096 --checkpoint /path/to/model.pt

주의:
    - 학습은 별도 터미널에서 실행해야 합니다 (Claude 터미널은 타임아웃 있음)
    - GPU 메모리에 따라 num_envs 조절 필요
"""
import argparse
import os
import sys

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK 환경 학습")
parser.add_argument("--num_envs", type=int, default=4096, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=5000, help="최대 학습 반복 횟수")
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트")

# AppLauncher 인자
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 실행 후 import
import torch
from envs.e0509_ik_env import E0509IKEnv, E0509IKEnvCfg

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL 설정
# =============================================================================
@configclass
class E0509IKPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """E0509 IK 환경 PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "e0509_ik"
    run_name = "ik_v1"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,               # IK 제어에 맞게 노이즈 축소
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
    env_cfg = E0509IKEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # 환경 생성
    env = E0509IKEnv(cfg=env_cfg)

    # RSL-RL 래퍼로 감싸기
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    agent_cfg = E0509IKPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations

    # =============================================================================
    # Runner 생성
    # =============================================================================
    log_dir = "./pen_grasp_rl/logs/e0509_ik"
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # 체크포인트 로드
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"체크포인트 로드: {args.checkpoint}")
        runner.load(args.checkpoint)

    # =============================================================================
    # 학습 시작
    # =============================================================================
    print("=" * 60)
    print("E0509 IK 환경 강화학습 시작 (Task Space Control)")
    print("=" * 60)
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space} (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)")
    print(f"  액션 스케일: {env_cfg.action_scale}")
    print(f"  IK 방식: {env_cfg.ik_method}")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 60)
    print("IK 제어 특징:")
    print("  - Task Space에서 직접 제어 (위치 + 자세)")
    print("  - 초기 자세: 그리퍼가 아래를 내려다봄")
    print("  - DifferentialIKController 사용 (dls method)")
    print("=" * 60)
    print("단계 전환 조건:")
    print(f"  PRE_GRASP: 펜캡 위 7cm + 정렬 (dot < -0.95)")
    print(f"  PRE_GRASP → DESCEND: 거리 < 3cm & dot < -0.95")
    print(f"  SUCCESS: 거리 < 2cm & dot < -0.95")
    print("=" * 60)

    # 학습 실행
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # =============================================================================
    # 학습 완료
    # =============================================================================
    print("\n학습 완료!")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
