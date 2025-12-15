"""
Pen Grasp RL 학습 스크립트

이 스크립트는 펜 잡기 강화학습 에이전트를 학습시킵니다.
RSL-RL 라이브러리의 PPO 알고리즘을 사용합니다.

사용법:
    # 기본 실행 (headless 모드, 4096개 환경)
    python train.py --headless --num_envs 4096 --max_iterations 5000

    # GUI 모드로 실행 (디버깅용)
    python train.py --num_envs 16 --max_iterations 100

주의:
    - 학습은 별도 터미널에서 실행해야 합니다 (Claude 터미널은 타임아웃 있음)
    - GPU 메모리에 따라 num_envs 조절 필요
"""
import argparse
import os
import sys

# 프로젝트 경로 추가 (envs 모듈 import를 위해)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# ============================================================
# 명령행 인자 파싱
# ============================================================
parser = argparse.ArgumentParser(description="펜 잡기 정책 학습")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="병렬 환경 개수 (기본: 4096)")
parser.add_argument("--max_iterations", type=int, default=3000,
                    help="최대 학습 반복 횟수 (기본: 3000)")

# AppLauncher 인자 추가 (--headless 등)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ============================================================
# Isaac Sim 실행
# ============================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 실행 후 import (순서 중요!)
import torch
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg

# RSL-RL 라이브러리
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,      # 학습 러너 설정
    RslRlPpoActorCriticCfg,      # Actor-Critic 네트워크 설정
    RslRlPpoAlgorithmCfg,        # PPO 알고리즘 설정
    RslRlVecEnvWrapper           # 환경 래퍼
)
from rsl_rl.runners import OnPolicyRunner


def main():
    """메인 학습 함수"""

    # ============================================================
    # 환경 설정
    # ============================================================
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # 환경 생성
    env = PenGraspEnv(cfg=env_cfg)

    # ============================================================
    # PPO 하이퍼파라미터 설정
    # ============================================================
    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=42,                          # 랜덤 시드 (재현성)
        device="cuda:0",                  # GPU 사용
        num_steps_per_env=24,             # 환경당 스텝 수 (rollout 길이)
        max_iterations=args.max_iterations,
        save_interval=100,                # 모델 저장 간격
        experiment_name="pen_grasp",      # 실험 이름 (로그 폴더)
        run_name="ppo_run",               # 실행 이름
        logger="tensorboard",             # TensorBoard 로깅
        obs_groups={"policy": ["policy"]},

        # Actor-Critic 네트워크 설정
        policy=RslRlPpoActorCriticCfg(
            # 초기 노이즈 표준편차
            # 2025-12-15 변경: 1.0 → 0.3 (너무 높으면 초기 행동이 랜덤)
            init_noise_std=0.3,

            # Actor 네트워크 구조 (3개 hidden layer)
            actor_hidden_dims=[256, 256, 128],
            # Critic 네트워크 구조 (3개 hidden layer)
            critic_hidden_dims=[256, 256, 128],

            activation="elu",              # 활성화 함수
            actor_obs_normalization=False,  # 관측 정규화 비활성화
            critic_obs_normalization=False,
        ),

        # PPO 알고리즘 설정
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,           # Value loss 가중치
            use_clipped_value_loss=True,   # Clipped value loss 사용
            clip_param=0.2,                # PPO 클리핑 파라미터
            entropy_coef=0.01,             # 엔트로피 보너스 (탐험 장려)
            num_learning_epochs=5,         # 데이터 재사용 횟수
            num_mini_batches=4,            # 미니배치 개수
            learning_rate=3e-4,            # 학습률
            schedule="adaptive",           # 적응적 학습률 스케줄
            gamma=0.99,                    # 할인 계수
            lam=0.95,                      # GAE lambda
            desired_kl=0.01,               # 목표 KL divergence
            max_grad_norm=1.0,             # 그래디언트 클리핑
        ),
    )

    # ============================================================
    # 학습 실행
    # ============================================================
    # RSL-RL 래퍼로 환경 감싸기
    env = RslRlVecEnvWrapper(env)

    # OnPolicyRunner 생성
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir="./logs/pen_grasp",
        device=agent_cfg.device
    )

    # 학습 시작
    print("=" * 60)
    print("펜 잡기 강화학습 시작")
    print("=" * 60)
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  초기 노이즈 std: {agent_cfg.policy.init_noise_std}")
    print(f"  학습률: {agent_cfg.algorithm.learning_rate}")
    print("=" * 60)

    runner.learn(num_learning_iterations=args.max_iterations)

    # ============================================================
    # 학습 완료 및 정리
    # ============================================================
    runner.save("final_model")
    print("\n학습 완료! 모델이 저장되었습니다.")

    # 환경 정리
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
