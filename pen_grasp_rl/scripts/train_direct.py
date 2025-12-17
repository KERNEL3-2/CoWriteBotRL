"""
E0509 Direct 환경 학습 스크립트

단계별 상태 머신을 사용하는 Direct 환경 학습

사용법:
    # 기본 실행 (headless 모드)
    python train_direct.py --headless --num_envs 4096

    # GUI 모드로 실행
    python train_direct.py --num_envs 64

    # 체크포인트에서 이어서 학습
    python train_direct.py --headless --num_envs 4096 --checkpoint /path/to/model.pt

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
parser = argparse.ArgumentParser(description="E0509 Direct 환경 학습")
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
from envs.e0509_direct_env import E0509DirectEnv, E0509DirectEnvCfg

# RSL-RL
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv


class DirectEnvWrapper(VecEnv):
    """RSL-RL용 Direct 환경 래퍼"""

    def __init__(self, env: E0509DirectEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env.cfg.observation_space
        self.num_privileged_obs = None
        self.num_actions = env.cfg.action_space
        self.max_episode_length = env.max_episode_length

    def get_observations(self):
        obs_dict = self.env._get_observations()
        return obs_dict["policy"], {"observations": {}}

    def reset(self):
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": {}}

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)

        # RSL-RL 형식으로 변환
        dones = terminated | truncated
        time_outs = truncated

        return obs_dict["policy"], rewards, dones, {
            "observations": {},
            "time_outs": time_outs,
        }


def main():
    """메인 학습 함수"""

    # =============================================================================
    # 환경 설정
    # =============================================================================
    env_cfg = E0509DirectEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # 환경 생성
    env = E0509DirectEnv(cfg=env_cfg)

    # RSL-RL 래퍼
    wrapped_env = DirectEnvWrapper(env)

    # =============================================================================
    # PPO 설정
    # =============================================================================
    train_cfg = {
        "seed": 42,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": 100,
        "experiment_name": "e0509_direct",
        "run_name": "direct_phase",
        "logger": "tensorboard",
        "policy": {
            "init_noise_std": 0.3,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }

    # =============================================================================
    # Runner 생성
    # =============================================================================
    log_dir = "./pen_grasp_rl/logs/e0509_direct"
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        wrapped_env,
        train_cfg,
        log_dir=log_dir,
        device=train_cfg["device"],
    )

    # 체크포인트 로드
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"체크포인트 로드: {args.checkpoint}")
        runner.load(args.checkpoint)

    # =============================================================================
    # 학습 시작
    # =============================================================================
    print("=" * 60)
    print("E0509 Direct 환경 강화학습 시작")
    print("=" * 60)
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  관찰 차원: {env_cfg.observation_space}")
    print(f"  액션 차원: {env_cfg.action_space}")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 60)
    print("단계 전환 조건:")
    print(f"  APPROACH → ALIGN: 거리 < 10cm")
    print(f"  ALIGN → GRASP: dot < -0.8")
    print(f"  SUCCESS: 거리 < 2cm & dot < -0.9")
    print("=" * 60)
    print("데이터 수집 설정:")
    print(f"  수집 활성화: {env_cfg.collect_data}")
    print(f"  저장 경로: {env_cfg.data_save_path}")
    print("=" * 60)

    # 학습 루프 (단계 통계 출력 포함)
    for iteration in range(args.max_iterations):
        # 학습 스텝
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))

        # 100 iteration마다 단계 통계 출력
        if (iteration + 1) % 100 == 0:
            stats = env.get_phase_stats()
            data_stats = env.get_data_stats()
            print(f"\n[Iter {iteration + 1}] Phase Stats:")
            print(f"  APPROACH: {stats['approach']}, ALIGN: {stats['align']}, GRASP: {stats['grasp']}")
            print(f"  Total Success: {stats['total_success']}")
            print(f"  Data: {data_stats['collected_count']} episodes, "
                  f"success rate: {data_stats['success_rate']*100:.2f}%")

    # =============================================================================
    # 학습 완료
    # =============================================================================
    runner.save("final_model")
    print("\n학습 완료! 모델 저장됨.")

    # 수집된 데이터 저장
    if env_cfg.collect_data:
        data_path = env.save_collected_data()
        final_stats = env.get_data_stats()
        print("\n" + "=" * 60)
        print("Feasibility 데이터 수집 완료!")
        print("=" * 60)
        print(f"  총 에피소드: {final_stats['total_episodes']}")
        print(f"  성공 에피소드: {final_stats['successful_episodes']}")
        print(f"  성공률: {final_stats['success_rate']*100:.2f}%")
        print(f"  저장 경로: {data_path}")
        print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
