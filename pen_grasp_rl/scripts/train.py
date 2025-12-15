"""
펜 잡기 강화학습 훈련 스크립트

=== 개요 ===
이 스크립트는 RSL-RL 라이브러리의 PPO 알고리즘을 사용하여
펜 잡기 정책을 훈련합니다.

=== 실행 방법 ===
# 기본 실행 (4096 환경, 3000 이터레이션)
python train.py --headless

# 환경 수 및 이터레이션 조정
python train.py --headless --num_envs 2048 --max_iterations 5000

# GUI 모드 (시뮬레이션 확인용)
python train.py --num_envs 64 --max_iterations 100

=== 출력 ===
- 학습 로그: ./logs/pen_grasp/{timestamp}/
- 체크포인트: ./logs/pen_grasp/{timestamp}/model_*.pt
- TensorBoard 로그: ./logs/pen_grasp/{timestamp}/events.*

=== 주의사항 ===
- 학습은 GPU 메모리를 많이 사용합니다 (4096 환경 기준 약 8GB+)
- --headless 옵션을 사용하면 GUI 없이 빠르게 학습됩니다
- Claude 터미널에서 실행하지 마세요 (타임아웃 발생)
"""
import argparse
import os
import sys
from datetime import datetime

# =============================================================================
# 프로젝트 경로 설정
# =============================================================================
# 스크립트 위치 기준으로 상위 디렉토리(pen_grasp_rl)를 Python 경로에 추가
# 이를 통해 envs.pen_grasp_env 모듈을 임포트할 수 있음
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Isaac Lab 앱 런처 설정
# =============================================================================
from isaaclab.app import AppLauncher

# 커맨드라인 인자 파싱
parser = argparse.ArgumentParser(description="Train pen grasping policy")
parser.add_argument(
    "--num_envs",
    type=int,
    default=4096,
    help="병렬로 실행할 환경 수 (기본값: 4096). 더 많은 환경 = 더 빠른 학습, 더 많은 메모리"
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=3000,
    help="최대 학습 이터레이션 수 (기본값: 3000)"
)

# AppLauncher 인자 추가 (--headless, --device 등)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac Sim 실행 (이 시점에서 시뮬레이션 창이 열림)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =============================================================================
# 나머지 임포트 (Isaac Sim 시작 후에 해야 함)
# =============================================================================
import torch
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg

# RSL-RL: NVIDIA에서 제공하는 강화학습 라이브러리
# PPO (Proximal Policy Optimization) 알고리즘 사용
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,   # 러너 설정
    RslRlPpoActorCriticCfg,   # Actor-Critic 네트워크 설정
    RslRlPpoAlgorithmCfg,     # PPO 알고리즘 하이퍼파라미터
    RslRlVecEnvWrapper        # Isaac Lab 환경을 RSL-RL에서 사용할 수 있게 래핑
)
from rsl_rl.runners import OnPolicyRunner  # PPO 훈련 러너


class BestModelRunner(OnPolicyRunner):
    """
    Best Model 저장 기능이 추가된 OnPolicyRunner

    매 이터레이션마다 평균 리워드를 확인하고,
    역대 최고 리워드를 달성하면 best_model.pt로 저장합니다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_reward = float('-inf')  # 역대 최고 리워드
        self.best_iteration = 0           # 최고 리워드 달성 이터레이션

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        학습 실행 (Best Model 저장 기능 포함)

        부모 클래스의 learn()을 한 이터레이션씩 실행하면서
        best model을 추적합니다.
        """
        # 초기화
        self.alg.init_storage(
            self.env.num_envs,
            num_learning_iterations,
            [self.env.num_obs],
            [self.env.num_privileged_obs] if self.env.num_privileged_obs is not None else [0],
            [self.env.num_actions],
        )

        obs, _ = self.env.get_observations()
        critic_obs = obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = []
        lenbuffer = []
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 메인 학습 루프
        for it in range(num_learning_iterations):
            # 데이터 수집
            for _ in range(self.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs)
                obs, rewards, dones, infos = self.env.step(actions)
                critic_obs = obs
                obs, critic_obs, rewards, dones = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )
                self.alg.process_env_step(rewards, dones, infos)

                # 에피소드 통계
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

            # PPO 업데이트
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            # 평균 리워드 계산
            if len(rewbuffer) > 0:
                mean_reward = sum(rewbuffer[-100:]) / len(rewbuffer[-100:])

                # Best Model 체크 및 저장
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.best_iteration = it
                    self.save("best_model")
                    print(f"[Best Model] Iteration {it}: reward = {mean_reward:.4f} (saved!)")

            # 로깅
            self.log(locals())

            # 주기적 체크포인트 저장
            if it % self.save_interval == 0:
                self.save(f"model_{it}")

        # 최종 모델 저장
        self.save(f"model_{num_learning_iterations}")

        print(f"\n{'='*50}")
        print(f"Best Model: iteration {self.best_iteration}, reward = {self.best_reward:.4f}")
        print(f"{'='*50}")


def main():
    """
    메인 학습 함수

    1. 환경 생성
    2. PPO 설정
    3. 학습 실행
    4. 모델 저장
    """

    # =========================================================================
    # 1. 환경 설정 및 생성
    # =========================================================================
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs  # 커맨드라인에서 지정한 환경 수로 업데이트

    # 환경 인스턴스 생성
    env = PenGraspEnv(cfg=env_cfg)

    # =========================================================================
    # 2. PPO 알고리즘 설정
    # =========================================================================
    agent_cfg = RslRlOnPolicyRunnerCfg(
        # --- 기본 설정 ---
        seed=42,                      # 랜덤 시드 (재현성)
        device="cuda:0",              # GPU 사용
        num_steps_per_env=24,         # 각 환경에서 한 번에 수집할 스텝 수
        max_iterations=args.max_iterations,  # 최대 학습 이터레이션
        save_interval=100,            # 체크포인트 저장 주기

        # --- 로깅 설정 ---
        experiment_name="pen_grasp",  # 실험 이름
        run_name="ppo_run",           # 실행 이름
        logger="tensorboard",         # 로거 타입 (tensorboard, wandb 등)
        obs_groups={"policy": ["policy"]},  # 관찰 그룹 매핑

        # --- Actor-Critic 네트워크 설정 ---
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,                  # 초기 행동 노이즈 (탐색용)
            actor_hidden_dims=[256, 256, 128],   # Actor(정책) 네트워크 히든 레이어
            critic_hidden_dims=[256, 256, 128],  # Critic(가치) 네트워크 히든 레이어
            activation="elu",                    # 활성화 함수
            actor_obs_normalization=False,       # Actor 관찰 정규화 비활성화
            critic_obs_normalization=False,      # Critic 관찰 정규화 비활성화
        ),

        # --- PPO 알고리즘 하이퍼파라미터 ---
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,          # 가치 손실 계수
            use_clipped_value_loss=True,  # 클리핑된 가치 손실 사용
            clip_param=0.2,               # PPO 클리핑 파라미터 (정책 업데이트 제한)
            entropy_coef=0.01,            # 엔트로피 보너스 (탐색 장려)
            num_learning_epochs=5,        # 수집된 데이터로 학습할 에포크 수
            num_mini_batches=4,           # 미니배치 수
            learning_rate=3e-4,           # 학습률
            schedule="adaptive",          # 학습률 스케줄 (적응적)
            gamma=0.99,                   # 할인율 (미래 보상 가중치)
            lam=0.95,                     # GAE lambda (Advantage 추정)
            desired_kl=0.01,              # 목표 KL 발산 (적응적 학습률용)
            max_grad_norm=1.0,            # 그래디언트 클리핑 (안정성)
        ),
    )

    # =========================================================================
    # 3. 환경 래핑 및 러너 생성
    # =========================================================================
    # Isaac Lab 환경을 RSL-RL에서 사용할 수 있는 형태로 래핑
    env = RslRlVecEnvWrapper(env)

    # 타임스탬프 기반 로그 디렉토리 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./logs/pen_grasp/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # 훈련 러너 생성 (Best Model 저장 기능 포함)
    runner = BestModelRunner(
        env,
        agent_cfg.to_dict(),      # 설정을 딕셔너리로 변환
        log_dir=log_dir,
        device=agent_cfg.device
    )

    # =========================================================================
    # 4. 학습 실행
    # =========================================================================
    print("=" * 50)
    print("학습 시작...")
    print(f"  환경 수: {args.num_envs}")
    print(f"  최대 이터레이션: {args.max_iterations}")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 50)

    # PPO 학습 실행
    runner.learn(num_learning_iterations=args.max_iterations)

    # =========================================================================
    # 5. 최종 모델 저장 및 정리
    # =========================================================================
    runner.save("final_model")
    print("학습 완료!")

    # 리소스 정리
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
