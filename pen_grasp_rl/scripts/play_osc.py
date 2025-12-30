"""
E0509 OSC 환경 테스트 스크립트 (Operational Space Control)

학습된 모델을 로드하여 시뮬레이션에서 테스트합니다.

사용법:
    # 학습된 모델 테스트
    python play_osc.py --checkpoint /path/to/model.pt

    # 환경 수 조절
    python play_osc.py --checkpoint /path/to/model.pt --num_envs 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 OSC 환경 테스트")
parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
parser.add_argument("--num_envs", type=int, default=50, help="환경 개수")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.e0509_osc_env import E0509OSCEnv, E0509OSCEnvCfg_PLAY

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL 설정 (학습 시 사용한 것과 동일해야 함)
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
    run_name = "osc_play"

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
    # 환경 설정
    # =============================================================================
    env_cfg = E0509OSCEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    env = E0509OSCEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # =============================================================================
    # PPO 설정 및 Runner 생성
    # =============================================================================
    agent_cfg = E0509OSCPPORunnerCfg()

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir="./pen_grasp_rl/logs/e0509_osc_play",
        device=agent_cfg.device,
    )

    # =============================================================================
    # 체크포인트 로드
    # =============================================================================
    if not os.path.exists(args.checkpoint):
        print(f"오류: 체크포인트를 찾을 수 없습니다: {args.checkpoint}")
        env.close()
        simulation_app.close()
        return

    print(f"체크포인트 로드: {args.checkpoint}")
    runner.load(args.checkpoint)

    # =============================================================================
    # 정책 가져오기
    # =============================================================================
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # =============================================================================
    # 테스트 루프
    # =============================================================================
    print("=" * 70)
    print("E0509 OSC 환경 테스트 시작")
    print("=" * 70)
    print(f"  환경 수: {args.num_envs}")
    print(f"  체크포인트: {args.checkpoint}")
    print("=" * 70)
    print("종료하려면 시뮬레이션 창을 닫으세요.")
    print("=" * 70)

    obs, _ = env.get_observations()
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        obs, _, dones, _ = env.step(actions)
        step += 1

        # 통계 출력 (1000 스텝마다)
        if step % 1000 == 0:
            print(f"[Step {step}] Running...")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
