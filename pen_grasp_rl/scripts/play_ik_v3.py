"""
E0509 IK V3 환경 테스트 스크립트

학습된 정책을 시각적으로 테스트
펜 축 기준 접근 방식 확인

사용법:
    # 기본 실행
    python play_ik_v3.py

    # 체크포인트 지정
    python play_ik_v3.py --checkpoint /path/to/model.pt

    # 환경 수 조절
    python play_ik_v3.py --num_envs 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V3 환경 테스트")
parser.add_argument("--num_envs", type=int, default=16, help="환경 개수")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./pen_grasp_rl/logs/e0509_ik_v3/model_4999.pt",
    help="체크포인트 경로"
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from envs.e0509_ik_env_v3 import E0509IKEnvV3, E0509IKEnvV3Cfg_PLAY

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class E0509IKV3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """테스트용 PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "e0509_ik_v3"
    run_name = "play"

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
    """테스트 실행"""

    # 환경 설정
    env_cfg = E0509IKEnvV3Cfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    env = E0509IKEnvV3(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)

    # Runner 설정
    agent_cfg = E0509IKV3PPORunnerCfg()
    log_dir = "./pen_grasp_rl/logs/e0509_ik_v3"

    runner = OnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # 체크포인트 로드
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"체크포인트 로드: {checkpoint_path}")
        runner.load(checkpoint_path)
    else:
        print(f"경고: 체크포인트 없음 - {checkpoint_path}")
        print("랜덤 정책으로 실행합니다.")

    # 정책 가져오기
    policy = runner.get_inference_policy(device=agent_cfg.device)

    print("=" * 70)
    print("E0509 IK V3 테스트 시작 (펜 축 기준 접근)")
    print("=" * 70)
    print("단계 (Phase):")
    print("  0: APPROACH - 펜 축 방향에서 접근")
    print("  1: ALIGN - 자세 정렬")
    print("  2: DESCEND - 하강")
    print("  3: GRASP - 그리퍼 닫기")
    print("=" * 70)

    # 테스트 루프
    obs, _ = wrapped_env.get_observations()
    step_count = 0
    total_reward = torch.zeros(args.num_envs, device=agent_cfg.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        obs, rewards, dones, truncations, infos = wrapped_env.step(actions)
        total_reward += rewards

        step_count += 1

        # 디버깅 출력 (100 스텝마다)
        if step_count % 100 == 0:
            # 단계별 통계
            phase_stats = env.get_phase_stats()

            # 메트릭 계산
            perp_dist, axis_dist, on_correct_side = env._compute_axis_metrics()
            grasp_pos = env._get_grasp_point()
            cap_pos = env._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
            gripper_z = env._get_gripper_z_axis()
            pen_z = env._get_pen_z_axis()
            dot = torch.sum(gripper_z * pen_z, dim=-1)

            mean_reward = total_reward.mean().item() / step_count

            print(f"\nStep {step_count}: reward={mean_reward:.4f}, "
                  f"phases=[APP:{phase_stats['approach']}, ALN:{phase_stats['align']}, "
                  f"DESC:{phase_stats['descend']}, GRP:{phase_stats['grasp']}], "
                  f"success={phase_stats['total_success']}")
            print(f"  → perp_dist={perp_dist.mean().item():.4f}m (need <0.03), "
                  f"axis_dist={axis_dist.mean().item():.4f}m")
            print(f"  → dist_cap={dist_to_cap.mean().item():.4f}m, "
                  f"dot={dot.mean().item():.4f} (need <-0.95)")
            print(f"  → on_correct_side={on_correct_side.float().mean().item()*100:.1f}%")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
