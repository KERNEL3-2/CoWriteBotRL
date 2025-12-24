"""
E0509 IK V6 환경 테스트 스크립트

V6 주요 변경사항:
- 3DoF 위치 제어 (자세는 자동 정렬)
- 2단계 구조: APPROACH → GRASP

사용법:
    python play_ik_v6.py --checkpoint /path/to/model.pt
    python play_ik_v6.py --checkpoint /path/to/model.pt --num_envs 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V6 환경 테스트")
parser.add_argument("--num_envs", type=int, default=16, help="환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
parser.add_argument("--num_steps", type=int, default=2000, help="실행할 스텝 수")
parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Curriculum Level (0: 수직, 1: 10°, 2: 20°, 3: 30°)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from envs.e0509_ik_env_v6 import (
    E0509IKEnvV6,
    E0509IKEnvV6Cfg_PLAY,
    E0509IKEnvV6Cfg_L0,
    E0509IKEnvV6Cfg_L1,
    E0509IKEnvV6Cfg_L2,
    E0509IKEnvV6Cfg_L3,
    CURRICULUM_TILT_MAX,
    PHASE_APPROACH,
    PHASE_GRASP,
)


class SimpleActor(nn.Module):
    """학습된 Actor 네트워크 (추론용)"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 128]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)


def get_env_cfg_for_level(level: int):
    """Curriculum Level에 맞는 환경 설정 반환"""
    cfg_map = {
        0: E0509IKEnvV6Cfg_L0,
        1: E0509IKEnvV6Cfg_L1,
        2: E0509IKEnvV6Cfg_L2,
        3: E0509IKEnvV6Cfg_L3,
    }
    return cfg_map.get(level, E0509IKEnvV6Cfg_L0)()


def main():
    """테스트 실행"""

    # 환경 설정
    env_cfg = get_env_cfg_for_level(args.level)
    env_cfg.scene.num_envs = args.num_envs

    env = E0509IKEnvV6(cfg=env_cfg)

    # 모델 로드
    print(f"모델 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    state_dict = checkpoint["model_state_dict"]

    obs_dim = env_cfg.observation_space
    action_dim = env_cfg.action_space

    policy = SimpleActor(obs_dim, action_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    # Actor 가중치만 추출해서 로드
    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            actor_state_dict[k] = v

    policy.load_state_dict(actor_state_dict)
    policy.eval()

    tilt_deg = CURRICULUM_TILT_MAX[args.level] * 180 / 3.14159
    print("=" * 70)
    print("E0509 IK V6 테스트 시작 (3DoF 위치 + 자동 자세 정렬)")
    print("=" * 70)
    print(f"  Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}°)")
    print(f"  환경 수: {args.num_envs}")
    print(f"  실행 스텝: {args.num_steps}")
    print(f"  관찰 차원: {obs_dim}")
    print(f"  액션 차원: {action_dim} (Δx, Δy, Δz)")
    print("=" * 70)
    print("V6 핵심 특징:")
    print("  - 3DoF 위치 제어 (자세는 펜 축 기반 자동 정렬)")
    print("  - 2단계: APPROACH → GRASP")
    print("  - GRASP 조건: dist < 1.5cm AND perp < 0.8cm")
    print("=" * 70)

    # 테스트 루프
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    for step in range(args.num_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"]

        # 디버깅 출력 (100 스텝마다)
        if step % 100 == 0:
            # 단계별 통계
            phase_stats = env.get_phase_stats()

            # 메트릭 계산
            grasp_pos = env._get_grasp_point()
            cap_pos = env._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
            perp_dist, axis_dist, on_correct_side = env._compute_axis_metrics()

            # 그리퍼 Z축과 펜 Z축 정렬
            gripper_z = env._get_gripper_z_axis()
            pen_z = env._get_pen_z_axis()
            dot = torch.sum(gripper_z * pen_z, dim=-1)

            # 그리퍼 관절 위치
            gripper_pos = env.robot.data.joint_pos[:, env._gripper_joint_ids]
            gripper_amount = gripper_pos.mean(dim=-1)

            mean_reward = rewards.mean().item()

            # 단계별 카운트
            approach_count = (env.phase == PHASE_APPROACH).sum().item()
            grasp_count = (env.phase == PHASE_GRASP).sum().item()

            # 캡 위/아래 판단
            correct_side_pct = on_correct_side.float().mean().item() * 100

            print(f"\nStep {step}: reward={mean_reward:.4f}")
            print(f"  단계: APPROACH={approach_count}, GRASP={grasp_count}, "
                  f"success={phase_stats['total_success']}")
            print(f"  dist_to_cap={dist_to_cap.mean().item()*100:.2f}cm (need <1.5cm)")
            print(f"  perp_dist={perp_dist.mean().item()*100:.2f}cm (need <0.8cm)")
            print(f"  axis_dist={axis_dist.mean().item()*100:.2f}cm (음수=캡위, 양수=캡아래지나침)")
            print(f"  캡 위에 있는 비율: {correct_side_pct:.0f}%")
            print(f"  dot(정렬)={dot.mean().item():.4f} (need ~-1.0)")
            print(f"  gripper={gripper_amount.mean().item():.3f}")

    # 최종 결과
    final_stats = env.get_phase_stats()
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print(f"Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}°)")
    print(f"총 성공 횟수: {final_stats['total_success']}")
    print(f"Approach Ratio: {final_stats['approach_ratio']:.1%}")
    print(f"Grasp Ratio: {final_stats['grasp_ratio']:.1%}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
