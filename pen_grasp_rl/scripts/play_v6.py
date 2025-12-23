"""
E0509 IK V6 환경 테스트 스크립트 (3DoF + Auto Orientation)

V6 핵심:
- RL: 위치(x, y, z)만 제어 -> 3DoF
- 자세: 펜 축 기반 자동 계산

사용법:
    # Level 0 (펜 수직) 테스트
    python play_v6.py --checkpoint /path/to/model.pt --level 0

    # Level 3 (30도 기울기) 테스트
    python play_v6.py --checkpoint /path/to/model.pt --level 3

    # 환경 수 조절
    python play_v6.py --checkpoint /path/to/model.pt --level 0 --num_envs 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V6 환경 테스트 (3DoF + Auto Orientation)")
parser.add_argument("--num_envs", type=int, default=16, help="환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
parser.add_argument("--num_steps", type=int, default=2000, help="실행할 스텝 수")
parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Curriculum Level (0: 수직, 1: 10도, 2: 20도, 3: 30도)")

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
    print("E0509 IK V6 테스트 시작 (3DoF + Auto Orientation)")
    print("=" * 70)
    print(f"  Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}도)")
    print(f"  환경 수: {args.num_envs}")
    print(f"  실행 스텝: {args.num_steps}")
    print(f"  관찰 차원: {obs_dim}")
    print(f"  액션 차원: {action_dim} (Dx, Dy, Dz - 위치만!)")
    print("=" * 70)
    print("V6 핵심 변경사항:")
    print("  - RL: 위치(x, y, z)만 제어 -> 3DoF")
    print("  - 자세: 펜 축 기반 자동 계산")
    print("  - 성공 조건: dot(자세) 제거!")
    print("=" * 70)
    print("단계 (Phase) - 2단계:")
    print("  0: APPROACH - RL: 펜 캡 위치로 접근 (자세는 자동)")
    print("  1: GRASP - 그리퍼 닫기 -> Good Grasp 시 성공")
    print("=" * 70)

    # 테스트 루프
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_success = 0
    episode_count = 0

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
            perp_dist, axis_dist, on_correct_side = env._compute_axis_metrics()
            grasp_pos = env._get_grasp_point()
            cap_pos = env._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

            # 그리퍼 관절 위치 (Good Grasp 체크용)
            gripper_pos = env.robot.data.joint_pos[:, env._gripper_joint_ids]
            gripper_amount = gripper_pos.mean(dim=-1)

            # Good Grasp 조건 체크 (V6: dot 제거!)
            good_grasp = (
                (perp_dist < 0.015) &
                (gripper_amount > 0.8) & (gripper_amount < 1.05)
            )
            good_grasp_pct = good_grasp.float().mean().item() * 100

            # 자세 정렬 확인 (참고용)
            gripper_z = env._get_gripper_z_axis()
            pen_z = env._get_pen_z_axis()
            dot = torch.sum(gripper_z * pen_z, dim=-1)

            mean_reward = rewards.mean().item()

            print(f"\nStep {step}: reward={mean_reward:.4f}, "
                  f"phases=[APP:{phase_stats['approach']}, GRP:{phase_stats['grasp']}], "
                  f"success={phase_stats['total_success']}")
            print(f"  -> perp_dist={perp_dist.mean().item():.4f}m (need <0.015), "
                  f"dist_cap={dist_to_cap.mean().item():.4f}m")
            print(f"  -> dot={dot.mean().item():.4f} (auto-aligned), "
                  f"gripper={gripper_amount.mean().item():.3f} (need 0.8~1.05)")
            print(f"  -> good_grasp={good_grasp_pct:.1f}%")

    # 최종 결과
    final_stats = env.get_phase_stats()
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print(f"Curriculum Level: {args.level} (펜 최대 기울기: {tilt_deg:.0f}도)")
    print(f"총 성공 횟수: {final_stats['total_success']}")
    print(f"최종 단계 분포:")
    print(f"  APPROACH={final_stats['approach']}, GRASP={final_stats['grasp']}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
