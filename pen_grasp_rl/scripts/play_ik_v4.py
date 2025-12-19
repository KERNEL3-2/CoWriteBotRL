"""
E0509 IK V4 환경 테스트 스크립트

Hybrid RL + TCP Control 테스트
- APPROACH, ALIGN: 학습된 RL 정책
- FINE_ALIGN, DESCEND, GRASP: TCP 제어

사용법:
    # 기본 실행
    python play_ik_v4.py --checkpoint /path/to/model.pt

    # 환경 수 조절
    python play_ik_v4.py --checkpoint /path/to/model.pt --num_envs 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V4 환경 테스트")
parser.add_argument("--num_envs", type=int, default=16, help="환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
parser.add_argument("--num_steps", type=int, default=2000, help="실행할 스텝 수")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from envs.e0509_ik_env_v4 import E0509IKEnvV4, E0509IKEnvV4Cfg_PLAY


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


def main():
    """테스트 실행"""

    # 환경 설정
    env_cfg = E0509IKEnvV4Cfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    env = E0509IKEnvV4(cfg=env_cfg)

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

    print("=" * 70)
    print("E0509 IK V4 테스트 시작 (Hybrid RL + TCP)")
    print("=" * 70)
    print(f"  환경 수: {args.num_envs}")
    print(f"  실행 스텝: {args.num_steps}")
    print(f"  관찰 차원: {obs_dim}")
    print(f"  액션 차원: {action_dim} (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)")
    print("=" * 70)
    print("단계 (Phase):")
    print("  0: APPROACH - RL: 펜 축 방향에서 접근")
    print("  1: ALIGN - RL: 대략적 자세 정렬")
    print("  2: FINE_ALIGN - TCP: 정밀 자세 정렬 (dot → -0.98)")
    print("  3: DESCEND - TCP: 수직 하강")
    print("  4: GRASP - TCP: 그리퍼 닫기")
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
            perp_dist, axis_dist, on_correct_side = env._compute_axis_metrics()
            grasp_pos = env._get_grasp_point()
            cap_pos = env._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
            gripper_z = env._get_gripper_z_axis()
            pen_z = env._get_pen_z_axis()
            dot = torch.sum(gripper_z * pen_z, dim=-1)

            mean_reward = rewards.mean().item()

            print(f"\nStep {step}: reward={mean_reward:.4f}, "
                  f"phases=[APP:{phase_stats['approach']}, ALN:{phase_stats['align']}, "
                  f"FINE:{phase_stats['fine_align']}, DESC:{phase_stats['descend']}, "
                  f"GRP:{phase_stats['grasp']}], success={phase_stats['total_success']}")
            print(f"  → perp_dist={perp_dist.mean().item():.4f}m (need <0.05), "
                  f"axis_dist={axis_dist.mean().item():.4f}m")
            print(f"  → dist_cap={dist_to_cap.mean().item():.4f}m, "
                  f"dot={dot.mean().item():.4f} (ALIGN needs <-0.85, FINE needs <-0.98)")
            print(f"  → on_correct_side={on_correct_side.float().mean().item()*100:.1f}%")

    # 최종 결과
    final_stats = env.get_phase_stats()
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print(f"총 성공 횟수: {final_stats['total_success']}")
    print(f"최종 단계 분포:")
    print(f"  APPROACH={final_stats['approach']}, ALIGN={final_stats['align']}, "
          f"FINE_ALIGN={final_stats['fine_align']}")
    print(f"  DESCEND={final_stats['descend']}, GRASP={final_stats['grasp']}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
