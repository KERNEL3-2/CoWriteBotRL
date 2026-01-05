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
import torch.nn as nn
from envs.e0509_osc_env import E0509OSCEnv, E0509OSCEnvCfg_PLAY


# =============================================================================
# 간단한 Actor 네트워크 (학습된 구조와 동일)
# =============================================================================
class SimpleActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 128]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)


def main():
    """메인 테스트 함수"""

    # =============================================================================
    # 환경 설정
    # =============================================================================
    env_cfg = E0509OSCEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs

    env = E0509OSCEnv(cfg=env_cfg)

    # =============================================================================
    # 정책 네트워크 생성 및 체크포인트 로드
    # =============================================================================
    obs_dim = env_cfg.observation_space
    action_dim = env_cfg.action_space

    policy = SimpleActor(obs_dim, action_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    if not os.path.exists(args.checkpoint):
        print(f"오류: 체크포인트를 찾을 수 없습니다: {args.checkpoint}")
        env.close()
        simulation_app.close()
        return

    print(f"체크포인트 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")

    # Actor 가중치 추출 및 로드
    actor_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("actor."):
            new_key = key.replace("actor.", "actor.")
            actor_state_dict[new_key] = value

    policy.load_state_dict(actor_state_dict)
    policy.eval()

    # =============================================================================
    # 테스트 루프
    # =============================================================================
    print("=" * 70)
    print("E0509 OSC 환경 테스트 시작")
    print("=" * 70)
    print(f"  환경 수: {args.num_envs}")
    print(f"  관찰 차원: {obs_dim}")
    print(f"  액션 차원: {action_dim}")
    print(f"  체크포인트: {args.checkpoint}")
    print("=" * 70)
    print("종료하려면 시뮬레이션 창을 닫으세요.")
    print("=" * 70)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    step = 0

    # 통계 수집용
    dot_history = []

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"]
        step += 1

        # dot product 계산 (그리퍼 Z축과 펜 Z축)
        gripper_z = env._get_gripper_z_axis()
        pen_z = env._get_pen_z_axis()
        dot = torch.sum(gripper_z * pen_z, dim=-1)
        dot_history.extend(dot.cpu().tolist())

        # 통계 출력 (500 스텝마다)
        if step % 500 == 0:
            dot_tensor = torch.tensor(dot_history[-5000:])  # 최근 5000개

            print(f"\n[Step {step}] Dot Product 분포 분석")
            print(f"  평균: {dot_tensor.mean():.4f}")
            print(f"  중앙값: {dot_tensor.median():.4f}")
            print(f"  표준편차: {dot_tensor.std():.4f}")
            print(f"  최소/최대: {dot_tensor.min():.4f} / {dot_tensor.max():.4f}")

            # 구간별 비율
            excellent = (dot_tensor < -0.9).float().mean() * 100
            good = ((dot_tensor >= -0.9) & (dot_tensor < -0.7)).float().mean() * 100
            fair = ((dot_tensor >= -0.7) & (dot_tensor < -0.5)).float().mean() * 100
            poor = (dot_tensor >= -0.5).float().mean() * 100

            print(f"  분포:")
            print(f"    dot < -0.9 (우수): {excellent:.1f}%")
            print(f"    -0.9 ~ -0.7 (양호): {good:.1f}%")
            print(f"    -0.7 ~ -0.5 (보통): {fair:.1f}%")
            print(f"    dot > -0.5 (불량): {poor:.1f}%")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
