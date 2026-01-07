"""
E0509 IK V8 환경 테스트 스크립트

V8 주요 변경사항 (V7 대비):
- 성공 거리: 3cm → 7cm
- 관절 제한 페널티 추가

사용법:
    python play_v8.py --checkpoint /path/to/model.pt
    python play_v8.py --checkpoint /path/to/model.pt --num_envs 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="E0509 IK V8 환경 테스트")
parser.add_argument("--num_envs", type=int, default=16, help="환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
parser.add_argument("--num_steps", type=int, default=2000, help="실행할 스텝 수")
parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Curriculum Level (0: 수직, 1: 10°, 2: 20°, 3: 30°)")
parser.add_argument("--tilt-min", type=float, default=None,
                    help="펜 기울기 최소값 (도). 설정 시 --level 무시")
parser.add_argument("--tilt-max", type=float, default=None,
                    help="펜 기울기 최대값 (도). 설정 시 --level 무시")
parser.add_argument("--smooth-alpha", type=float, default=1.0,
                    help="Action smoothing factor (0=no smooth, 1=full smooth). Default: 1.0 (OFF)")
parser.add_argument("--dead-zone", type=float, default=0.0,
                    help="Dead zone distance in cm. If dist < dead_zone, action=0. Default: 0 (OFF)")
parser.add_argument("--scale-by-dist", action="store_true",
                    help="Scale action by distance (closer = smaller action)")
parser.add_argument("--scale-min", type=float, default=0.1,
                    help="Minimum scale factor when using --scale-by-dist. Default: 0.1")
parser.add_argument("--scale-range", type=float, default=10.0,
                    help="Distance in cm at which scale=1.0. Default: 10cm")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from envs.e0509_ik_env_v8 import (
    E0509IKEnvV8,
    E0509IKEnvV8Cfg_PLAY,
    E0509IKEnvV8Cfg_L0,
    E0509IKEnvV8Cfg_L1,
    E0509IKEnvV8Cfg_L2,
    E0509IKEnvV8Cfg_L3,
    CURRICULUM_TILT_MAX,
    PHASE_APPROACH,
    SUCCESS_DIST_TO_CAP,
    SUCCESS_PERP_DIST,
    SUCCESS_HOLD_STEPS,
)


class ActionProcessor:
    """Action 후처리 클래스 (Sim2Real 호환)"""
    def __init__(
        self,
        smooth_alpha: float = 1.0,
        dead_zone_cm: float = 0.0,
        scale_by_dist: bool = False,
        scale_min: float = 0.1,
        scale_range_cm: float = 10.0,
    ):
        self.smooth_alpha = smooth_alpha
        self.dead_zone = dead_zone_cm / 100.0
        self.scale_by_dist = scale_by_dist
        self.scale_min = scale_min
        self.scale_range = scale_range_cm / 100.0
        self.prev_action = None

    def process(self, action: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        processed = action.clone()

        if self.dead_zone > 0:
            in_dead_zone = dist < self.dead_zone
            processed[in_dead_zone] = 0.0

        if self.scale_by_dist:
            scale = torch.clamp(dist / self.scale_range, min=self.scale_min, max=1.0)
            processed = processed * scale.unsqueeze(-1)

        if self.smooth_alpha < 1.0:
            if self.prev_action is None:
                self.prev_action = processed.clone()
            else:
                processed = self.smooth_alpha * processed + (1 - self.smooth_alpha) * self.prev_action
                self.prev_action = processed.clone()

        return processed

    def reset(self):
        self.prev_action = None

    def get_status_str(self) -> str:
        status = []
        if self.smooth_alpha < 1.0:
            status.append(f"Smoothing(α={self.smooth_alpha})")
        if self.dead_zone > 0:
            status.append(f"DeadZone({self.dead_zone*100:.1f}cm)")
        if self.scale_by_dist:
            status.append(f"ScaleByDist(min={self.scale_min}, range={self.scale_range*100:.0f}cm)")
        return ", ".join(status) if status else "None"


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
        0: E0509IKEnvV8Cfg_L0,
        1: E0509IKEnvV8Cfg_L1,
        2: E0509IKEnvV8Cfg_L2,
        3: E0509IKEnvV8Cfg_L3,
    }
    return cfg_map.get(level, E0509IKEnvV8Cfg_L0)()


def main():
    """테스트 실행"""

    # 환경 설정
    env_cfg = get_env_cfg_for_level(args.level)
    env_cfg.scene.num_envs = args.num_envs

    # --tilt-min, --tilt-max가 지정되면 직접 설정
    if args.tilt_min is not None and args.tilt_max is not None:
        env_cfg.pen_tilt_min = args.tilt_min * 3.14159 / 180.0
        env_cfg.pen_tilt_max = args.tilt_max * 3.14159 / 180.0

    env = E0509IKEnvV8(cfg=env_cfg)

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

    # Action Processor 초기화
    processor = ActionProcessor(
        smooth_alpha=args.smooth_alpha,
        dead_zone_cm=args.dead_zone,
        scale_by_dist=args.scale_by_dist,
        scale_min=args.scale_min,
        scale_range_cm=args.scale_range,
    )

    # tilt 범위 계산
    if args.tilt_min is not None and args.tilt_max is not None:
        tilt_min_deg = args.tilt_min
        tilt_max_deg = args.tilt_max
        tilt_info = f"펜 기울기: {tilt_min_deg:.0f}°~{tilt_max_deg:.0f}° (직접 설정)"
    else:
        tilt_min_deg = 0
        tilt_max_deg = CURRICULUM_TILT_MAX[args.level] * 180 / 3.14159
        tilt_info = f"Curriculum Level: {args.level} (펜 기울기: 0°~{tilt_max_deg:.0f}°)"

    print("=" * 70)
    print("E0509 IK V8 테스트 시작 (관절 제한 + 7cm)")
    print("=" * 70)
    print(f"  {tilt_info}")
    print(f"  환경 수: {args.num_envs}")
    print(f"  실행 스텝: {args.num_steps}")
    print(f"  관찰 차원: {obs_dim}")
    print(f"  액션 차원: {action_dim} (Δx, Δy, Δz)")
    print("=" * 70)
    print("V8 핵심 특징:")
    print("  - 3DoF 위치 제어 (자세는 펜 축 기반 자동 정렬)")
    print("  - 관절 제한 페널티 (90% 초과시)")
    print(f"  - 성공 조건: dist < {SUCCESS_DIST_TO_CAP*100:.0f}cm, perp < {SUCCESS_PERP_DIST*100:.0f}cm, 캡 위")
    print(f"  - Action 후처리: {processor.get_status_str()}")
    print("=" * 70)

    # 테스트 루프
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    for step in range(args.num_steps):
        # 현재 거리 계산
        grasp_pos = env._get_grasp_point()
        cap_pos = env._get_pen_cap_pos()
        dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

        with torch.no_grad():
            raw_actions = policy(obs)
            actions = processor.process(raw_actions, dist_to_cap)

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"]

        # 디버깅 출력 (100 스텝마다)
        if step % 100 == 0:
            phase_stats = env.get_phase_stats()

            grasp_pos = env._get_grasp_point()
            cap_pos = env._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
            perp_dist, axis_dist, on_correct_side = env._compute_axis_metrics()

            gripper_z = env._get_gripper_z_axis()
            pen_z = env._get_pen_z_axis()
            dot = torch.sum(gripper_z * pen_z, dim=-1)

            mean_reward = rewards.mean().item()
            correct_side_pct = on_correct_side.float().mean().item() * 100

            # 관절 제한 페널티
            joint_limit_penalty = env._compute_joint_limit_penalty()

            success_condition = (
                (dist_to_cap < SUCCESS_DIST_TO_CAP) &
                (perp_dist < SUCCESS_PERP_DIST) &
                on_correct_side
            )
            success_pct = success_condition.float().mean().item() * 100

            print(f"\nStep {step}: reward={mean_reward:.4f}")
            print(f"  success={phase_stats['total_success']}, near_success={phase_stats.get('near_success', 0)}")
            print(f"  dist_to_cap={dist_to_cap.mean().item()*100:.2f}cm (need <{SUCCESS_DIST_TO_CAP*100:.0f}cm)")
            print(f"  perp_dist={perp_dist.mean().item()*100:.2f}cm (need <{SUCCESS_PERP_DIST*100:.0f}cm)")
            print(f"  axis_dist={axis_dist.mean().item()*100:.2f}cm (음수=캡위)")
            print(f"  캡 위에 있는 비율: {correct_side_pct:.0f}%")
            print(f"  dot(정렬)={dot.mean().item():.4f}")
            print(f"  joint_limit_penalty={joint_limit_penalty.mean().item():.4f}")
            print(f"  성공 조건 충족: {success_pct:.1f}%")

    # 최종 결과
    final_stats = env.get_phase_stats()
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print(f"  {tilt_info}")
    print(f"총 성공 횟수: {final_stats['total_success']}")
    print(f"캡 위에 있는 환경: {final_stats.get('on_correct_side', 0)}/{args.num_envs}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
