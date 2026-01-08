#!/usr/bin/env python3
"""
BC + RL Fine-tuning 테스트 스크립트

ACT(BC) 모델 + RL residual 정책 테스트

사용법:
    python play_bc_rl.py \
        --bc_checkpoint ../checkpoints/act/checkpoints/best_model_compat.pth \
        --rl_checkpoint ./pen_grasp_rl/logs/bc_rl_finetune_w0.7_s0.3/model_8000.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="BC + RL Fine-tuning 테스트")
parser.add_argument("--num_envs", type=int, default=16, help="환경 개수")
parser.add_argument("--bc_checkpoint", type=str, required=True, help="ACT 체크포인트 경로")
parser.add_argument("--rl_checkpoint", type=str, required=True, help="RL 체크포인트 경로")
parser.add_argument("--bc_weight", type=float, default=0.7, help="BC 정책 가중치")
parser.add_argument("--residual_scale", type=float, default=0.3, help="RL residual 스케일")
parser.add_argument("--num_steps", type=int, default=2000, help="실행할 스텝 수")
parser.add_argument("--stop_on_success", action="store_true",
                    help="성공 조건 충족 시 action=0 (진동 방지)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

from envs.e0509_ik_env_v7 import E0509IKEnvV7, E0509IKEnvV7Cfg

# ACT 모델
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imitation_learning"))
from act_model import ACT


# =============================================================================
# BC+RL 환경 (train_bc_rl.py에서 복사)
# =============================================================================
class E0509BCRLEnv(E0509IKEnvV7):
    """BC + RL Residual 환경 (옵션 A)"""

    def __init__(self, cfg, bc_model, obs_mean, obs_std, bc_weight=0.7, residual_scale=0.3,
                 stop_on_success=False, **kwargs):
        self._bc_model = bc_model
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self._bc_weight = bc_weight
        self._residual_scale = residual_scale
        self._stop_on_success = stop_on_success

        super().__init__(cfg, **kwargs)

        print(f"[E0509BCRLEnv] BC+RL 환경 초기화")
        print(f"  BC weight: {self._bc_weight}")
        print(f"  Residual scale: {self._residual_scale}")
        if stop_on_success:
            print(f"  Stop on success: ON (성공 조건 충족 시 정지)")

    def _create_bc_observation(self):
        """ACT용 관측값 생성 (25차원)"""
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        ee_pos_local = ee_pos_w - self.scene.env_origins
        ee_quat_xyzw = torch.cat([ee_quat_w[:, 1:4], ee_quat_w[:, 0:1]], dim=-1)

        pen_pos_w = self.pen.data.root_pos_w
        pen_pos_local = pen_pos_w - self.scene.env_origins

        pen_quat = self.pen.data.root_quat_w
        qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
        pen_axis = torch.stack([
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy)
        ], dim=-1)

        obs = torch.cat([
            joint_pos, joint_vel, ee_pos_local, ee_quat_xyzw, pen_pos_local, pen_axis,
        ], dim=-1)

        return obs

    def _get_bc_action(self):
        """BC(ACT) 모델에서 action 얻기"""
        bc_obs = self._create_bc_observation()
        obs_normalized = (bc_obs - self._obs_mean) / self._obs_std

        with torch.no_grad():
            bc_action = self._bc_model.get_action(obs_normalized)

        return bc_action

    def _compute_bc_jacobian_pos(self):
        """BC용 Jacobian 계산 (position 부분만, 3x6)"""
        jacobian_full = super()._compute_ee_jacobian()
        return jacobian_full[:, :3, :]

    def _apply_action(self):
        """BC + RL residual action 적용"""
        # 1. BC action (joint position target)
        bc_joint_target = self._get_bc_action()

        # 2. Joint delta
        current_joint = self.robot.data.joint_pos[:, :6]
        bc_joint_delta = bc_joint_target - current_joint

        # 3. Jacobian: joint delta → EE delta
        jacobian_pos = self._compute_bc_jacobian_pos()
        bc_ee_delta = torch.bmm(jacobian_pos, bc_joint_delta.unsqueeze(-1)).squeeze(-1)

        # 4. Combine with RL action
        rl_ee_delta = self.actions
        final_ee_delta = self._bc_weight * bc_ee_delta + self._residual_scale * rl_ee_delta

        # 5. Stop on success: 성공 조건 충족 시 action=0 (진동 방지)
        if self._stop_on_success:
            grasp_pos = self._get_grasp_point()
            cap_pos = self._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
            perp_dist, _, on_correct_side = self._compute_axis_metrics()

            # 성공 조건: dist < 7cm, perp < 5cm, 캡 위
            success_mask = (
                (dist_to_cap < 0.07) &
                (perp_dist < 0.05) &
                on_correct_side
            )
            final_ee_delta[success_mask] = 0.0

        # 6. Call parent's _apply_action
        self.actions = final_ee_delta
        super()._apply_action()


# =============================================================================
# 간단한 Actor 네트워크
# =============================================================================
class SimpleActor(nn.Module):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================================
    # ACT 모델 로드
    # ==========================================================================
    print(f"\nACT 체크포인트 로드: {args.bc_checkpoint}")
    bc_ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)

    config = bc_ckpt['config']
    bc_model = ACT(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        chunk_size=config['chunk_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_layers'],
        num_decoder_layers=config['num_layers'],
        latent_dim=config['latent_dim'],
    ).to(device)

    bc_model.load_state_dict(bc_ckpt['model_state_dict'])
    bc_model.eval()

    # 정규화 통계
    if 'obs_mean' in bc_ckpt:
        obs_mean_list = bc_ckpt['obs_mean']
        obs_std_list = bc_ckpt['obs_std']
        if isinstance(obs_mean_list, list):
            obs_mean = torch.tensor(obs_mean_list, dtype=torch.float32, device=device)
            obs_std = torch.tensor(obs_std_list, dtype=torch.float32, device=device)
        else:
            obs_mean = torch.tensor(obs_mean_list, dtype=torch.float32, device=device)
            obs_std = torch.tensor(obs_std_list, dtype=torch.float32, device=device)
    else:
        obs_mean = torch.zeros(25, device=device)
        obs_std = torch.ones(25, device=device)

    print("ACT 모델 로드 완료")

    # ==========================================================================
    # 환경 생성
    # ==========================================================================
    env_cfg = E0509IKEnvV7Cfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 10.0

    env_cfg.pen_cap_pos_range = {
        "x": (0.25, 0.60),
        "y": (-0.15, 0.15),
        "z": (0.15, 0.50),
    }
    env_cfg.pen_tilt_min = 0.0
    env_cfg.pen_tilt_max = 0.523  # 30도

    env = E0509BCRLEnv(
        cfg=env_cfg,
        bc_model=bc_model,
        obs_mean=obs_mean,
        obs_std=obs_std,
        bc_weight=args.bc_weight,
        residual_scale=args.residual_scale,
        stop_on_success=args.stop_on_success,
    )

    print(f"\n환경 생성 완료: {args.num_envs}개")

    # ==========================================================================
    # RL 모델 로드
    # ==========================================================================
    print(f"\nRL 체크포인트 로드: {args.rl_checkpoint}")
    rl_ckpt = torch.load(args.rl_checkpoint, map_location=device)
    state_dict = rl_ckpt["model_state_dict"]

    obs_dim = env_cfg.observation_space
    action_dim = env_cfg.action_space

    policy = SimpleActor(obs_dim, action_dim, hidden_dims=[256, 256, 128]).to(device)

    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            actor_state_dict[k] = v

    policy.load_state_dict(actor_state_dict)
    policy.eval()

    print("RL 모델 로드 완료")

    # ==========================================================================
    # 테스트 시작
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BC + RL Fine-tuning 테스트 시작")
    print("=" * 70)
    print(f"  BC weight: {args.bc_weight}")
    print(f"  Residual scale: {args.residual_scale}")
    print(f"  환경 수: {args.num_envs}")
    print(f"  실행 스텝: {args.num_steps}")
    if args.stop_on_success:
        print(f"  Stop on Success: ON (dist<7cm, perp<5cm, 캡 위 → 정지)")
    print("=" * 70)
    print("Action = BC_weight * ACT(obs) + residual_scale * RL(obs)")
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

        # 에피소드 종료 시 카운트
        done = terminated | truncated
        if done.any():
            episode_count += done.sum().item()

        # 100 스텝마다 상태 출력
        if step % 100 == 0:
            phase_stats = env.get_phase_stats()

            grasp_pos = env._get_grasp_point()
            cap_pos = env._get_pen_cap_pos()
            dist_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
            perp_dist, axis_dist, on_correct_side = env._compute_axis_metrics()

            mean_reward = rewards.mean().item()

            print(f"\nStep {step}: reward={mean_reward:.2f}")
            print(f"  total_success={phase_stats['total_success']}")
            print(f"  dist_to_cap={dist_to_cap.mean().item()*100:.2f}cm")
            print(f"  perp_dist={perp_dist.mean().item()*100:.2f}cm")
            print(f"  on_correct_side={on_correct_side.float().mean().item()*100:.0f}%")

            total_success = phase_stats['total_success']

    # 최종 결과
    final_stats = env.get_phase_stats()
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print(f"  총 에피소드: {episode_count}")
    print(f"  총 성공 횟수: {final_stats['total_success']}")
    if episode_count > 0:
        success_rate = final_stats['total_success'] / episode_count * 100
        print(f"  성공률: {success_rate:.1f}%")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
