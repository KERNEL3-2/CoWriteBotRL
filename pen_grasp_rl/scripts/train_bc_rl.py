#!/usr/bin/env python3
"""
BC + RL Fine-tuning (Residual Policy Learning)

ACT(BC) 모델을 기본 정책으로 사용하고, RL로 residual을 학습합니다.

Action = BC_weight * ACT(obs) + residual_scale * RL_action

사용법:
    python train_bc_rl.py --headless --num_envs 4096 \
        --bc_checkpoint pen_grasp_rl/checkpoints/act/checkpoints/best_model_compat.pth

주의:
    - 학습은 별도 터미널에서 실행해야 합니다
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# =============================================================================
# 명령행 인자
# =============================================================================
parser = argparse.ArgumentParser(description="BC + RL Fine-tuning")
parser.add_argument("--num_envs", type=int, default=4096, help="병렬 환경 개수")
parser.add_argument("--max_iterations", type=int, default=3000, help="최대 학습 반복 횟수")
parser.add_argument("--bc_checkpoint", type=str, required=True, help="ACT 체크포인트 경로")
parser.add_argument("--bc_weight", type=float, default=0.7,
                    help="BC 정책 가중치 (0=RL만, 1=BC만)")
parser.add_argument("--residual_scale", type=float, default=0.3,
                    help="RL residual 스케일")
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# Isaac Sim 실행
# =============================================================================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

# 환경
from envs.e0509_ik_env_v7 import E0509IKEnvV7, E0509IKEnvV7Cfg
from isaaclab.utils.math import subtract_frame_transforms

# RSL-RL
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv

# ACT 모델
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imitation_learning"))
from act_model import ACT


# =============================================================================
# BC+RL 환경 (IK 환경 상속) - 옵션 A: BC→FK→EE delta
# =============================================================================
class E0509BCRLEnv(E0509IKEnvV7):
    """
    BC + RL Residual 환경 (옵션 A)

    BC joint position → Jacobian → EE delta 변환
    기존 IK 환경의 자세 자동 정렬 유지

    Action space: 3D (EE delta) - 기존 IK 환경과 동일
    """

    def __init__(self, cfg, bc_model, obs_mean, obs_std, bc_weight=0.7, residual_scale=0.3, **kwargs):
        # BC 관련 설정 저장 (super().__init__ 전에)
        self._bc_model = bc_model
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self._bc_weight = bc_weight
        self._residual_scale = residual_scale

        super().__init__(cfg, **kwargs)

        # EE body index (부모에서 이미 설정됨)
        print(f"[E0509BCRLEnv] BC+RL 환경 초기화 (옵션 A: Jacobian 변환)")
        print(f"  BC weight: {self._bc_weight}")
        print(f"  Residual scale: {self._residual_scale}")
        print(f"  Action space: 3D (EE delta) - IK 환경 유지")

    def _create_bc_observation(self):
        """ACT용 관측값 생성 (25차원)"""
        # 로봇 상태
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        # EE 위치/방향
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]  # wxyz

        # 로컬 좌표로 변환
        ee_pos_local = ee_pos_w - self.scene.env_origins

        # wxyz -> xyzw
        ee_quat_xyzw = torch.cat([ee_quat_w[:, 1:4], ee_quat_w[:, 0:1]], dim=-1)

        # 펜 위치/축
        pen_pos_w = self.pen.data.root_pos_w
        pen_pos_local = pen_pos_w - self.scene.env_origins

        pen_quat = self.pen.data.root_quat_w  # wxyz
        qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
        pen_axis = torch.stack([
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy)
        ], dim=-1)

        # 관측값 결합
        obs = torch.cat([
            joint_pos,
            joint_vel,
            ee_pos_local,
            ee_quat_xyzw,
            pen_pos_local,
            pen_axis,
        ], dim=-1)

        return obs

    def _get_bc_action(self):
        """BC(ACT) 모델에서 action 얻기"""
        bc_obs = self._create_bc_observation()

        # 정규화
        obs_normalized = (bc_obs - self._obs_mean) / self._obs_std

        with torch.no_grad():
            bc_action = self._bc_model.get_action(obs_normalized)

        return bc_action

    def _compute_bc_jacobian_pos(self):
        """BC용 Jacobian 계산 (position 부분만, 3x6)

        Note: 부모 클래스의 _compute_ee_jacobian()과 다른 이름 사용
        """
        # 부모 클래스의 Jacobian 계산 사용 (6x6)
        jacobian_full = super()._compute_ee_jacobian()  # (num_envs, 6, 6)
        return jacobian_full[:, :3, :]  # position 부분만 (num_envs, 3, 6)

    def _apply_action(self):
        """BC + RL residual action 적용 (Option A: Jacobian 변환)"""
        # 1. BC action (joint position target) 가져오기
        bc_joint_target = self._get_bc_action()  # (num_envs, 6)

        # 2. 현재 joint position과의 차이 계산
        current_joint = self.robot.data.joint_pos[:, :6]
        bc_joint_delta = bc_joint_target - current_joint  # (num_envs, 6)

        # 3. Jacobian으로 joint delta → EE delta (3D) 변환
        # J * dq = dx (position)
        jacobian_pos = self._compute_bc_jacobian_pos()  # (num_envs, 3, 6)
        bc_ee_delta = torch.bmm(jacobian_pos, bc_joint_delta.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)

        # 4. RL action (3D EE delta)과 결합
        rl_ee_delta = self.actions  # (num_envs, 3)
        final_ee_delta = self._bc_weight * bc_ee_delta + self._residual_scale * rl_ee_delta

        # 5. 부모 클래스의 action으로 설정 후 _apply_action 호출
        # IK 환경의 자동 orientation 제어 유지
        self.actions = final_ee_delta
        super()._apply_action()


# =============================================================================
# RSL-RL 설정
# =============================================================================
@configclass
class BCRLRunnerCfg(RslRlOnPolicyRunnerCfg):
    """BC + RL Fine-tuning PPO 설정"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "bc_rl_finetune"
    run_name = "act_residual"

    empirical_normalization = False

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
        entropy_coef=0.005,  # 낮은 entropy (BC 정책 유지)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,  # 낮은 LR
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def main():
    """메인 학습 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================================
    # ACT 모델 로드
    # ==========================================================================
    print(f"\nACT 체크포인트 로드: {args.bc_checkpoint}")
    checkpoint = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)

    config = checkpoint['config']
    print(f"ACT config: {config}")

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

    bc_model.load_state_dict(checkpoint['model_state_dict'])
    bc_model.eval()

    # 정규화 통계
    if 'obs_mean' in checkpoint:
        obs_mean_list = checkpoint['obs_mean']
        obs_std_list = checkpoint['obs_std']
        # list인 경우 tensor로 변환
        if isinstance(obs_mean_list, list):
            obs_mean = torch.tensor(obs_mean_list, dtype=torch.float32, device=device)
            obs_std = torch.tensor(obs_std_list, dtype=torch.float32, device=device)
        else:
            obs_mean = torch.tensor(obs_mean_list, dtype=torch.float32, device=device)
            obs_std = torch.tensor(obs_std_list, dtype=torch.float32, device=device)
        print("정규화 통계 로드됨")
    else:
        print("경고: 정규화 통계 없음")
        obs_mean = torch.zeros(25, device=device)
        obs_std = torch.ones(25, device=device)

    print(f"ACT 모델 로드 완료")

    # ==========================================================================
    # 환경 생성
    # ==========================================================================
    env_cfg = E0509IKEnvV7Cfg()  # 3D action space (IK 환경 유지)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 10.0

    # 펜 범위 (BC 학습 데이터와 동일)
    env_cfg.pen_cap_pos_range = {
        "x": (0.25, 0.60),
        "y": (-0.15, 0.15),
        "z": (0.15, 0.50),
    }
    env_cfg.pen_tilt_min = 0.0
    env_cfg.pen_tilt_max = 0.523  # 30도

    # BC+RL 환경 생성
    env = E0509BCRLEnv(
        cfg=env_cfg,
        bc_model=bc_model,
        obs_mean=obs_mean,
        obs_std=obs_std,
        bc_weight=args.bc_weight,
        residual_scale=args.residual_scale,
    )

    # RSL-RL 래퍼
    env = RslRlVecEnvWrapper(env)

    print(f"\n환경 생성 완료: {args.num_envs}개")

    # ==========================================================================
    # PPO Runner 설정
    # ==========================================================================
    agent_cfg = BCRLRunnerCfg()
    agent_cfg.max_iterations = args.max_iterations

    log_dir = f"./pen_grasp_rl/logs/bc_rl_finetune_w{args.bc_weight}_s{args.residual_scale}"
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"체크포인트 로드: {args.checkpoint}")
        runner.load(args.checkpoint)

    # ==========================================================================
    # 학습 시작
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BC + RL Fine-tuning 시작 (Residual Policy Learning)")
    print("=" * 70)
    print(f"  BC 체크포인트: {args.bc_checkpoint}")
    print(f"  BC weight: {args.bc_weight}")
    print(f"  Residual scale: {args.residual_scale}")
    print(f"  병렬 환경 수: {args.num_envs}")
    print(f"  최대 반복 횟수: {args.max_iterations}")
    print(f"  로그 디렉토리: {log_dir}")
    print("=" * 70)
    print("Action = BC_weight * ACT(obs) + residual_scale * RL(obs)")
    print("=" * 70)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)
    print(f"모델 저장 위치: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
