#!/usr/bin/env python3
"""
ACT (Action Chunking with Transformers) 모델 테스트

Temporal Ensembling 사용하여 더 부드러운 실행

사용법:
    python play_act.py --checkpoint /path/to/best_model.pth --num_envs 16
"""
import argparse
import os
import sys

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_DIR), "IsaacLab/pen_grasp_rl"))

from isaaclab.app import AppLauncher

# 명령행 인자
parser = argparse.ArgumentParser(description="ACT 모델 테스트")
parser.add_argument("--num_envs", type=int, default=16, help="병렬 환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="ACT 모델 체크포인트")
parser.add_argument("--max_episodes", type=int, default=100, help="최대 에피소드 수")
parser.add_argument("--use_ensemble", action="store_true", help="Temporal ensemble 사용")
parser.add_argument("--debug", action="store_true", help="디버그 출력")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

# ACT 모델
sys.path.insert(0, os.path.join(PROJECT_DIR, "imitation_learning"))
from act_model import ACT, TemporalEnsemble

# RL 환경 import (IK 환경 - 초기 관절 위치가 BC와 동일)
from envs.e0509_ik_env_v7 import E0509IKEnvV7, E0509IKEnvV7Cfg

PEN_LENGTH = 0.1207

# BC 학습에 사용된 초기 관절 위치
BC_INITIAL_JOINT_POS = [0.0, -0.3, 0.8, 0.0, 1.57, 0.0]


def create_observation(env, env_ids=None):
    """
    환경에서 관측값 생성 (25차원)

    joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) + pen_pos(3) + pen_axis(3)
    """
    if env_ids is None:
        env_ids = slice(None)

    # 로봇 상태
    joint_pos = env.robot.data.joint_pos[env_ids, :6]
    joint_vel = env.robot.data.joint_vel[env_ids, :6]

    # EE 위치/방향
    ee_pos_w = env.robot.data.body_pos_w[env_ids, env.ee_body_idx]
    ee_quat_w = env.robot.data.body_quat_w[env_ids, env.ee_body_idx]  # wxyz

    # 로컬 좌표로 변환
    ee_pos_local = ee_pos_w - env.scene.env_origins[env_ids]

    # wxyz -> xyzw
    ee_quat_xyzw = torch.cat([ee_quat_w[:, 1:4], ee_quat_w[:, 0:1]], dim=-1)

    # 펜 위치/축
    pen_pos_w = env.pen.data.root_pos_w[env_ids]
    pen_pos_local = pen_pos_w - env.scene.env_origins[env_ids]

    pen_quat = env.pen.data.root_quat_w[env_ids]  # wxyz
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================================================
    # ACT 모델 로드
    # ==========================================================================
    print(f"\n체크포인트 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint['config']
    print(f"Model config: {config}")

    model = ACT(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        chunk_size=config['chunk_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_layers'],
        num_decoder_layers=config['num_layers'],
        latent_dim=config['latent_dim'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"모델 로드 완료 (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('valid_loss', '?'):.6f})")

    chunk_size = config['chunk_size']
    action_dim = config['action_dim']

    # Temporal ensemble (환경별로 하나씩)
    if args.use_ensemble:
        ensembles = [TemporalEnsemble(chunk_size, action_dim, device) for _ in range(args.num_envs)]
        print("Temporal Ensemble 활성화")
    else:
        ensembles = None

    # ==========================================================================
    # 환경 생성
    # ==========================================================================
    env_cfg = E0509IKEnvV7Cfg()
    env_cfg.scene.num_envs = args.num_envs

    # 펜 범위 설정 (BC 학습 데이터와 동일)
    env_cfg.pen_cap_pos_range = {
        "x": (0.25, 0.60),
        "y": (-0.15, 0.15),
        "z": (0.15, 0.50),
    }
    env_cfg.pen_tilt_min = 0.0
    env_cfg.pen_tilt_max = 0.523  # 30도

    env_cfg.episode_length_s = 5.0

    env = E0509IKEnvV7(env_cfg)

    # EE body index
    ee_body_idx = env.robot.find_bodies("link_6")[0][0]
    env.ee_body_idx = ee_body_idx

    print(f"\n환경 생성 완료: {args.num_envs}개")
    print(f"EE body index: {ee_body_idx}")

    # 초기 리셋
    obs, _ = env.reset()

    initial_joint_pos = env.robot.data.joint_pos[0, :6].cpu().numpy()
    print(f"초기 관절 위치: {initial_joint_pos}")
    print(f"목표 초기 위치: {BC_INITIAL_JOINT_POS}")

    # ==========================================================================
    # 테스트 루프
    # ==========================================================================
    total_episodes = 0
    total_success = 0
    step = 0

    # Action chunk 버퍼 (각 환경별)
    action_buffers = [None] * args.num_envs  # (chunk_size, action_dim) or None
    action_indices = [0] * args.num_envs  # 현재 chunk에서 몇 번째 action을 사용할지

    # 에피소드별 거리 추적
    episode_min_dist = torch.full((args.num_envs,), float('inf'), device=device)

    print("\n" + "=" * 60)
    print("ACT 모델 테스트 시작")
    print("=" * 60)

    while simulation_app.is_running() and total_episodes < args.max_episodes:
        # 관측값 생성
        obs = create_observation(env)

        # 각 환경에 대해 action 결정
        actions_to_apply = torch.zeros(args.num_envs, 6, device=device)

        for env_idx in range(args.num_envs):
            env_obs = obs[env_idx:env_idx+1]  # (1, obs_dim)

            # 새로운 chunk 예측 필요한지 확인
            if action_buffers[env_idx] is None or action_indices[env_idx] >= chunk_size:
                # 새로운 action chunk 예측
                with torch.no_grad():
                    action_chunk = model.get_action_chunk(env_obs)  # (chunk_size, action_dim)

                if args.use_ensemble:
                    action_buffers[env_idx] = action_chunk
                    action = ensembles[env_idx].update(action_chunk)
                else:
                    action_buffers[env_idx] = action_chunk
                    action_indices[env_idx] = 0
                    action = action_chunk[0]
            else:
                # 기존 chunk에서 다음 action 사용
                if args.use_ensemble:
                    # Ensemble 모드에서는 매 스텝 새로 예측
                    with torch.no_grad():
                        action_chunk = model.get_action_chunk(env_obs)
                    action = ensembles[env_idx].update(action_chunk)
                else:
                    action = action_buffers[env_idx][action_indices[env_idx]]
                    action_indices[env_idx] += 1

            actions_to_apply[env_idx] = action

        # 디버그 출력
        if args.debug and step < 10:
            print(f"\n[Step {step}]")
            print(f"  현재 joint_pos: {env.robot.data.joint_pos[0, :6].cpu().numpy()}")
            print(f"  예측 action:    {actions_to_apply[0].cpu().numpy()}")
            pen_pos = env.pen.data.root_pos_w[0] - env.scene.env_origins[0]
            print(f"  pen_pos:        {pen_pos.cpu().numpy()}")

        # 관절 한계 클리핑
        joint_lower = env.robot.data.soft_joint_pos_limits[0, :6, 0]
        joint_upper = env.robot.data.soft_joint_pos_limits[0, :6, 1]
        actions_to_apply = torch.clamp(actions_to_apply, joint_lower, joint_upper)

        # Full action (그리퍼 포함)
        full_action = torch.zeros(args.num_envs, 10, device=device)
        full_action[:, :6] = actions_to_apply
        full_action[:, 6:] = 0.0  # 그리퍼 열림

        # 직접 joint position target 설정
        env.robot.set_joint_position_target(full_action)

        # 시뮬레이션 스텝
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.cfg.sim.dt)

        env.episode_length_buf += 1
        step += 1

        # 성공/종료 조건 체크
        pen_pos = env.pen.data.root_pos_w
        pen_quat = env.pen.data.root_quat_w
        qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
        pen_axis = torch.stack([
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy)
        ], dim=-1)
        cap_pos = pen_pos + pen_axis * (PEN_LENGTH / 2)

        ee_pos = env.robot.data.body_pos_w[:, ee_body_idx]
        dist_to_cap = torch.norm(ee_pos - cap_pos, dim=-1)

        # 최소 거리 추적
        episode_min_dist = torch.minimum(episode_min_dist, dist_to_cap)

        # 성공 조건
        success = dist_to_cap < 0.07

        # 타임아웃
        timeout = env.episode_length_buf >= env.max_episode_length
        terminated = success
        truncated = timeout & ~success

        # 에피소드 종료 처리
        done = terminated | truncated
        if done.any():
            done_indices = torch.where(done)[0]

            for idx in done_indices:
                idx_int = idx.item()
                total_episodes += 1
                min_d = episode_min_dist[idx].item()

                if success[idx]:
                    total_success += 1
                    result_str = f"성공 (min_dist: {min_d:.3f}m)"
                else:
                    result_str = f"실패 (min_dist: {min_d:.3f}m)"

                if args.debug or total_episodes <= 10:
                    print(f"  Episode {total_episodes}: {result_str}")

                # 버퍼 리셋
                action_buffers[idx_int] = None
                action_indices[idx_int] = 0
                if args.use_ensemble:
                    ensembles[idx_int].reset()

            # 최소 거리 리셋
            episode_min_dist[done_indices] = float('inf')

            # 환경 리셋
            env._reset_idx(done_indices)
            env.episode_length_buf[done_indices] = 0

            # 진행 상황 출력
            if total_episodes % 10 == 0:
                rate = total_success / total_episodes if total_episodes > 0 else 0
                print(f"\nEpisodes: {total_episodes} | Success: {total_success} | Rate: {rate:.1%}")

    # ==========================================================================
    # 결과 출력
    # ==========================================================================
    success_rate = total_success / total_episodes if total_episodes > 0 else 0

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"총 에피소드: {total_episodes}")
    print(f"성공: {total_success}")
    print(f"성공률: {success_rate:.1%}")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
