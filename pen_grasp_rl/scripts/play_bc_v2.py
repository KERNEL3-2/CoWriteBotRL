#!/usr/bin/env python3
"""
BC (Behavioral Cloning) 모델 테스트 - RL 환경 프레임워크 기반

기존 RL 환경을 활용하여 초기화 문제 해결

핵심:
- BC 모델: 절대 관절 위치를 출력
- 환경: Direct 환경 프레임워크 사용 (적절한 초기화)
- 제어: 직접 joint position target 설정 (delta 아님)

사용법:
    python play_bc_v2.py --checkpoint /path/to/best_model.pth --num_envs 16
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
parser = argparse.ArgumentParser(description="BC 모델 테스트 (RL 환경 기반)")
parser.add_argument("--num_envs", type=int, default=16, help="병렬 환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="BC 모델 체크포인트")
parser.add_argument("--max_episodes", type=int, default=100, help="최대 에피소드 수")
parser.add_argument("--debug", action="store_true", help="디버그 출력 활성화")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

# RL 환경 import (IK 환경 - 초기 관절 위치가 BC와 동일)
from envs.e0509_ik_env_v7 import E0509IKEnvV7, E0509IKEnvV7Cfg

PEN_LENGTH = 0.1207

# BC 학습에 사용된 초기 관절 위치 (MoveIt2 궤적 시작점)
BC_INITIAL_JOINT_POS = [0.0, -0.3, 0.8, 0.0, 1.57, 0.0]


def set_bc_initial_joint_pos(env, env_ids=None):
    """
    BC 학습 시 사용한 초기 관절 위치로 로봇 설정

    MoveIt2 궤적이 이 위치에서 시작하므로, BC 테스트도 동일하게 시작해야 함
    """
    device = env.device

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=device)

    num_reset = len(env_ids)

    # BC 초기 관절 위치 (6 DOF arm + 4 DOF gripper)
    joint_pos = torch.zeros(num_reset, 10, device=device)
    joint_pos[:, 0] = BC_INITIAL_JOINT_POS[0]  # joint_1
    joint_pos[:, 1] = BC_INITIAL_JOINT_POS[1]  # joint_2
    joint_pos[:, 2] = BC_INITIAL_JOINT_POS[2]  # joint_3
    joint_pos[:, 3] = BC_INITIAL_JOINT_POS[3]  # joint_4
    joint_pos[:, 4] = BC_INITIAL_JOINT_POS[4]  # joint_5
    joint_pos[:, 5] = BC_INITIAL_JOINT_POS[5]  # joint_6
    # gripper는 0.0 (열림)

    joint_vel = torch.zeros(num_reset, 10, device=device)

    # 전체 환경용 joint position (타겟 설정용)
    full_joint_pos = env.robot.data.joint_pos.clone()
    full_joint_pos[env_ids] = joint_pos

    # 1. 관절 상태 직접 설정 (물리 시뮬레이션)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # 2. 컨트롤러 타겟도 동일하게 설정 (중요!)
    env.robot.set_joint_position_target(full_joint_pos)

    # 시뮬레이션에 반영
    env.scene.write_data_to_sim()

    # 여러 번 스텝하여 안정화
    for _ in range(20):
        env.robot.set_joint_position_target(full_joint_pos)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.cfg.sim.dt)


# =============================================================================
# BC Policy 네트워크 (train_bc.py와 동일)
# =============================================================================
class BCPolicy(nn.Module):
    def __init__(self, obs_dim=25, action_dim=6, hidden_dims=[512, 512, 256]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def create_bc_observation(env, env_ids=None, debug=False):
    """
    RL 환경에서 BC 관측 (25차원) 생성

    BC 학습 데이터 형식:
    - joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) + pen_pos(3) + pen_axis(3) = 25

    주의사항:
    - ee_pos/pen_pos: 로봇 베이스 기준 좌표 (Isaac Lab에서는 env_origins 뺀 값)
    - ee_quat: xyzw 형식 (scipy 기준, Isaac Lab은 wxyz이므로 변환 필요)
    """
    if env_ids is None:
        env_ids = slice(None)

    # 로봇 상태
    joint_pos = env.robot.data.joint_pos[env_ids, :6]  # (N, 6)
    joint_vel = env.robot.data.joint_vel[env_ids, :6]  # (N, 6)

    # EE 위치/방향 (월드 -> 로봇 베이스 기준)
    ee_pos_w = env.robot.data.body_pos_w[env_ids, env.ee_body_idx]  # (N, 3)
    ee_quat_w = env.robot.data.body_quat_w[env_ids, env.ee_body_idx]  # (N, 4) wxyz

    # 로봇 베이스 기준 좌표로 변환 (MoveIt FK와 동일하게)
    ee_pos_local = ee_pos_w - env.scene.env_origins[env_ids]

    # wxyz -> xyzw (scipy/ROS 형식)
    ee_quat_xyzw = torch.cat([ee_quat_w[:, 1:4], ee_quat_w[:, 0:1]], dim=-1)

    # 펜 위치/축
    pen_pos_w = env.pen.data.root_pos_w[env_ids]  # (N, 3)
    pen_pos_local = pen_pos_w - env.scene.env_origins[env_ids]

    pen_quat = env.pen.data.root_quat_w[env_ids]  # (N, 4) wxyz
    # 펜 축 계산 (Z축 방향 = 펜의 길이 방향)
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_axis = torch.stack([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy)
    ], dim=-1)

    # BC 관측 결합 (25차원)
    bc_obs = torch.cat([
        joint_pos,      # 0:6
        joint_vel,      # 6:12
        ee_pos_local,   # 12:15
        ee_quat_xyzw,   # 15:19
        pen_pos_local,  # 19:22
        pen_axis,       # 22:25
    ], dim=-1)

    if debug:
        print(f"  [Observation Debug]")
        print(f"    joint_pos: {joint_pos[0].cpu().numpy()}")
        print(f"    joint_vel: {joint_vel[0].cpu().numpy()}")
        print(f"    ee_pos:    {ee_pos_local[0].cpu().numpy()}")
        print(f"    ee_quat:   {ee_quat_xyzw[0].cpu().numpy()}")
        print(f"    pen_pos:   {pen_pos_local[0].cpu().numpy()}")
        print(f"    pen_axis:  {pen_axis[0].cpu().numpy()}")

    return bc_obs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================================================
    # BC 모델 로드
    # ==========================================================================
    print(f"\n체크포인트 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = BCPolicy().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"모델 로드 완료 (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('valid_loss', '?'):.6f})")

    # ==========================================================================
    # RL 환경 생성 (IK 환경 - 초기 관절 위치가 BC와 동일)
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

    # 에피소드 길이
    env_cfg.episode_length_s = 5.0  # 5초

    env = E0509IKEnvV7(env_cfg)

    # EE body index 찾기
    ee_body_idx = env.robot.find_bodies("link_6")[0][0]
    env.ee_body_idx = ee_body_idx

    print(f"\n환경 생성 완료: {args.num_envs}개")
    print(f"EE body index: {ee_body_idx}")

    # 초기 리셋 (IK 환경은 이미 BC 초기 위치 설정됨)
    obs, _ = env.reset()

    initial_joint_pos = env.robot.data.joint_pos[0, :6].cpu().numpy()
    print(f"초기 관절 위치: {initial_joint_pos}")
    print(f"목표 초기 위치: {BC_INITIAL_JOINT_POS}")

    # 위치 오차 확인
    error = np.abs(np.array(initial_joint_pos) - np.array(BC_INITIAL_JOINT_POS))
    print(f"관절 위치 오차: {error}")
    if np.max(error) > 0.1:
        print("⚠️  경고: 초기 관절 위치가 목표와 많이 다릅니다!")

    # ==========================================================================
    # 테스트 루프
    # ==========================================================================
    total_episodes = 0
    total_success = 0
    step = 0

    print("\n" + "=" * 60)
    print("BC 모델 테스트 시작")
    print("=" * 60)

    # 에피소드별 거리 추적
    episode_min_dist = torch.full((args.num_envs,), float('inf'), device=device)

    while simulation_app.is_running() and total_episodes < args.max_episodes:
        # BC 관측 생성
        debug_obs = args.debug and step < 3
        bc_obs = create_bc_observation(env, debug=debug_obs)

        # BC 모델로 action 예측
        with torch.no_grad():
            predicted_joint_pos = model(bc_obs)  # (N, 6)

        # 디버그 출력 (처음 몇 스텝)
        if args.debug and step < 10:
            current_jp = env.robot.data.joint_pos[0, :6].cpu().numpy()
            pred_jp = predicted_joint_pos[0].cpu().numpy()
            diff = pred_jp - current_jp
            print(f"\n[Step {step}]")
            print(f"  현재 joint_pos: {current_jp}")
            print(f"  예측 action:    {pred_jp}")
            print(f"  차이 (delta):   {diff}")
            pen_pos = env.pen.data.root_pos_w[0] - env.scene.env_origins[0]
            print(f"  pen_pos:        {pen_pos.cpu().numpy()}")

        # RL 환경의 action 형식으로 변환
        # e0509_ik_env_v7은 IK 기반이므로, 여기서는 직접 joint position을 사용
        # 환경이 IK action을 기대하면 맞지 않을 수 있음
        # 대신 직접 joint position target 설정

        # 관절 한계 클리핑
        joint_lower = env.robot.data.soft_joint_pos_limits[0, :6, 0]
        joint_upper = env.robot.data.soft_joint_pos_limits[0, :6, 1]
        predicted_joint_pos = torch.clamp(predicted_joint_pos, joint_lower, joint_upper)

        # 그리퍼 포함한 full action
        full_action = torch.zeros(args.num_envs, 10, device=device)
        full_action[:, :6] = predicted_joint_pos
        full_action[:, 6:] = 0.0  # 그리퍼 열림

        # 직접 joint position target 설정
        env.robot.set_joint_position_target(full_action)

        # 시뮬레이션 직접 스텝 (env.step 대신)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.cfg.sim.dt)

        # 에피소드 카운터 증가
        env.episode_length_buf += 1

        # 성공/종료 조건 체크
        # 펜 캡 위치 계산
        pen_pos = env.pen.data.root_pos_w
        pen_quat = env.pen.data.root_quat_w
        qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
        pen_axis = torch.stack([
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy)
        ], dim=-1)
        cap_pos = pen_pos + pen_axis * (PEN_LENGTH / 2)

        # EE 위치
        ee_pos = env.robot.data.body_pos_w[:, ee_body_idx]

        # 캡까지 거리 계산
        dist_to_cap = torch.norm(ee_pos - cap_pos, dim=-1)

        # 최소 거리 추적
        episode_min_dist = torch.minimum(episode_min_dist, dist_to_cap)

        # 성공 조건: EE가 캡에 가까움 (7cm 이내 = grasp_offset 근처)
        success = dist_to_cap < 0.07

        # 타임아웃
        timeout = env.episode_length_buf >= env.max_episode_length

        terminated = success
        truncated = timeout & ~success

        step += 1

        # 에피소드 종료 처리
        done = terminated | truncated
        if done.any():
            done_indices = torch.where(done)[0]

            for idx in done_indices:
                total_episodes += 1
                min_d = episode_min_dist[idx].item()

                if success[idx]:
                    total_success += 1
                    result_str = f"성공 (min_dist: {min_d:.3f}m)"
                else:
                    result_str = f"실패 (min_dist: {min_d:.3f}m)"

                if args.debug or total_episodes <= 5:
                    print(f"  Episode {total_episodes}: {result_str}")

            # 최소 거리 리셋
            episode_min_dist[done_indices] = float('inf')

            # 환경 리셋 (tensor로 전달) - IK 환경은 이미 BC 초기 위치 설정됨
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
