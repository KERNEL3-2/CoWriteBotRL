"""
환경 설정 테스트 스크립트

=== 개요 ===
이 스크립트는 학습 전에 환경이 제대로 설정되었는지 확인하기 위한 테스트입니다.
랜덤 행동을 사용하여 다음 항목들을 검증합니다:

1. 펜 모델이 시뮬레이션에 제대로 로드되었는지
2. 펜 충돌이 작동하는지 (그리퍼가 펜을 밀 수 있는지)
3. 시각화 마커가 정상적으로 표시되는지
4. 종료 조건이 정상 작동하는지 (펜이 15cm 이상 이동 시 에피소드 종료)

=== 실행 방법 ===
python test_env.py

# 더 많은 환경으로 테스트
python test_env.py --num_envs 16

=== 시각화 마커 ===
- 파란색 점들: 펜 Z축 방향 (캡 방향)
- 노란색 점들: 그리퍼 Z축 방향

=== 출력 정보 ===
매 100스텝마다 다음 정보를 출력:
- Reward: 평균 보상
- Distance to cap: 그리퍼에서 캡까지 거리 (cm)
- Pen displacement: 펜이 초기 위치에서 이동한 거리 (cm)
- Gripper state: 그리퍼 열림/닫힘 상태

=== 예상 동작 ===
- 랜덤 행동이므로 로봇이 불규칙하게 움직임
- 그리퍼가 펜에 접촉하면 펜이 밀려남
- 펜이 15cm 이상 이동하면 에피소드가 종료되고 리셋됨
"""
import argparse
import os
import sys

# =============================================================================
# 프로젝트 경로 설정
# =============================================================================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Isaac Lab 앱 런처 설정
# =============================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test pen grasp environment setup")
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="테스트할 환경 수 (기본값: 4, 테스트이므로 적은 수로 시작)"
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =============================================================================
# 나머지 임포트 (Isaac Sim 시작 후에 해야 함)
# =============================================================================
import torch
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg, PEN_LENGTH

# 시각화 마커 도구
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# =============================================================================
# 유틸리티 함수
# =============================================================================
def quat_rotate_vector(quat, vec):
    """
    쿼터니언으로 벡터를 회전

    쿼터니언 형식: (w, x, y, z) - Isaac Lab 표준

    Args:
        quat: (N, 4) 형태의 쿼터니언
        vec: (N, 3) 형태의 벡터

    Returns:
        (N, 3) 형태의 회전된 벡터
    """
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # t = 2 * (q_xyz × vec)
    t = 2.0 * torch.stack([
        q_y * vec[:, 2] - q_z * vec[:, 1],
        q_z * vec[:, 0] - q_x * vec[:, 2],
        q_x * vec[:, 1] - q_y * vec[:, 0]
    ], dim=-1)

    # result = vec + q_w * t + (q_xyz × t)
    result = vec + q_w.unsqueeze(-1) * t + torch.stack([
        q_y * t[:, 2] - q_z * t[:, 1],
        q_z * t[:, 0] - q_x * t[:, 2],
        q_x * t[:, 1] - q_y * t[:, 0]
    ], dim=-1)
    return result


def main():
    """
    메인 테스트 함수

    1. 환경 생성
    2. 시각화 마커 설정
    3. 랜덤 행동으로 테스트 루프 실행
    4. 통계 출력
    """

    # =========================================================================
    # 1. 환경 생성
    # =========================================================================
    env_cfg = PenGraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = PenGraspEnv(cfg=env_cfg)

    # =========================================================================
    # 2. 시각화 마커 설정
    # =========================================================================
    # 캡, 그립 포인트, 축 방향 시각화
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PenMarkers",
        markers={
            # 펜 캡 위치 (빨간색) - 잡아야 할 목표
            "cap": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            # 그리퍼 잡기 포인트 (녹색)
            "grasp_point": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            # 펜 Z축 방향 (파란색 점들)
            "pen_axis": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),
            ),
            # 그리퍼 Z축 방향 (노란색 점들)
            "gripper_axis": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        }
    )
    pen_markers = VisualizationMarkers(marker_cfg)

    # 축 시각화 설정
    AXIS_POINTS = 5     # 축당 마커 개수
    AXIS_LENGTH = 0.10  # 축 길이 (10cm)

    # =========================================================================
    # 3. 환경 정보 출력
    # =========================================================================
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    act_dim = env.action_manager.total_action_dim

    print("=" * 60)
    print("환경 테스트 - 랜덤 행동")
    print("=" * 60)
    print(f"  관찰 차원: {obs_dim}")
    print(f"  행동 차원: {act_dim}")
    print(f"  환경 수: {args.num_envs}")
    print(f"  펜 길이: {PEN_LENGTH * 100:.1f} cm")
    print("=" * 60)
    print("마커 설명:")
    print("  파란색 = 펜 Z축 방향 (캡 방향)")
    print("  노란색 = 그리퍼 Z축 방향")
    print("=" * 60)
    print("종료하려면 Ctrl+C를 누르세요")
    print()

    # =========================================================================
    # 4. 랜덤 행동 테스트 루프
    # =========================================================================
    obs, _ = env.reset()
    step_count = 0
    episode_count = 0      # 완료된 에피소드 수
    termination_count = 0  # 종료 발생 횟수 (펜 이탈)

    try:
        while simulation_app.is_running():
            # --- 랜덤 행동 생성 ---
            # 표준편차 0.5로 랜덤 행동 (너무 큰 움직임 방지)
            actions = torch.randn(args.num_envs, act_dim, device="cuda:0") * 0.5

            # --- 환경 스텝 ---
            step_result = env.step(actions)

            # 반환 형식 처리 (Isaac Lab 버전에 따라 다를 수 있음)
            if len(step_result) == 2:
                # 구버전: (obs, extras)
                obs, extras = step_result
                rewards = extras.get("rewards", torch.zeros(args.num_envs, device="cuda:0"))
                dones = extras.get("dones", torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0"))
            else:
                # 신버전: (obs, rewards, dones, truncated, info)
                obs, rewards, dones, truncated, info = step_result
            step_count += 1

            # --- 종료 카운트 ---
            if dones.any():
                termination_count += dones.sum().item()
                episode_count += dones.sum().item()

            # =========================================================
            # 시각화 마커 업데이트
            # =========================================================
            pen_pos = env.scene["pen"].data.root_pos_w
            pen_quat = env.scene["pen"].data.root_quat_w
            robot = env.scene["robot"]

            # --- 잡기 포인트 계산 ---
            l1_pos = robot.data.body_pos_w[:, 7, :]   # 왼쪽 손가락 베이스
            r1_pos = robot.data.body_pos_w[:, 8, :]   # 오른쪽 손가락 베이스
            l2_pos = robot.data.body_pos_w[:, 9, :]   # 왼쪽 손가락 끝
            r2_pos = robot.data.body_pos_w[:, 10, :]  # 오른쪽 손가락 끝
            base_center = (l1_pos + r1_pos) / 2.0
            tip_center = (l2_pos + r2_pos) / 2.0
            finger_dir = tip_center - base_center
            finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)
            grasp_point = base_center + finger_dir * 0.02

            # --- 펜 캡 위치 계산 ---
            half_len = PEN_LENGTH / 2.0
            pen_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            pen_axis_world = quat_rotate_vector(pen_quat, pen_axis)
            cap_pos = pen_pos + pen_axis_world * half_len  # +Z 방향이 뒷캡 (잡을 부분)

            # --- 그리퍼 Z축 계산 ---
            link6_quat = robot.data.body_quat_w[:, 6, :]
            gripper_z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            gripper_z_world = quat_rotate_vector(link6_quat, gripper_z_axis)

            # --- 펜 이동 거리 계산 ---
            pen_pos_local = pen_pos - env.scene.env_origins
            init_pos = torch.tensor([0.5, 0.0, 0.3], device=pen_pos.device)
            displacement = torch.norm(pen_pos_local - init_pos, dim=-1)

            # --- 마커 위치 구성 ---
            # 각 환경마다: cap(1) + grasp_point(1) + pen_axis(5) + gripper_axis(5) = 12개
            num_envs = pen_pos.shape[0]
            markers_per_env = 2 + AXIS_POINTS * 2
            all_positions = torch.zeros((num_envs * markers_per_env, 3), device=pen_pos.device)
            marker_indices = []

            for i in range(num_envs):
                base_idx = i * markers_per_env
                # 캡 마커 (빨간색)
                all_positions[base_idx] = cap_pos[i]
                marker_indices.append(0)
                # 잡기 포인트 마커 (녹색)
                all_positions[base_idx + 1] = grasp_point[i]
                marker_indices.append(1)

                # 펜 Z축 마커들 (파란색)
                for j in range(AXIS_POINTS):
                    t = (j + 1) / AXIS_POINTS * AXIS_LENGTH
                    all_positions[base_idx + 2 + j] = pen_pos[i] + pen_axis_world[i] * t
                    marker_indices.append(2)  # pen_axis

                # 그리퍼 Z축 마커들 (노란색)
                for j in range(AXIS_POINTS):
                    t = (j + 1) / AXIS_POINTS * AXIS_LENGTH
                    all_positions[base_idx + 2 + AXIS_POINTS + j] = grasp_point[i] + gripper_z_world[i] * t
                    marker_indices.append(3)  # gripper_axis

            pen_markers.visualize(translations=all_positions, marker_indices=marker_indices)

            # =========================================================
            # 상태 출력 (100스텝마다)
            # =========================================================
            if step_count % 100 == 0:
                dist_to_cap = torch.norm(grasp_point - cap_pos, dim=-1)
                gripper_pos = robot.data.joint_pos[:, 6:10].mean(dim=-1)

                print(f"스텝 {step_count:5d} | 에피소드: {episode_count} | 종료: {termination_count}")
                print(f"  보상: {rewards.mean().item():+.4f}")
                print(f"  캡까지 거리: {dist_to_cap.mean().item()*100:.1f} cm")
                print(f"  펜 이동 거리: {displacement.mean().item()*100:.1f} cm (최대: {displacement.max().item()*100:.1f})")
                print(f"  그리퍼 상태: {gripper_pos.mean().item():.2f}")
                print()

    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트 중단됨")

    # =========================================================================
    # 5. 최종 통계 출력
    # =========================================================================
    print("=" * 60)
    print(f"테스트 완료: {step_count}스텝, {termination_count}회 종료")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
