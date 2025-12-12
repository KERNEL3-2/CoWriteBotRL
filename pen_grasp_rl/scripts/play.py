"""
학습된 정책 테스트 스크립트

=== 개요 ===
이 스크립트는 학습된 모델 체크포인트를 로드하여
펜 잡기 정책의 성능을 시각적으로 확인합니다.

=== 실행 방법 ===
# 기본 실행 (체크포인트 경로 필수)
python play.py --checkpoint ./logs/pen_grasp/2024-01-01_12-00-00/model_3000.pt

# 환경 수 조정 (더 적은 환경으로 빠르게 테스트)
python play.py --checkpoint ./path/to/model.pt --num_envs 8

=== 시각화 마커 ===
- 빨간색 구: 펜 캡 위치 (잡아야 할 목표)
- 녹색 구: 그리퍼 잡기 포인트 위치
- 파란색 점들: 펜 Z축 방향
- 노란색 점들: 그리퍼 Z축 방향

=== 출력 정보 ===
매 50스텝마다 다음 정보를 출력:
- Mean Reward: 평균 보상
- GraspPoint→Cap: 그리퍼에서 캡까지 거리 (cm)
- Z-axis alignment: Z축 정렬 상태 (-1=반대, 0=수직, 1=평행)
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

parser = argparse.ArgumentParser(description="Test trained pen grasping policy")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="모델 체크포인트 파일 경로 (예: ./logs/pen_grasp/2024-01-01/model_3000.pt)"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="테스트할 환경 수 (기본값: 16)"
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# GUI 모드로 시작 (--headless 옵션 없이)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =============================================================================
# 나머지 임포트 (Isaac Sim 시작 후에 해야 함)
# =============================================================================
import torch
import torch.nn as nn
from envs.pen_grasp_env import PenGraspEnv, PenGraspEnvCfg, PEN_LENGTH

# 시각화 마커 도구
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# =============================================================================
# 정책 네트워크 정의 (추론용 간단한 버전)
# =============================================================================
class SimplePolicy(nn.Module):
    """
    추론용 간단한 MLP 정책 네트워크

    RSL-RL로 학습된 Actor 네트워크의 가중치를 로드하여
    추론(inference)만 수행합니다.

    === 네트워크 구조 ===
    입력 → Linear(256) → ELU → Linear(256) → ELU → Linear(128) → ELU → Linear(출력)

    학습 시 사용된 actor_hidden_dims=[256, 256, 128]과 동일한 구조입니다.
    """
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256, 128]):
        """
        Args:
            obs_dim: 관찰 벡터 차원 (입력)
            act_dim: 행동 벡터 차원 (출력)
            hidden_dims: 히든 레이어 크기 리스트
        """
        super().__init__()

        # 네트워크 레이어 순차적으로 구성
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))  # 선형 레이어
            layers.append(nn.ELU())                        # 활성화 함수
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))  # 출력 레이어 (활성화 없음)

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        """관찰을 입력받아 행동을 출력"""
        return self.network(obs)


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

    # 쿼터니언 회전 공식: v' = q * v * q^(-1)
    # 최적화된 형태로 계산

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
    3. 학습된 정책 로드
    4. 추론 루프 실행
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
    # 디버깅용 3D 마커 (펜 캡, 그리퍼 위치, 축 방향 시각화)
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PenMarkers",
        markers={
            # 펜 캡 위치 (빨간색) - 잡아야 할 목표
            "cap": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            # 그리퍼 잡기 포인트 (녹색) - 현재 그리퍼 위치
            "grasp_point": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            # 펜 Z축 방향 (파란색 점들)
            "pen_axis": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),
            ),
            # 그리퍼 Z축 방향 (노란색 점들)
            "gripper_axis": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        }
    )
    pen_markers = VisualizationMarkers(marker_cfg)

    # 축 시각화 설정
    AXIS_POINTS = 5     # 축당 마커 개수
    AXIS_LENGTH = 0.15  # 축 길이 (15cm)

    # =========================================================================
    # 3. 정책 네트워크 로드
    # =========================================================================
    # 관찰/행동 차원 확인
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    act_dim = env.action_manager.total_action_dim

    print(f"체크포인트 로딩: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)

    # 정책 네트워크 생성 (학습 시와 동일한 구조)
    policy = SimplePolicy(obs_dim, act_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    # 체크포인트에서 Actor 가중치 추출
    # RSL-RL 체크포인트 구조: model_state_dict 내에 actor.* 키로 저장됨
    model_state = checkpoint["model_state_dict"]

    # RSL-RL의 Actor 레이어 → SimplePolicy 네트워크 레이어로 매핑
    # actor.0 (Linear) → network.0, actor.2 → network.2, ...
    # (중간 번호는 활성화 함수로 저장되지 않음)
    policy_state = {}
    actor_layer_map = {
        "actor.0.weight": "network.0.weight",   # 첫 번째 Linear
        "actor.0.bias": "network.0.bias",
        "actor.2.weight": "network.2.weight",   # 두 번째 Linear
        "actor.2.bias": "network.2.bias",
        "actor.4.weight": "network.4.weight",   # 세 번째 Linear
        "actor.4.bias": "network.4.bias",
        "actor.6.weight": "network.6.weight",   # 출력 Linear
        "actor.6.bias": "network.6.bias",
    }

    for ckpt_key, policy_key in actor_layer_map.items():
        if ckpt_key in model_state:
            policy_state[policy_key] = model_state[ckpt_key]

    # 가중치 로드
    try:
        policy.load_state_dict(policy_state, strict=False)
        print("정책 가중치 로드 성공")
        print(f"  로드된 레이어: {len(policy_state)}개")
    except Exception as e:
        print(f"가중치 로드 실패: {e}")
        print("랜덤 정책으로 시각화합니다")

    policy.eval()  # 평가 모드 (드롭아웃 등 비활성화)

    # =========================================================================
    # 4. 정보 출력
    # =========================================================================
    print("=" * 50)
    print("학습된 정책 테스트 중...")
    print(f"  관찰 차원: {obs_dim}")
    print(f"  행동 차원: {act_dim}")
    print(f"  환경 수: {args.num_envs}")
    print("=" * 50)

    # 디버그: 로봇 바디 이름 출력 (인덱스 확인용)
    print("\n[DEBUG] 로봇 바디 이름:")
    body_names = env.scene["robot"].data.body_names
    for i, name in enumerate(body_names):
        print(f"  [{i}] {name}")
    print("=" * 50)
    print("종료하려면 Ctrl+C를 누르세요")

    # =========================================================================
    # 5. 추론 루프
    # =========================================================================
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    step_count = 0

    try:
        while simulation_app.is_running():
            # --- 정책 추론 ---
            with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리/속도 최적화)
                actions = policy(policy_obs)
                # 행동 값 클리핑 (-1 ~ 1 범위로 제한)
                actions = torch.clamp(actions, -1.0, 1.0)

            # --- 환경 스텝 ---
            obs, rewards, dones, truncated, info = env.step(actions)
            policy_obs = obs["policy"]
            step_count += 1

            # =========================================================
            # 시각화 마커 업데이트
            # =========================================================
            pen_pos = env.scene["pen"].data.root_pos_w      # (num_envs, 3)
            pen_quat = env.scene["pen"].data.root_quat_w    # (num_envs, 4)

            # 그리퍼 링크 위치 가져오기
            robot = env.scene["robot"]
            l1_pos = robot.data.body_pos_w[:, 7, :]   # 왼쪽 손가락 베이스
            r1_pos = robot.data.body_pos_w[:, 8, :]   # 오른쪽 손가락 베이스
            l2_pos = robot.data.body_pos_w[:, 9, :]   # 왼쪽 손가락 끝
            r2_pos = robot.data.body_pos_w[:, 10, :]  # 오른쪽 손가락 끝

            # 잡기 포인트 계산: 손가락 베이스 중심에서 손가락 방향으로 2cm
            base_center = (l1_pos + r1_pos) / 2.0
            tip_center = (l2_pos + r2_pos) / 2.0
            finger_dir = tip_center - base_center
            finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)
            grasp_point = base_center + finger_dir * 0.02

            # 펜 캡 위치 계산
            half_len = PEN_LENGTH / 2.0
            pen_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            pen_axis_world = quat_rotate_vector(pen_quat, pen_axis)
            cap_pos = pen_pos + pen_axis_world * half_len  # +Z 방향이 캡

            # 그리퍼 Z축 계산 (link_6 방향)
            link6_quat = robot.data.body_quat_w[:, 6, :]
            gripper_z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=pen_pos.device).expand(pen_pos.shape[0], -1)
            gripper_z_world = quat_rotate_vector(link6_quat, gripper_z_axis)

            # 거리 및 정렬 계산
            dist_grasp_to_cap = torch.norm(grasp_point - cap_pos, dim=-1)
            axis_dot = torch.sum(pen_axis_world * gripper_z_world, dim=-1)

            # --- 마커 위치 조합 ---
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
                    marker_indices.append(2)

                # 그리퍼 Z축 마커들 (노란색)
                for j in range(AXIS_POINTS):
                    t = (j + 1) / AXIS_POINTS * AXIS_LENGTH
                    all_positions[base_idx + 2 + AXIS_POINTS + j] = grasp_point[i] + gripper_z_world[i] * t
                    marker_indices.append(3)

            pen_markers.visualize(translations=all_positions, marker_indices=marker_indices)

            # =========================================================
            # 상세 정보 출력 (50스텝마다)
            # =========================================================
            if step_count % 50 == 0:
                print(f"\n{'='*60}")
                print(f"스텝 {step_count}")
                print(f"  평균 보상:        {rewards.mean().item():>8.4f}")
                print(f"  잡기포인트→캡:    {dist_grasp_to_cap.mean().item()*100:>6.2f} cm  (최소: {dist_grasp_to_cap.min().item()*100:.2f})")
                print(f"  Z축 정렬:         {axis_dot.mean().item():>6.3f}  (1.0=평행, -1.0=반대, 0=수직)")
                print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트 중단됨")

    # 리소스 정리
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
