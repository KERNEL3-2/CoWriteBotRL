"""
펜 잡기 강화학습 환경 (Pen Grasp Environment for Isaac Lab)

=== 개요 ===
이 파일은 Isaac Lab 시뮬레이션에서 로봇이 펜을 잡는 작업을 학습하기 위한
강화학습 환경을 정의합니다.

=== 하드웨어 구성 ===
- 로봇: Doosan E0509 (6축 로봇팔)
- 그리퍼: RH-P12-RN-A (4개 관절, 2손가락)
- 대상 객체: 펜 (지름 19.8mm, 길이 117mm, 무게 16.3g)

=== 학습 목표 ===
1. 그리퍼를 펜 캡 위치로 이동
2. 그리퍼 Z축을 펜 Z축과 정렬 (반대 방향으로 접근)
3. 그리퍼를 닫아서 펜 캡을 잡기

=== 파일 구조 ===
1. Scene Configuration (장면 설정): 로봇, 펜, 조명 등 시뮬레이션 객체 정의
2. Action Term (행동 정의): 로봇 관절 제어 방식 정의
3. Observation Terms (관찰 정의): 에이전트가 받는 상태 정보
4. Reward Terms (보상 정의): 학습 목표에 따른 보상 함수들
5. Termination Terms (종료 조건): 에피소드 종료 조건
6. Environment Configuration (환경 설정): 모든 설정을 통합

=== MDP (Markov Decision Process) 구성 ===
- 상태(State): 관절 위치/속도, 그리퍼 위치, 펜 위치/방향 등
- 행동(Action): 6개 팔 관절 + 1개 그리퍼 명령 (총 7차원)
- 보상(Reward): 거리 보상, 정렬 보상, 잡기 성공 보상 등
"""
from __future__ import annotations

import os
import torch

# =============================================================================
# USD 파일 경로 설정
# =============================================================================
# 이 파일 위치를 기준으로 USD 모델 파일 경로를 계산합니다.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")  # 로봇 + 그리퍼 USD
PEN_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "pen.usd")  # 펜 USD (현재 미사용)

# =============================================================================
# Isaac Lab 임포트
# =============================================================================
import isaaclab.envs.mdp as mdp  # MDP 관련 유틸리티 (reset, termination 등)
import isaaclab.sim as sim_utils  # 시뮬레이션 유틸리티 (USD 로딩, 물리 속성 등)
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # 강화학습 환경 베이스 클래스
from isaaclab.managers import EventTermCfg as EventTerm  # 이벤트 설정 (리셋 등)
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 관찰 그룹 설정
from isaaclab.managers import ObservationTermCfg as ObsTerm  # 관찰 항목 설정
from isaaclab.managers import RewardTermCfg as RewTerm  # 보상 항목 설정
from isaaclab.managers import SceneEntityCfg  # 장면 엔티티 참조 설정
from isaaclab.managers import TerminationTermCfg as DoneTerm  # 종료 조건 설정
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg  # 행동 정의
from isaaclab.scene import InteractiveSceneCfg  # 인터랙티브 장면 베이스
from isaaclab.terrains import TerrainImporterCfg  # 지형 설정
from isaaclab.utils import configclass  # 설정 클래스 데코레이터
from isaaclab.actuators import ImplicitActuatorCfg  # 액추에이터(모터) 설정

# =============================================================================
# 펜 물리 사양 (실제 펜 측정값)
# =============================================================================
PEN_DIAMETER = 0.0198  # 펜 최대 지름: 19.8mm
PEN_LENGTH = 0.1207    # 펜 전체 길이: 120.7mm (뚜껑 포함)
PEN_MASS = 0.0163      # 펜 무게: 16.3g


# #############################################################################
#                         1. 장면 설정 (Scene Configuration)
# #############################################################################
@configclass
class PenGraspSceneCfg(InteractiveSceneCfg):
    """
    펜 잡기 장면 설정

    이 클래스는 시뮬레이션 장면에 포함되는 모든 객체를 정의합니다:
    - terrain: 바닥면 (평면)
    - light: 조명 (돔 라이트)
    - robot: Doosan E0509 로봇 + 그리퍼
    - pen: 잡을 대상인 펜 객체
    """

    # =========================================================================
    # 바닥면 설정
    # =========================================================================
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # USD 장면 내 경로
        terrain_type="plane",        # 평면 지형
        debug_vis=False              # 디버그 시각화 끄기
    )

    # =========================================================================
    # 조명 설정 (돔 라이트로 전체 장면 조명)
    # =========================================================================
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),  # 약간 따뜻한 흰색
            intensity=500.0         # 밝기
        ),
    )

    # =========================================================================
    # 로봇 설정: Doosan E0509 + RH-P12-RN-A 그리퍼
    # =========================================================================
    robot: ArticulationCfg = ArticulationCfg(
        # USD 경로 (환경별로 복제됨, {ENV_REGEX_NS}가 env_0, env_1 등으로 치환)
        prim_path="{ENV_REGEX_NS}/Robot",

        # USD 파일 로딩 및 물리 속성 설정
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,           # 로봇 USD 파일 경로
            activate_contact_sensors=True,     # 접촉 센서 활성화 (잡기 감지용)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,         # 중력 적용
                max_depenetration_velocity=5.0,  # 충돌 시 분리 속도 제한
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,        # 자기 충돌 비활성화 (성능 향상)
                solver_position_iteration_count=8,    # 위치 솔버 반복 횟수
                solver_velocity_iteration_count=0,    # 속도 솔버 반복 횟수
            ),
        ),

        # 초기 상태 설정: 로봇 위치 및 관절 각도
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # 월드 원점에 배치
            joint_pos={
                # === Doosan E0509 팔 관절 (6개) ===
                "joint_1": 0.0,      # 베이스 회전
                "joint_2": 0.0,      # 숄더
                "joint_3": 1.57,     # 엘보우 (약 90도 구부림)
                "joint_4": 0.0,      # 손목 1
                "joint_5": -1.57,    # 손목 2 (약 -90도)
                "joint_6": 0.0,      # 손목 3 (엔드이펙터 회전)
                # === RH-P12-RN-A 그리퍼 관절 (4개) ===
                # r1, r2: 오른쪽 손가락, l1, l2: 왼쪽 손가락
                "gripper_rh_r1": 0.0,  # 오른쪽 손가락 1 (열림=0)
                "gripper_rh_r2": 0.0,  # 오른쪽 손가락 2
                "gripper_rh_l1": 0.0,  # 왼쪽 손가락 1
                "gripper_rh_l2": 0.0,  # 왼쪽 손가락 2
            },
        ),

        # 액추에이터(모터) 설정
        actuators={
            # 팔 관절 액추에이터: 높은 토크, 적절한 속도
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],  # joint_1 ~ joint_6
                effort_limit=200.0,   # 최대 토크 (Nm)
                velocity_limit=3.14,  # 최대 속도 (rad/s)
                stiffness=400.0,      # 강성 (위치 제어 게인)
                damping=40.0,         # 감쇠 (속도 제어 게인)
            ),
            # 그리퍼 액추에이터: 높은 강성으로 정밀한 잡기
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_rh_.*"],  # 모든 그리퍼 관절
                effort_limit=50.0,    # 최대 잡기 힘
                velocity_limit=1.0,   # 잡기 속도
                stiffness=2000.0,     # 높은 강성 (정밀 위치 제어)
                damping=100.0,        # 적절한 감쇠
            ),
        },
    )

    # =========================================================================
    # 펜 객체 설정
    # =========================================================================
    # USD 파일로 정의된 펜 (실제 펜 형상: 원뿔대 + 반구)
    pen: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PEN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,     # 중력 비활성화 (공중에 고정)
                kinematic_enabled=False,  # 물리 충돌 활성화 (그리퍼가 밀 수 있음)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PEN_MASS),  # 16.3g
            collision_props=sim_utils.CollisionPropertiesCfg(),    # 충돌 활성화
        ),
        # 초기 위치: 작업 공간 중심 (로봇 앞쪽)
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.3),  # x=50cm 앞, y=0, z=30cm 높이
            # 랜덤화 범위: x(0.3~0.7), y(±0.3), z(0.1~0.5)
        ),
    )


# #############################################################################
#                         2. 행동 정의 (Action Term)
# #############################################################################
class ArmGripperActionTerm(ActionTerm):
    """
    팔 + 그리퍼 통합 행동 제어

    이 클래스는 강화학습 에이전트의 행동을 로봇 관절 명령으로 변환합니다.

    === 행동 공간 (7차원) ===
    - [0:6] 팔 관절 델타 위치 (현재 위치에서의 변화량)
    - [6] 그리퍼 명령 (0=열림, 1=닫힘)

    === 관절 매핑 ===
    - 인덱스 0-5: joint_1 ~ joint_6 (팔)
    - 인덱스 6-9: gripper_rh_r1, r2, l1, l2 (그리퍼, 모두 동일 값)

    === 그리퍼 미믹(Mimic) ===
    그리퍼의 4개 관절은 모두 같은 값을 받습니다 (단일 명령 → 4개 관절).
    이를 통해 행동 공간을 10차원 → 7차원으로 줄입니다.
    """

    _asset: Articulation  # 제어할 로봇 에셋

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        """행동 제어기 초기화"""
        super().__init__(cfg, env)

        # 내부 버퍼 초기화
        # raw_actions: 에이전트에서 받은 원본 행동 (7차원)
        self._raw_actions = torch.zeros(env.num_envs, 7, device=self.device)
        # processed_actions: 처리된 행동 (현재는 동일)
        self._processed_actions = torch.zeros(env.num_envs, 7, device=self.device)
        # joint_pos_target: 실제 관절 타겟 (10차원: 6 팔 + 4 그리퍼)
        self._joint_pos_target = torch.zeros(env.num_envs, 10, device=self.device)

        # 행동 스케일 설정
        self.arm_scale = 0.1     # 팔 행동 스케일 (작은 움직임으로 안정적 학습)
        self.gripper_scale = 1.0  # 그리퍼 스케일 (0~1 범위 그대로 사용)

    @property
    def action_dim(self) -> int:
        """행동 차원: 6 팔 + 1 그리퍼 = 7"""
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        """에이전트의 원본 행동 반환"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """처리된 행동 반환"""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        에이전트 행동 저장

        Args:
            actions: (num_envs, 7) 형태의 행동 텐서
        """
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

    def apply_actions(self):
        """
        행동을 로봇 관절 명령으로 변환 및 적용

        1. 팔 관절: 현재 위치 + (행동 * 스케일) = 델타 위치 제어
        2. 그리퍼: 단일 명령을 4개 관절에 복사 (미믹)
        """
        # 현재 관절 위치 가져오기
        current_pos = self._asset.data.joint_pos

        # --- 팔 관절 제어 (델타 위치) ---
        # 현재 위치에 작은 변화량을 더해서 부드러운 움직임
        arm_delta = self._processed_actions[:, :6] * self.arm_scale
        self._joint_pos_target[:, :6] = current_pos[:, :6] + arm_delta

        # --- 그리퍼 제어 (미믹) ---
        # 단일 그리퍼 명령(인덱스 6)을 4개 관절(인덱스 6-9)에 복사
        gripper_cmd = self._processed_actions[:, 6:7] * self.gripper_scale
        self._joint_pos_target[:, 6:10] = gripper_cmd.repeat(1, 4)

        # 로봇에 관절 위치 타겟 적용
        self._asset.set_joint_position_target(self._joint_pos_target)


@configclass
class ArmGripperActionTermCfg(ActionTermCfg):
    """팔 + 그리퍼 행동 설정 클래스"""
    class_type: type = ArmGripperActionTerm


# #############################################################################
#                         3. 관찰 정의 (Observation Terms)
# #############################################################################

def joint_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    로봇 관절 위치 관찰

    Returns:
        (num_envs, 10) - 6 팔 관절 + 4 그리퍼 관절의 현재 위치 (라디안)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def joint_vel_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    로봇 관절 속도 관찰

    Returns:
        (num_envs, 10) - 각 관절의 현재 각속도 (rad/s)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def get_grasp_point(robot: Articulation) -> torch.Tensor:
    """
    그리퍼의 이상적인 잡기 포인트 계산

    그리퍼가 열리거나 닫혀도 안정적인 위치를 반환합니다.
    왼쪽/오른쪽 손가락 베이스의 중심에서 손가락 방향으로 2cm 앞 지점을 계산합니다.

    === 로봇 바디 인덱스 ===
    - [7] gripper_rh_p12_rn_l1: 왼쪽 손가락 베이스
    - [8] gripper_rh_p12_rn_r1: 오른쪽 손가락 베이스
    - [9] gripper_rh_p12_rn_l2: 왼쪽 손가락 끝
    - [10] gripper_rh_p12_rn_r2: 오른쪽 손가락 끝

    Returns:
        (num_envs, 3) - 잡기 포인트의 월드 좌표
    """
    # 각 손가락 링크의 월드 좌표
    l1 = robot.data.body_pos_w[:, 7, :]   # 왼쪽 베이스
    r1 = robot.data.body_pos_w[:, 8, :]   # 오른쪽 베이스
    l2 = robot.data.body_pos_w[:, 9, :]   # 왼쪽 끝
    r2 = robot.data.body_pos_w[:, 10, :]  # 오른쪽 끝

    # 베이스 중심과 끝 중심 계산
    base_center = (l1 + r1) / 2.0
    tip_center = (l2 + r2) / 2.0

    # 손가락 방향 벡터 (정규화)
    finger_dir = tip_center - base_center
    finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

    # 베이스 중심에서 손가락 방향으로 2cm 앞 = 잡기 포인트
    return base_center + finger_dir * 0.02


def ee_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    그리퍼 잡기 포인트 위치 (환경 원점 기준)

    Returns:
        (num_envs, 3) - 잡기 포인트의 로컬 좌표 (x, y, z)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    grasp_pos_w = get_grasp_point(asset)
    return grasp_pos_w - env.scene.env_origins


def pen_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    펜 중심 위치 (환경 원점 기준)

    Returns:
        (num_envs, 3) - 펜 중심의 로컬 좌표
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def relative_ee_pen_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    그리퍼에서 펜 중심까지의 상대 위치

    에이전트가 펜과의 거리/방향을 직접 학습할 수 있도록 합니다.

    Returns:
        (num_envs, 3) - (펜 위치 - 그리퍼 위치)
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]
    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    return pen_pos - grasp_pos


def pen_orientation_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    펜 방향 (쿼터니언)

    펜의 캡/팁 방향을 구분하기 위해 방향 정보가 필요합니다.

    Returns:
        (num_envs, 4) - 쿼터니언 (w, x, y, z)
    """
    pen: RigidObject = env.scene["pen"]
    return pen.data.root_quat_w


def gripper_state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    그리퍼 열림/닫힘 상태

    4개 그리퍼 관절의 평균값으로 상태를 나타냅니다.
    - 0에 가까움: 그리퍼 열림
    - 1에 가까움: 그리퍼 닫힘

    Returns:
        (num_envs, 1) - 그리퍼 상태 값
    """
    robot: Articulation = env.scene["robot"]
    gripper_pos = robot.data.joint_pos[:, 6:10]  # 인덱스 6-9가 그리퍼
    return gripper_pos.mean(dim=-1, keepdim=True)


def pen_cap_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    펜 캡 위치 (환경 원점 기준)

    펜은 실린더이며, 로컬 Z축이 길이 방향입니다.
    - 포인트 a (팁): -Z 방향 끝
    - 포인트 b (캡): +Z 방향 끝 ← 잡아야 할 위치

    Returns:
        (num_envs, 3) - 펜 캡의 로컬 좌표
    """
    pen: RigidObject = env.scene["pen"]
    pen_pos = pen.data.root_pos_w   # 펜 중심
    pen_quat = pen.data.root_quat_w  # 펜 방향

    # 쿼터니언을 사용하여 로컬 [0,0,1] 벡터를 월드 좌표로 변환
    # 이것이 펜의 캡 방향 (+Z)
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]

    # 쿼터니언 회전 공식: v' = q * v * q^(-1)
    # [0, 0, 1] 벡터를 회전
    cap_dir_x = 2.0 * (qx * qz + qw * qy)
    cap_dir_y = 2.0 * (qy * qz - qw * qx)
    cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)

    # 캡 위치 = 중심 + (펜 길이/2) * 캡 방향
    cap_pos = pen_pos + (PEN_LENGTH / 2) * cap_dir

    return cap_pos - env.scene.env_origins


def relative_ee_cap_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    그리퍼에서 펜 캡까지의 상대 위치

    이것이 에이전트가 최소화해야 할 핵심 정보입니다.

    Returns:
        (num_envs, 3) - (캡 위치 - 그리퍼 위치)
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w

    # 캡 방향 계산 (위와 동일)
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    cap_dir_x = 2.0 * (qx * qz + qw * qy)
    cap_dir_y = 2.0 * (qy * qz - qw * qx)
    cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)

    # 캡 위치
    cap_pos = pen_pos + (PEN_LENGTH / 2) * cap_dir

    return cap_pos - grasp_pos


# #############################################################################
#                         4. 보상 정의 (Reward Terms)
# #############################################################################

def distance_ee_cap_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    그리퍼-캡 거리 보상

    그리퍼가 펜 캡에 가까워질수록 높은 보상을 줍니다.

    보상 공식: 1 / (1 + distance * 10)
    - 거리 0cm: 보상 1.0
    - 거리 10cm: 보상 0.5
    - 거리 50cm: 보상 0.17

    Returns:
        (num_envs,) - 각 환경의 거리 보상
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w

    # 캡 방향 및 위치 계산
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    cap_dir_x = 2.0 * (qx * qz + qw * qy)
    cap_dir_y = 2.0 * (qy * qz - qw * qx)
    cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)
    cap_pos = pen_pos + (PEN_LENGTH / 2) * cap_dir

    # 유클리드 거리 계산
    distance = torch.norm(grasp_pos - cap_pos, dim=-1)

    # 거리 기반 보상 (가까울수록 높음)
    return 1.0 / (1.0 + distance * 10.0)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    행동 크기 패널티

    큰 행동(급격한 움직임)에 페널티를 줘서 부드러운 동작을 유도합니다.

    Returns:
        (num_envs,) - 음수 패널티 값
    """
    return -torch.sum(torch.square(env.action_manager.action), dim=-1) * 0.001


def floor_collision_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    바닥 충돌 패널티

    로봇 링크가 바닥에 너무 가까워지면 패널티를 줍니다.
    link_2 ~ link_6, 그리퍼(인덱스 2-10)의 Z 좌표를 확인합니다.

    임계값: 5cm

    Returns:
        (num_envs,) - 바닥 충돌 시 -1.0, 아니면 0.0
    """
    robot: Articulation = env.scene["robot"]

    # 움직이는 링크들의 Z 좌표 (인덱스 2-10)
    link_z = robot.data.body_pos_w[:, 2:11, 2]  # (num_envs, 9)

    # 5cm 미만인 링크가 있는지 확인
    floor_threshold = 0.05
    below_floor = (link_z < floor_threshold).any(dim=-1).float()

    return -below_floor


def z_axis_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Z축 정렬 보상

    그리퍼의 Z축을 펜의 Z축과 반대 방향으로 정렬하도록 유도합니다.
    (그리퍼가 펜 캡 방향에서 접근해야 잡기 쉬움)

    === 정렬 점수 ===
    - dot product = -1.0: 반대 방향 (정답) → 정렬 점수 1.0
    - dot product = 0.0: 수직 → 정렬 점수 0.0
    - dot product = +1.0: 같은 방향 → 정렬 점수 0.0

    거리 가중치: 가까울수록 정렬 보상이 더 높아짐

    Returns:
        (num_envs,) - 정렬 보상
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    # --- 펜 Z축 계산 ---
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_z_x = 2.0 * (qx * qz + qw * qy)
    pen_z_y = 2.0 * (qy * qz - qw * qx)
    pen_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pen_z_axis = torch.stack([pen_z_x, pen_z_y, pen_z_z], dim=-1)

    # 캡 위치
    cap_pos = pen_pos + (PEN_LENGTH / 2) * pen_z_axis

    # --- 거리 계산 ---
    grasp_pos = get_grasp_point(robot)
    distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

    # --- 그리퍼 Z축 계산 (link_6의 방향에서) ---
    link6_quat = robot.data.body_quat_w[:, 6, :]  # link_6 방향
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)

    # --- 정렬 계산 ---
    # dot product: +1 = 같은 방향, -1 = 반대 방향, 0 = 수직
    # 잡기에서는 반대 방향(-1)이 정답
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # 정렬 점수: -dot_product를 0~1 범위로 클램프
    alignment_score = torch.clamp(-dot_product, min=0.0)  # 0 ~ 1

    # 거리 가중치: 가까울수록 높은 가중치
    distance_weight = 1.0 / (distance_to_cap + 0.05)

    # 최종 정렬 보상 (스케일 조정)
    return alignment_score * distance_weight * 0.1


def pen_displacement_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    펜 이동 패널티

    펜이 충돌로 인해 밀려나면 패널티를 줍니다.
    제대로 잡지 않고 밀기만 하는 행동을 방지합니다.

    Returns:
        (num_envs,) - 펜 속도에 비례한 음수 패널티
    """
    pen: RigidObject = env.scene["pen"]

    # 펜의 선속도 크기
    pen_vel = pen.data.root_lin_vel_w  # (num_envs, 3)
    vel_magnitude = torch.norm(pen_vel, dim=-1)

    # 속도에 비례한 패널티
    return -vel_magnitude * 0.5


def grasp_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    잡기 성공 보상 (큰 보너스)

    === 성공 조건 (모두 충족 시) ===
    1. 그리퍼가 펜 캡에 가까움 (< 3cm)
    2. 그리퍼 Z축이 펜 Z축과 반대 방향 (dot < -0.8)
    3. 그리퍼가 닫힘 (관절 값 > 0.5)

    모든 조건 충족 시 5.0 보상 (가중치 2.0 적용 시 실제 10.0)

    Returns:
        (num_envs,) - 성공 시 5.0, 아니면 0.0
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    # --- 펜 캡 위치 계산 ---
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_z_x = 2.0 * (qx * qz + qw * qy)
    pen_z_y = 2.0 * (qy * qz - qw * qx)
    pen_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pen_z_axis = torch.stack([pen_z_x, pen_z_y, pen_z_z], dim=-1)
    cap_pos = pen_pos + (PEN_LENGTH / 2) * pen_z_axis

    # --- 거리 조건 ---
    grasp_pos = get_grasp_point(robot)
    distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

    # --- 정렬 조건 ---
    link6_quat = robot.data.body_quat_w[:, 6, :]
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # --- 그리퍼 닫힘 조건 ---
    gripper_pos = robot.data.joint_pos[:, 6:10]
    gripper_closed = (gripper_pos > 0.5).all(dim=-1).float()

    # --- 성공 조건 결합 ---
    close_enough = (distance_to_cap < 0.03).float()  # 3cm 이내
    aligned = (dot_product < -0.8).float()           # 반대 방향 정렬

    # 모든 조건 충족 시 큰 보상
    return close_enough * aligned * gripper_closed * 5.0


# #############################################################################
#                         5. 종료 조건 (Termination Terms)
# #############################################################################

def pen_dropped_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    펜 이탈 종료 조건

    펜이 초기 위치에서 15cm 이상 벗어나면 에피소드를 종료합니다.
    펜을 제대로 잡지 않고 밀어버리는 행동을 조기 종료합니다.

    초기 위치: (0.5, 0.0, 0.3) ± 랜덤화

    Returns:
        (num_envs,) - True/False 종료 여부
    """
    pen: RigidObject = env.scene["pen"]

    # 환경 원점 기준 펜 위치
    pen_pos = pen_pos = pen.data.root_pos_w - env.scene.env_origins

    # 초기 위치 중심
    init_pos = torch.tensor([0.5, 0.0, 0.3], device=pen_pos.device)

    # 이동 거리 계산
    displacement = torch.norm(pen_pos - init_pos, dim=-1)

    # 15cm 초과 시 종료
    return displacement > 0.15


# #############################################################################
#                         6. 설정 클래스 (Configuration Classes)
# #############################################################################

@configclass
class ActionsCfg:
    """
    행동 설정

    arm_gripper: 7차원 행동 (6 팔 + 1 그리퍼)
    """
    arm_gripper = ArmGripperActionTermCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """
    관찰 설정

    === 관찰 공간 (총 35차원) ===
    - joint_pos: (10) 관절 위치
    - joint_vel: (10) 관절 속도
    - ee_pos: (3) 그리퍼 위치
    - pen_pos: (3) 펜 위치
    - pen_orientation: (4) 펜 방향 (쿼터니언)
    - relative_ee_pen: (3) 그리퍼→펜 상대 위치
    - relative_ee_cap: (3) 그리퍼→캡 상대 위치
    - gripper_state: (1) 그리퍼 상태
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 네트워크용 관찰 그룹"""

        # 로봇 상태
        joint_pos = ObsTerm(func=joint_pos_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=joint_vel_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        ee_pos = ObsTerm(func=ee_pos_obs, params={"asset_cfg": SceneEntityCfg("robot")})

        # 펜 상태
        pen_pos = ObsTerm(func=pen_pos_obs, params={"asset_cfg": SceneEntityCfg("pen")})
        pen_orientation = ObsTerm(func=pen_orientation_obs)  # (4,) 쿼터니언

        # 상대 위치 (핵심 정보)
        relative_ee_pen = ObsTerm(func=relative_ee_pen_obs)
        relative_ee_cap = ObsTerm(func=relative_ee_cap_obs)  # 캡까지 거리

        # 그리퍼 상태
        gripper_state = ObsTerm(func=gripper_state_obs)  # (1,) 열림/닫힘

        def __post_init__(self):
            self.enable_corruption = False   # 노이즈 없음
            self.concatenate_terms = True    # 모든 관찰을 하나의 텐서로 연결

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    보상 설정

    === 보상 구성 ===
    양수 보상 (목표 행동 유도):
    - distance_to_cap (w=1.0): 캡에 가까워지기
    - z_axis_alignment (w=0.5): 축 정렬
    - grasp_success (w=2.0): 잡기 성공 (5.0 * 2.0 = 10.0)

    음수 보상 (나쁜 행동 억제):
    - floor_collision (w=1.0): 바닥 충돌
    - pen_displacement (w=1.0): 펜 밀기
    - action_rate (w=0.1): 급격한 움직임
    """
    distance_to_cap = RewTerm(func=distance_ee_cap_reward, weight=1.0)
    z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=0.5)
    floor_collision = RewTerm(func=floor_collision_penalty, weight=1.0)
    pen_displacement = RewTerm(func=pen_displacement_penalty, weight=1.0)
    grasp_success = RewTerm(func=grasp_success_reward, weight=2.0)
    action_rate = RewTerm(func=action_rate_penalty, weight=0.1)


@configclass
class TerminationsCfg:
    """
    종료 조건 설정

    - time_out: 에피소드 시간 초과 (10초)
    - pen_dropped: 펜이 15cm 이상 이동
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pen_dropped = DoneTerm(func=pen_dropped_termination)


@configclass
class EventsCfg:
    """
    이벤트 설정 (에피소드 리셋 시 실행)

    매 에피소드 시작 시 로봇과 펜의 초기 상태를 랜덤화합니다.
    다양한 시작 조건에서 학습하여 일반화 성능을 높입니다.
    """
    # 로봇 관절 리셋: 초기 위치 ± 0.1 rad 랜덤화
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),  # 관절 위치 랜덤 범위
            "velocity_range": (-0.1, 0.1),  # 관절 속도 랜덤 범위
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 펜 위치/방향 리셋: 작업 공간 내 랜덤 배치
    reset_pen = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),         # 0.3~0.7m (로봇 앞)
                "y": (-0.3, 0.3),         # ±30cm 좌우
                "z": (-0.2, 0.2),         # 0.1~0.5m 높이
                "roll": (-3.14, 3.14),    # 전체 회전 (펜 뒤집힘 가능)
                "pitch": (-3.14, 3.14),   # 캡이 어느 방향이든 가능
                "yaw": (-3.14, 3.14),     # Z축 회전
            },
            "velocity_range": {},         # 속도는 0으로 시작
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )


# #############################################################################
#                         7. 환경 설정 및 클래스
# #############################################################################

@configclass
class PenGraspEnvCfg(ManagerBasedRLEnvCfg):
    """
    펜 잡기 환경 최종 설정

    이 클래스는 위에서 정의한 모든 설정을 통합합니다.

    === 기본 설정 ===
    - num_envs: 64 (병렬 환경 수)
    - env_spacing: 2.0m (환경 간 간격)
    - decimation: 2 (물리 스텝 / 제어 스텝)
    - dt: 1/60초 (시뮬레이션 주기)
    - episode_length: 10초
    """

    # 장면 설정 (로봇, 펜, 조명 등)
    scene: PenGraspSceneCfg = PenGraspSceneCfg(num_envs=64, env_spacing=2.0)

    # MDP 구성요소
    observations: ObservationsCfg = ObservationsCfg()  # 관찰 (상태)
    actions: ActionsCfg = ActionsCfg()                  # 행동
    rewards: RewardsCfg = RewardsCfg()                  # 보상
    terminations: TerminationsCfg = TerminationsCfg()  # 종료 조건
    events: EventsCfg = EventsCfg()                    # 이벤트 (리셋)

    def __post_init__(self):
        """환경 초기화 후 추가 설정"""
        # 제어 주기: 물리 2스텝당 제어 1스텝
        self.decimation = 2
        # 물리 시뮬레이션 주기: 60Hz
        self.sim.dt = 1.0 / 60.0
        # 에피소드 길이: 10초
        self.episode_length_s = 10.0


class PenGraspEnv(ManagerBasedRLEnv):
    """
    펜 잡기 강화학습 환경 클래스

    Isaac Lab의 ManagerBasedRLEnv를 상속받아 구현됩니다.
    대부분의 기능은 Configuration에서 정의되며,
    이 클래스는 추가 커스터마이징이 필요할 때 확장됩니다.
    """
    cfg: PenGraspEnvCfg
