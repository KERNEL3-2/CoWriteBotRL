"""
E0509 IK 환경 V6 (3DoF 위치 제어 + 자세 자동 정렬)

=== V6 핵심 컨셉 ===
- RL: 위치(x, y, z)만 제어 → 3DoF
- 자세(roll, pitch, yaw): 펜 축 기반 자동 계산 → IK가 처리
- RL 부담 감소, 정밀한 자세 정렬 자동 달성

=== V5.9 대비 변경사항 ===
1. 액션 공간: 6DoF → 3DoF (위치만)
2. 자세 자동 계산: 그리퍼 Z축이 펜 Z축 반대 방향을 향하도록
3. 보상 단순화: 자세 정렬 보상 제거 (자동이므로 불필요)
4. 단계 단순화: 2단계 (APPROACH → GRASP)

=== 단계 (Phase) - 2단계 ===
0. APPROACH: 펜 캡 위치로 접근 (자세는 자동)
1. GRASP: 그리퍼 닫기 → Good Grasp 시 성공

=== Curriculum Levels ===
Level 0: 펜 수직 (tilt_max = 0°)
Level 1: 펜 10° 기울기
Level 2: 펜 20° 기울기
Level 3: 펜 30° 기울기
"""
from __future__ import annotations

import os
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.actuators import ImplicitActuatorCfg

# IK Controller
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import isaaclab.utils.math as math_utils


# =============================================================================
# 경로 및 상수
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")
PEN_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "pen.usd")

PEN_LENGTH = 0.1207  # 120.7mm

# 단계 정의 (V6: 2단계로 대폭 단순화)
PHASE_APPROACH = 0    # RL: 펜 캡 위치로 접근 (자세는 자동)
PHASE_GRASP = 1       # 그리퍼 닫기 → Good Grasp 시 성공

# V6.1: 단계 전환 조건 (더 엄격하게 - 실제로 잡을 수 있는 위치)
APPROACH_TO_GRASP_DIST = 0.015   # 캡까지 거리 < 1.5cm (3cm→1.5cm)
APPROACH_TO_GRASP_PERP = 0.008   # 펜 축에서 벗어난 거리 < 8mm (1.5cm→8mm)

# GRASP 설정
GRIPPER_CLOSE_TARGET = 1.1        # 그리퍼 닫기 목표
GRASP_HOLD_STEPS = 30             # 30 스텝 유지하면 성공

# Good Grasp 조건 (V6: 자세는 자동이므로 dot 제거!)
# 펜 두께 16-17mm → 그리퍼가 완전히 안 닫히고 펜에 걸린 상태
GOOD_GRASP_GRIPPER_MIN = 0.8      # 그리퍼가 어느정도 닫힘
GOOD_GRASP_GRIPPER_MAX = 1.05     # 완전히 닫힌 건 아님 (펜이 있음)
GOOD_GRASP_PERP_DIST = 0.015      # V6: 펜 축 거리 (15mm)

# Curriculum Learning 레벨별 펜 기울기 (라디안)
CURRICULUM_TILT_MAX = {
    0: 0.0,     # Level 0: 수직 (0°)
    1: 0.175,   # Level 1: 10° (≈ 0.175 rad)
    2: 0.35,    # Level 2: 20° (≈ 0.35 rad)
    3: 0.52,    # Level 3: 30° (≈ 0.52 rad)
}


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class E0509IKEnvV6Cfg(DirectRLEnvCfg):
    """E0509 IK 환경 V6 설정 (3DoF 위치 제어 + 자세 자동 정렬)"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 15.0
    action_scale = 0.03       # V6: 위치만 제어하므로 스케일 약간 증가
    action_space = 3          # V6: [Δx, Δy, Δz] 위치만!
    observation_space = 27    # V6: 실제 관찰 차원 (6+6+3+3+3+3+1+1+1)
    state_space = 0

    # Curriculum Learning 설정
    curriculum_level = 0  # 0: 수직, 1: 10°, 2: 20°, 3: 30°

    # 시뮬레이션
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=2,
    )

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # 로봇 설정 (V4: 초기 자세 높이기)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": -0.3,       # V4: -0.5 → -0.3 (더 높게)
                "joint_3": 0.8,        # V4: 1.0 → 0.8 (더 높게)
                "joint_4": 0.0,
                "joint_5": 1.57,       # 90도 - 그리퍼가 아래를 향함
                "joint_6": 0.0,
                "gripper_rh_r1": 0.0,
                "gripper_rh_r2": 0.0,
                "gripper_rh_l1": 0.0,
                "gripper_rh_l2": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=3.14,
                stiffness=800.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_rh_.*"],
                effort_limit=50.0,
                velocity_limit=1.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

    # 펜 설정
    pen_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PEN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.3),
        ),
    )

    # 펜 캡 위치 범위 (V4: z 범위 낮춤 - 더 접근하기 쉽게)
    pen_cap_pos_range = {
        "x": (0.3, 0.5),
        "y": (-0.15, 0.15),
        "z": (0.20, 0.35),  # V3: 0.25~0.40 → V4: 0.20~0.35
    }

    # 펜 방향 랜덤화 (V5: Curriculum Learning으로 동적 설정)
    # pen_tilt_max는 curriculum_level에 따라 _reset_idx에서 결정됨
    pen_tilt_max = 0.0   # 기본값 (curriculum_level=0: 수직)
    pen_yaw_range = (-3.14, 3.14)  # Z축 회전은 전체 (360°)

    # IK 컨트롤러 설정
    ik_method = "dls"
    ik_lambda = 0.05
    ee_body_name = "link_6"
    ee_offset_pos = [0.0, 0.0, 0.15]

    # ==========================================================================
    # 보상 스케일 (V6 - 2단계, 위치만)
    # ==========================================================================
    # V6: 자세 보상 제거, 위치 보상만 사용

    # APPROACH 단계 - 펜 캡으로 접근
    rew_scale_dist_to_cap = -10.0      # 캡까지 거리 페널티
    rew_scale_perp_dist = -5.0         # 펜 축에서 벗어난 거리 페널티
    rew_scale_approach_progress = 5.0  # 접근 진행 보상

    # GRASP 단계 (V6.1: 보상 강화)
    rew_scale_grasp_close = 10.0       # 그리퍼 닫기 보상 (5→10)
    rew_scale_grasp_hold = 20.0        # 위치 유지 보상 (10→20)

    # 공통
    rew_scale_success = 200.0          # V6: 성공 보상
    rew_scale_phase_transition = 100.0 # 전환 보상 (50→100)
    rew_scale_action = -0.01           # V6: 액션 페널티 약간 증가

    # 페널티
    rew_scale_collision = -10.0


class E0509IKEnvV6(DirectRLEnv):
    """E0509 IK 환경 V6 (3DoF 위치 제어 + 자세 자동 정렬)"""

    cfg: E0509IKEnvV6Cfg

    def __init__(self, cfg: E0509IKEnvV6Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스
        self._arm_joint_ids, self._arm_joint_names = self.robot.find_joints(["joint_[1-6]"])
        self._gripper_joint_ids, _ = self.robot.find_joints(["gripper_rh_.*"])
        self._num_arm_joints = len(self._arm_joint_ids)

        # End-effector body index
        body_ids, body_names = self.robot.find_bodies(self.cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected 1 body for '{self.cfg.ee_body_name}', found {len(body_ids)}")
        self._ee_body_idx = body_ids[0]

        # Jacobian index
        if self.robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._arm_joint_ids]

        print(f"[E0509IKEnvV6] EE body: {body_names[0]} (idx={self._ee_body_idx})")
        print(f"[E0509IKEnvV6] Arm joints: {self._arm_joint_names}")

        # IK Controller
        self._ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method=self.cfg.ik_method,
                ik_params={"lambda_val": self.cfg.ik_lambda} if self.cfg.ik_method == "dls" else None,
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

        # EE offset
        self._ee_offset_pos = torch.tensor(
            self.cfg.ee_offset_pos, device=self.device
        ).repeat(self.num_envs, 1)
        self._ee_offset_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        self.action_scale = self.cfg.action_scale

        # 관절 한계
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :6, 0]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :6, 1]

        # 상태 머신 (V6: 2단계)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 이전 거리 (접근 진행 보상용)
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)

        # 그리퍼 상태
        self.gripper_closed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # GRASP 단계 스텝 카운트
        self.grasp_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 성공 카운터
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Phase 출력용 global step 카운터
        self._global_step = 0
        self._phase_print_interval = 5000  # 5000 step마다 출력

    def _setup_scene(self):
        """씬 구성"""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.pen = RigidObject(self.cfg.pen_cfg)
        self.scene.rigid_objects["pen"] = self.pen

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """액션 전처리"""
        self.actions = actions.clone()

    def _compute_auto_orientation(self) -> torch.Tensor:
        """
        V6 핵심: 펜 축 기반 자동 자세 계산

        그리퍼의 Z축이 펜의 Z축 반대 방향(-pen_z)을 향하도록 계산
        그리퍼의 X축은 펜 축과 월드 Z축의 외적으로 결정

        Returns:
            target_quat: [num_envs, 4] 목표 쿼터니언 (wxyz)
        """
        pen_z = self._get_pen_z_axis()  # [num_envs, 3]

        # 그리퍼 Z축 = -펜 Z축 (펜을 위에서 잡음)
        gripper_z = -pen_z

        # 그리퍼 X축: 펜 축과 월드 Z축의 외적
        world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        gripper_x = torch.cross(world_z, gripper_z, dim=-1)
        gripper_x_norm = torch.norm(gripper_x, dim=-1, keepdim=True)

        # 펜이 거의 수직일 때 (외적이 0에 가까움) → 월드 X축 사용
        nearly_vertical = gripper_x_norm.squeeze(-1) < 0.1
        world_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        gripper_x = torch.where(
            nearly_vertical.unsqueeze(-1).expand(-1, 3),
            world_x,
            gripper_x / (gripper_x_norm + 1e-6)
        )

        # 그리퍼 Y축: Z × X
        gripper_y = torch.cross(gripper_z, gripper_x, dim=-1)
        gripper_y = gripper_y / (torch.norm(gripper_y, dim=-1, keepdim=True) + 1e-6)

        # 회전 행렬 → 쿼터니언 변환
        # R = [x | y | z] (각 열이 축)
        rot_matrix = torch.stack([gripper_x, gripper_y, gripper_z], dim=-1)  # [N, 3, 3]

        # 회전 행렬에서 쿼터니언 추출
        target_quat = math_utils.quat_from_matrix(rot_matrix)  # [N, 4] wxyz

        return target_quat

    def _apply_action(self) -> None:
        """
        액션 적용 (V6: 3DoF 위치 + 자동 자세)

        RL이 출력하는 액션: [Δx, Δy, Δz] (3차원)
        자세는 _compute_auto_orientation()으로 자동 계산
        """
        ee_pos_curr, ee_quat_curr = self._compute_ee_pose()

        # ============================================================
        # V6: 3DoF 위치 변화 + 자동 자세 계산
        # ============================================================
        # RL 액션: [Δx, Δy, Δz]
        pos_delta = self.actions * self.action_scale  # [num_envs, 3]

        # 목표 자세: 펜 축 기반 자동 계산
        target_quat = self._compute_auto_orientation()

        # 현재 자세 → 목표 자세의 상대 회전 계산
        # relative_quat = target * inverse(current)
        quat_curr_inv = math_utils.quat_inv(ee_quat_curr)
        quat_delta = math_utils.quat_mul(target_quat, quat_curr_inv)

        # 쿼터니언 → axis-angle 변환 후 스케일링
        rot_delta_axis_angle = math_utils.axis_angle_from_quat(quat_delta)  # [num_envs, 3]

        # 자세 변화 스케일링 (너무 급격한 회전 방지)
        rot_scale = 0.3  # 30% 씩 목표 자세로 접근
        rot_delta_scaled = rot_delta_axis_angle * rot_scale

        # 6DoF IK 명령 조합: [Δx, Δy, Δz, Δrx, Δry, Δrz]
        ik_command = torch.cat([pos_delta, rot_delta_scaled], dim=-1)  # [num_envs, 6]

        # IK 컨트롤러 실행
        self._ik_controller.set_command(ik_command, ee_pos_curr, ee_quat_curr)

        jacobian = self._compute_ee_jacobian()
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]

        joint_pos_target = self._ik_controller.compute(
            ee_pos_curr, ee_quat_curr, jacobian, joint_pos
        )

        joint_pos_target = torch.clamp(
            joint_pos_target,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # 그리퍼 제어 (V6: GRASP 단계에서만 닫기)
        gripper_target = torch.zeros(self.num_envs, 4, device=self.device)

        grasp_mask = (self.phase == PHASE_GRASP)
        if grasp_mask.any():
            gripper_target[grasp_mask] = GRIPPER_CLOSE_TARGET
            self.gripper_closed[grasp_mask] = True

        full_target = torch.zeros(self.num_envs, 10, device=self.device)
        full_target[:, :6] = joint_pos_target
        full_target[:, 6:] = gripper_target

        self.robot.set_joint_position_target(full_target)

    def _compute_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """End-effector pose 계산"""
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._ee_offset_pos, self._ee_offset_rot
        )

        return ee_pos_b, ee_quat_b

    def _compute_ee_jacobian(self) -> torch.Tensor:
        """End-effector Jacobian 계산"""
        jacobian_w = self.robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]

        base_rot = self.robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))

        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        jacobian_b[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._ee_offset_pos),
            jacobian_b[:, 3:, :]
        )

        return jacobian_b

    # ==========================================================================
    # 펜 축 기준 계산 함수들
    # ==========================================================================
    def _compute_axis_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """펜 축 기준 메트릭 계산"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        pen_axis = self._get_pen_z_axis()

        grasp_to_cap = cap_pos - grasp_pos
        axis_distance = torch.sum(grasp_to_cap * pen_axis, dim=-1)

        projection = axis_distance.unsqueeze(-1) * pen_axis
        perpendicular_vec = grasp_to_cap - projection
        perpendicular_dist = torch.norm(perpendicular_vec, dim=-1)

        on_correct_side = axis_distance < 0

        return perpendicular_dist, axis_distance, on_correct_side

    def _check_pen_collision(self) -> torch.Tensor:
        """펜 몸체와의 충돌 감지"""
        grasp_pos = self._get_grasp_point()
        pen_pos = self.pen.data.root_pos_w
        pen_axis = self._get_pen_z_axis()

        grasp_to_pen = pen_pos - grasp_pos
        proj_length = torch.sum(grasp_to_pen * pen_axis, dim=-1)

        proj_vec = proj_length.unsqueeze(-1) * pen_axis
        perp_to_pen = grasp_to_pen - proj_vec
        perp_dist = torch.norm(perp_to_pen, dim=-1)

        in_pen_range = (torch.abs(proj_length) < PEN_LENGTH / 2) & (proj_length < 0)
        collision = in_pen_range & (perp_dist < 0.03)

        return collision

    def _get_observations(self) -> dict:
        """
        관찰값 계산 (V6: 24차원으로 단순화)

        자세 관련 정보 제거 (자동 계산이므로 불필요)
        위치 정보에 집중
        """
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        grasp_pos = self._get_grasp_point()
        grasp_pos_local = grasp_pos - self.scene.env_origins

        cap_pos = self._get_pen_cap_pos()
        cap_pos_local = cap_pos - self.scene.env_origins

        rel_pos = cap_pos - grasp_pos  # 캡까지 상대 위치

        # V6: 자세 정보 추가 (RL이 자세를 직접 제어하지 않지만, 현재 상태 파악용)
        pen_z = self._get_pen_z_axis()  # 펜 축 방향 (자동 자세 계산에 사용됨)

        perpendicular_dist, axis_distance, _ = self._compute_axis_metrics()
        distance_to_cap = torch.norm(rel_pos, dim=-1)

        # 현재 단계 (정규화) - V6: 2단계 (0~1)
        phase_normalized = self.phase.float()

        obs = torch.cat([
            joint_pos,                           # 6
            joint_vel,                           # 6
            grasp_pos_local,                     # 3
            cap_pos_local,                       # 3
            rel_pos,                             # 3
            pen_z,                               # 3 (자동 자세 계산 방향)
            perpendicular_dist.unsqueeze(-1),   # 1 (펜 축까지 거리)
            distance_to_cap.unsqueeze(-1),       # 1 (캡까지 거리)
            phase_normalized.unsqueeze(-1),      # 1
        ], dim=-1)  # 총 24

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        보상 계산 (V6: 2단계 단순화, 위치 중심)

        자세 보상 제거 (자동 계산이므로 불필요!)
        """
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()

        perpendicular_dist, axis_distance, on_correct_side = self._compute_axis_metrics()
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        self.phase_step_count += 1

        # =========================================================
        # APPROACH 단계 - 펜 캡으로 접근 (자세는 자동!)
        # =========================================================
        approach_mask = (self.phase == PHASE_APPROACH)
        if approach_mask.any():
            # 캡까지 거리 페널티
            rewards[approach_mask] += self.cfg.rew_scale_dist_to_cap * distance_to_cap[approach_mask]

            # 펜 축에서 벗어난 거리 페널티
            rewards[approach_mask] += self.cfg.rew_scale_perp_dist * perpendicular_dist[approach_mask]

            # 접근 진행 보상 (이전보다 가까워지면 보상)
            progress = self.prev_distance_to_cap[approach_mask] - distance_to_cap[approach_mask]
            rewards[approach_mask] += self.cfg.rew_scale_approach_progress * torch.clamp(progress * 50, min=0, max=1)

            # 펜 축 근처 보너스 (perp_dist < 3cm)
            near_axis = perpendicular_dist[approach_mask] < 0.03
            rewards[approach_mask] += 2.0 * near_axis.float()

        # =========================================================
        # GRASP 단계 - 그리퍼 닫기 + 위치 유지
        # =========================================================
        grasp_mask = (self.phase == PHASE_GRASP)
        if grasp_mask.any():
            self.grasp_step_count[grasp_mask] += 1

            # 위치 유지 보상 (캡 근처에 머무르기)
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_hold * torch.exp(-distance_to_cap[grasp_mask] * 30.0)

            # 그리퍼 닫기 보상
            gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]
            gripper_closed_amount = gripper_pos.mean(dim=-1)
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_close * gripper_closed_amount[grasp_mask]

        # =========================================================
        # 단계 전환: APPROACH → GRASP
        # =========================================================
        transition_to_grasp = (
            approach_mask &
            (distance_to_cap < APPROACH_TO_GRASP_DIST) &
            (perpendicular_dist < APPROACH_TO_GRASP_PERP)
        )
        if transition_to_grasp.any():
            self.phase[transition_to_grasp] = PHASE_GRASP
            rewards[transition_to_grasp] += self.cfg.rew_scale_phase_transition
            self.phase_step_count[transition_to_grasp] = 0
            self.grasp_step_count[transition_to_grasp] = 0

        # =========================================================
        # 성공 조건 (V6: Good Grasp + 30스텝 유지)
        # V6에서는 dot(자세) 조건 제거! → 자동 계산이므로
        # =========================================================
        gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]
        gripper_closed_amount = gripper_pos.mean(dim=-1)

        good_grasp = (
            (perpendicular_dist < GOOD_GRASP_PERP_DIST) &
            (gripper_closed_amount > GOOD_GRASP_GRIPPER_MIN) &
            (gripper_closed_amount < GOOD_GRASP_GRIPPER_MAX)
        )

        success = grasp_mask & good_grasp & (self.grasp_step_count >= GRASP_HOLD_STEPS)
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # =========================================================
        # 페널티
        # =========================================================
        # 펜 몸체 충돌 페널티 (APPROACH 단계에서만)
        collision = self._check_pen_collision()
        collision_penalty_mask = collision & approach_mask
        if collision_penalty_mask.any():
            rewards[collision_penalty_mask] += self.cfg.rew_scale_collision

        # 액션 페널티 (작은 움직임 유도)
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # =========================================================
        # 이전 거리 업데이트
        # =========================================================
        self.prev_distance_to_cap = distance_to_cap.clone()

        # =========================================================
        # Phase 분포 출력 + TensorBoard 기록
        # =========================================================
        self._global_step += 1

        # TensorBoard 기록
        phase_stats = self.get_phase_stats()
        if "log" not in self.extras:
            self.extras["log"] = {}

        total_envs = float(self.num_envs)
        self.extras["log"]["Phase/approach_ratio"] = phase_stats['approach'] / total_envs
        self.extras["log"]["Phase/grasp_ratio"] = phase_stats['grasp'] / total_envs
        self.extras["log"]["Phase/total_success"] = float(phase_stats['total_success'])

        # 메트릭 로깅
        self.extras["log"]["Metrics/dist_to_cap_mean"] = distance_to_cap.mean().item()
        self.extras["log"]["Metrics/perp_dist_mean"] = perpendicular_dist.mean().item()

        # 콘솔 출력 (N step마다)
        if self._global_step % self._phase_print_interval == 0:
            print(f"  [Step {self._global_step}] Phase: "
                  f"APPROACH:{phase_stats['approach']} GRASP:{phase_stats['grasp']} "
                  f"| Success:{phase_stats['total_success']}", flush=True)
            print(f"    → dist_cap={distance_to_cap.mean().item():.4f}m, "
                  f"perp_dist={perpendicular_dist.mean().item():.4f}m", flush=True)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        종료 조건 (V6: Good Grasp + 30스텝)

        V6에서는 dot(자세) 조건 제거! → 자동 계산이므로
        """
        perpendicular_dist, _, _ = self._compute_axis_metrics()

        gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]
        gripper_closed_amount = gripper_pos.mean(dim=-1)

        # V6: dot 조건 제거! (자세는 자동)
        good_grasp = (
            (perpendicular_dist < GOOD_GRASP_PERP_DIST) &
            (gripper_closed_amount > GOOD_GRASP_GRIPPER_MIN) &
            (gripper_closed_amount < GOOD_GRASP_GRIPPER_MAX)
        )

        grasp_mask = (self.phase == PHASE_GRASP)
        success = grasp_mask & good_grasp & (self.grasp_step_count >= GRASP_HOLD_STEPS)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """환경 리셋 (V6: 단순화)"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        super()._reset_idx(env_ids)

        # 로봇 리셋
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, :6] += sample_uniform(
            -0.05, 0.05,
            (len(env_ids), 6),
            device=self.device,
        )
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 펜 리셋 (위치 + Curriculum Learning 기울기)
        pen_state = self.pen.data.default_root_state[env_ids].clone()

        cap_pos_x = sample_uniform(
            self.cfg.pen_cap_pos_range["x"][0], self.cfg.pen_cap_pos_range["x"][1],
            (len(env_ids),), device=self.device
        )
        cap_pos_y = sample_uniform(
            self.cfg.pen_cap_pos_range["y"][0], self.cfg.pen_cap_pos_range["y"][1],
            (len(env_ids),), device=self.device
        )
        cap_pos_z = sample_uniform(
            self.cfg.pen_cap_pos_range["z"][0], self.cfg.pen_cap_pos_range["z"][1],
            (len(env_ids),), device=self.device
        )

        pen_center_x = cap_pos_x
        pen_center_y = cap_pos_y
        pen_center_z = cap_pos_z - PEN_LENGTH / 2

        pen_state[:, 0] = self.scene.env_origins[env_ids, 0] + pen_center_x
        pen_state[:, 1] = self.scene.env_origins[env_ids, 1] + pen_center_y
        pen_state[:, 2] = self.scene.env_origins[env_ids, 2] + pen_center_z

        # Curriculum Learning: 펜 기울기 설정
        curriculum_tilt_max = CURRICULUM_TILT_MAX.get(self.cfg.curriculum_level, 0.0)
        tilt = sample_uniform(0, curriculum_tilt_max, (len(env_ids),), device=self.device)
        azimuth = sample_uniform(0, 2 * 3.14159, (len(env_ids),), device=self.device)
        yaw = sample_uniform(
            self.cfg.pen_yaw_range[0], self.cfg.pen_yaw_range[1],
            (len(env_ids),), device=self.device
        )
        pen_quat = self._cone_angle_to_quat(tilt, azimuth, yaw)
        pen_state[:, 3:7] = pen_quat

        self.pen.write_root_pose_to_sim(pen_state[:, :7], env_ids)
        self.pen.write_root_velocity_to_sim(pen_state[:, 7:], env_ids)

        # 상태 리셋 (V6: 2단계)
        self.phase[env_ids] = PHASE_APPROACH
        self.phase_step_count[env_ids] = 0
        self.gripper_closed[env_ids] = False
        self.grasp_step_count[env_ids] = 0

        self._ik_controller.reset(env_ids_tensor)

        # 초기 거리 계산
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        self.prev_distance_to_cap[env_ids] = torch.norm(
            grasp_pos[env_ids] - cap_pos[env_ids], dim=-1
        )

    # ==========================================================================
    # 헬퍼 함수들
    # ==========================================================================

    def _get_grasp_point(self) -> torch.Tensor:
        """그리퍼 잡기 포인트"""
        l1 = self.robot.data.body_pos_w[:, 7, :]
        r1 = self.robot.data.body_pos_w[:, 8, :]
        l2 = self.robot.data.body_pos_w[:, 9, :]
        r2 = self.robot.data.body_pos_w[:, 10, :]

        base_center = (l1 + r1) / 2.0
        tip_center = (l2 + r2) / 2.0

        finger_dir = tip_center - base_center
        finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

        return base_center + finger_dir * 0.02

    def _get_pen_cap_pos(self) -> torch.Tensor:
        """펜 캡 위치"""
        pen_pos = self.pen.data.root_pos_w
        pen_quat = self.pen.data.root_quat_w

        qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
        cap_dir_x = 2.0 * (qx * qz + qw * qy)
        cap_dir_y = 2.0 * (qy * qz - qw * qx)
        cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)

        return pen_pos + (PEN_LENGTH / 2) * cap_dir

    def _get_gripper_z_axis(self) -> torch.Tensor:
        """그리퍼 Z축 방향"""
        link6_quat = self.robot.data.body_quat_w[:, 6, :]
        qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]

        z_x = 2.0 * (qx * qz + qw * qy)
        z_y = 2.0 * (qy * qz - qw * qx)
        z_z = 1.0 - 2.0 * (qx * qx + qy * qy)

        return torch.stack([z_x, z_y, z_z], dim=-1)

    def _get_pen_z_axis(self) -> torch.Tensor:
        """펜 Z축 방향"""
        pen_quat = self.pen.data.root_quat_w
        qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]

        z_x = 2.0 * (qx * qz + qw * qy)
        z_y = 2.0 * (qy * qz - qw * qx)
        z_z = 1.0 - 2.0 * (qx * qx + qy * qy)

        return torch.stack([z_x, z_y, z_z], dim=-1)

    def _euler_to_quat(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """오일러 → 쿼터니언"""
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=-1)

    def _cone_angle_to_quat(self, tilt: torch.Tensor, azimuth: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """원뿔 각도 → 쿼터니언

        Args:
            tilt: Z축에서 기울어진 각도 (0 ~ max_tilt)
            azimuth: 기울어진 방향 (0 ~ 2π, XY 평면에서)
            yaw: 펜 자체 Z축 회전 (0 ~ 2π)

        Returns:
            쿼터니언 [w, x, y, z]

        구현:
            1. Y축 기준 tilt 회전 (기울기)
            2. Z축 기준 azimuth 회전 (기울어진 방향)
            3. Z축 기준 yaw 회전 (펜 자체 회전)
        """
        # 1. Y축 기준 tilt 회전
        q1_w = torch.cos(tilt * 0.5)
        q1_x = torch.zeros_like(tilt)
        q1_y = torch.sin(tilt * 0.5)
        q1_z = torch.zeros_like(tilt)

        # 2. Z축 기준 azimuth 회전
        q2_w = torch.cos(azimuth * 0.5)
        q2_x = torch.zeros_like(azimuth)
        q2_y = torch.zeros_like(azimuth)
        q2_z = torch.sin(azimuth * 0.5)

        # 3. Z축 기준 yaw 회전
        q3_w = torch.cos(yaw * 0.5)
        q3_x = torch.zeros_like(yaw)
        q3_y = torch.zeros_like(yaw)
        q3_z = torch.sin(yaw * 0.5)

        # 쿼터니언 곱셈: q2 * q1 (azimuth 먼저, 그 다음 tilt)
        # q = q2 * q1
        r1_w = q2_w * q1_w - q2_z * q1_y
        r1_x = q2_w * q1_x + q2_z * q1_z
        r1_y = q2_w * q1_y + q2_z * q1_x
        r1_z = q2_z * q1_w + q2_w * q1_z

        # 최종: q3 * (q2 * q1) = q3 * r1
        w = q3_w * r1_w - q3_z * r1_z
        x = q3_w * r1_x + q3_z * r1_y
        y = q3_w * r1_y - q3_z * r1_x
        z = q3_w * r1_z + q3_z * r1_w

        return torch.stack([w, x, y, z], dim=-1)

    def get_phase_stats(self) -> dict:
        """단계별 통계 (V6: 2단계)"""
        current_phases = torch.bincount(self.phase, minlength=2)  # V6: 2단계
        return {
            "approach": current_phases[0].item(),
            "grasp": current_phases[1].item(),
            "total_success": self.success_count.sum().item(),
            "curriculum_level": self.cfg.curriculum_level,
        }


@configclass
class E0509IKEnvV6Cfg_PLAY(E0509IKEnvV6Cfg):
    """V6 테스트용 설정"""

    def __post_init__(self):
        self.scene.num_envs = 50


# Curriculum Level별 설정 (V6 학습용)
@configclass
class E0509IKEnvV6Cfg_L0(E0509IKEnvV6Cfg):
    """Level 0: 펜 수직 (기본 동작 학습)"""
    curriculum_level = 0


@configclass
class E0509IKEnvV6Cfg_L1(E0509IKEnvV6Cfg):
    """Level 1: 펜 10° 기울기"""
    curriculum_level = 1


@configclass
class E0509IKEnvV6Cfg_L2(E0509IKEnvV6Cfg):
    """Level 2: 펜 20° 기울기"""
    curriculum_level = 2


@configclass
class E0509IKEnvV6Cfg_L3(E0509IKEnvV6Cfg):
    """Level 3: 펜 30° 기울기 (최종 목표)"""
    curriculum_level = 3
