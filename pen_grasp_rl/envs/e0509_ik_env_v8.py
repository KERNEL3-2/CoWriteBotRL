"""
E0509 IK 환경 V8 (V7 기반 + 관절 제한 페널티 + 수직거리 7cm)

=== V8 핵심 변경사항 (V7 대비) ===
1. SUCCESS_DIST_TO_CAP: 3cm → 7cm (수직거리 증가)
2. 관절 제한 페널티 추가 (한계에 가까워지면 페널티)
3. 관절 제한 마진: 90% 넘으면 페널티 시작

=== V7에서 유지 ===
- RL: 위치(x, y, z)만 제어 → 3DoF
- 자세: 펜 축 기반 자동 계산 → IK가 처리
- GRASP 제거: 실제 로봇에서 별도 처리 (Sim2Real Gap 회피)
- Curriculum Learning

=== 성공 조건 ===
- 캡까지 거리 < 7cm (변경!)
- 펜 축 정렬 (perp_dist < 1cm)
- 캡 위에 있음 (on_correct_side)
- 30 스텝 유지
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

# 단계 정의 (V8: APPROACH만!)
PHASE_APPROACH = 0    # RL: 펜 캡 위치로 접근 (자세는 자동)

# V8 성공 조건 (수직거리 7cm로 증가!)
SUCCESS_DIST_TO_CAP = 0.07       # 캡까지 거리 < 7cm (V7: 3cm → V8: 7cm)
SUCCESS_PERP_DIST = 0.01         # 펜 축에서 벗어난 거리 < 1cm
SUCCESS_HOLD_STEPS = 30          # 30 스텝 유지하면 성공

# V8: 관절 제한 페널티 설정
JOINT_LIMIT_MARGIN = 0.9         # 90% 넘으면 페널티 시작

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
class E0509IKEnvV8Cfg(DirectRLEnvCfg):
    """E0509 IK 환경 V8 설정 (관절 제한 + 수직거리 7cm)"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 15.0
    action_scale = 0.03       # 위치만 제어하므로 스케일 약간 증가
    action_space = 3          # [Δx, Δy, Δz] 위치만!
    observation_space = 27    # 실제 관찰 차원 (6+6+3+3+3+3+1+1+1)
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

    # 로봇 설정
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
                "joint_2": -0.3,
                "joint_3": 0.8,
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

    # 펜 캡 위치 범위
    pen_cap_pos_range = {
        "x": (0.3, 0.5),
        "y": (-0.15, 0.15),
        "z": (0.20, 0.35),
    }

    # 펜 방향 랜덤화 (Curriculum Learning으로 동적 설정)
    pen_tilt_min = None  # None이면 curriculum_level 사용
    pen_tilt_max = None  # None이면 curriculum_level 사용
    pen_yaw_range = (-3.14, 3.14)  # Z축 회전은 전체 (360°)

    # IK 컨트롤러 설정
    ik_method = "dls"
    ik_lambda = 0.05
    ee_body_name = "link_6"
    ee_offset_pos = [0.0, 0.0, 0.15]

    # ==========================================================================
    # 보상 스케일 (V8)
    # ==========================================================================
    # APPROACH 단계 - 펜 캡으로 접근
    rew_scale_dist_to_cap = -15.0      # 캡까지 거리 페널티 (선형)
    rew_scale_dist_exp = 10.0          # 캡까지 거리 보상 (지수)
    rew_scale_perp_dist = -8.0         # 펜 축 거리 페널티
    rew_scale_perp_exp = 5.0           # 펜 축 거리 보상 (지수)
    rew_scale_approach_progress = 3.0  # 접근 진행 보상

    # 자세 정렬 보상 (dot product 기반)
    rew_scale_alignment = 5.0          # 자세 정렬 보상
    rew_scale_ready_bonus = 10.0       # 성공 조건 근접 보너스

    # 공통
    rew_scale_success = 100.0          # 성공 보상
    rew_scale_action = -0.01           # 액션 페널티

    # 페널티
    rew_scale_collision = -10.0

    # V8: 관절 제한 페널티
    rew_scale_joint_limit = -5.0       # 관절 제한 페널티


class E0509IKEnvV8(DirectRLEnv):
    """E0509 IK 환경 V8 (관절 제한 + 수직거리 7cm)"""

    cfg: E0509IKEnvV8Cfg

    def __init__(self, cfg: E0509IKEnvV8Cfg, render_mode: str | None = None, **kwargs):
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

        print(f"[E0509IKEnvV8] EE body: {body_names[0]} (idx={self._ee_body_idx})")
        print(f"[E0509IKEnvV8] Arm joints: {self._arm_joint_names}")
        print(f"[E0509IKEnvV8] SUCCESS_DIST_TO_CAP: {SUCCESS_DIST_TO_CAP*100:.0f}cm")
        print(f"[E0509IKEnvV8] Joint limit penalty enabled (margin: {JOINT_LIMIT_MARGIN*100:.0f}%)")

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

        # V8: APPROACH만 (단일 단계)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 이전 거리 (접근 진행 보상용)
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)

        # V8: 성공 조건 유지 카운트
        self.success_hold_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

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
        펜 축 기반 자동 자세 계산

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
        rot_matrix = torch.stack([gripper_x, gripper_y, gripper_z], dim=-1)  # [N, 3, 3]
        target_quat = math_utils.quat_from_matrix(rot_matrix)  # [N, 4] wxyz

        return target_quat

    def _apply_action(self) -> None:
        """
        액션 적용 (V8: 3DoF 위치 + 자동 자세)

        RL이 출력하는 액션: [Δx, Δy, Δz] (3차원)
        자세는 _compute_auto_orientation()으로 자동 계산
        """
        ee_pos_curr, ee_quat_curr = self._compute_ee_pose()

        # 3DoF 위치 변화 + 자동 자세 계산
        pos_delta = self.actions * self.action_scale  # [num_envs, 3]

        # 목표 자세: 펜 축 기반 자동 계산
        target_quat = self._compute_auto_orientation()

        # 현재 자세 → 목표 자세의 상대 회전 계산
        quat_curr_inv = math_utils.quat_inv(ee_quat_curr)
        quat_delta = math_utils.quat_mul(target_quat, quat_curr_inv)

        # 쿼터니언 → axis-angle 변환 후 스케일링
        rot_delta_axis_angle = math_utils.axis_angle_from_quat(quat_delta)

        # 자세 변화 스케일링 (너무 급격한 회전 방지)
        rot_scale = 0.3
        rot_delta_scaled = rot_delta_axis_angle * rot_scale

        # 6DoF IK 명령 조합
        ik_command = torch.cat([pos_delta, rot_delta_scaled], dim=-1)

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

        # 그리퍼 항상 열린 상태
        gripper_target = torch.zeros(self.num_envs, 4, device=self.device)

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
    # V8: 관절 제한 페널티 계산
    # ==========================================================================
    def _compute_joint_limit_penalty(self) -> torch.Tensor:
        """
        관절 제한 페널티 계산

        관절이 한계의 90% 이상에 도달하면 페널티 부여
        한계에 가까울수록 페널티 증가 (선형)

        Returns:
            penalty: [num_envs] 각 환경의 관절 제한 페널티 합
        """
        joint_pos = self.robot.data.joint_pos[:, :6]  # [num_envs, 6]

        # 각 관절의 범위 계산
        joint_range = self.robot_dof_upper_limits - self.robot_dof_lower_limits  # [6]
        joint_center = (self.robot_dof_upper_limits + self.robot_dof_lower_limits) / 2  # [6]

        # 현재 위치가 범위의 몇 %인지 계산 (0 = 중앙, 1 = 한계)
        normalized_pos = torch.abs(joint_pos - joint_center) / (joint_range / 2 + 1e-6)  # [num_envs, 6]

        # 90% 이상일 때만 페널티 (0~1 범위로 정규화)
        over_margin = torch.clamp(normalized_pos - JOINT_LIMIT_MARGIN, min=0) / (1 - JOINT_LIMIT_MARGIN)

        # 각 환경의 모든 관절 페널티 합산
        penalty = over_margin.sum(dim=-1)  # [num_envs]

        return penalty

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
        """관찰값 계산"""
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        grasp_pos = self._get_grasp_point()
        grasp_pos_local = grasp_pos - self.scene.env_origins

        cap_pos = self._get_pen_cap_pos()
        cap_pos_local = cap_pos - self.scene.env_origins

        rel_pos = cap_pos - grasp_pos

        pen_z = self._get_pen_z_axis()

        perpendicular_dist, axis_distance, _ = self._compute_axis_metrics()
        distance_to_cap = torch.norm(rel_pos, dim=-1)

        phase_normalized = self.phase.float()

        obs = torch.cat([
            joint_pos,                           # 6
            joint_vel,                           # 6
            grasp_pos_local,                     # 3
            cap_pos_local,                       # 3
            rel_pos,                             # 3
            pen_z,                               # 3
            perpendicular_dist.unsqueeze(-1),   # 1
            distance_to_cap.unsqueeze(-1),       # 1
            phase_normalized.unsqueeze(-1),      # 1
        ], dim=-1)  # 총 27

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """보상 계산 (V8: 관절 제한 페널티 추가)"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()

        perpendicular_dist, axis_distance, on_correct_side = self._compute_axis_metrics()
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        self.phase_step_count += 1

        # 자세 정렬 계산
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()
        dot = torch.sum(gripper_z * pen_z, dim=-1)

        # =========================================================
        # APPROACH 단계
        # =========================================================
        # 캡까지 거리 - 선형 페널티 + 지수 보상
        rewards += self.cfg.rew_scale_dist_to_cap * distance_to_cap
        rewards += self.cfg.rew_scale_dist_exp * torch.exp(-distance_to_cap * 15.0)

        # 펜 축 거리 - 선형 페널티 + 지수 보상
        rewards += self.cfg.rew_scale_perp_dist * perpendicular_dist
        rewards += self.cfg.rew_scale_perp_exp * torch.exp(-perpendicular_dist * 50.0)

        # 접근 진행 보상 (이전보다 가까워지면 보상)
        progress = self.prev_distance_to_cap - distance_to_cap
        rewards += self.cfg.rew_scale_approach_progress * torch.clamp(progress * 50, min=0, max=1)

        # 자세 정렬 보상
        alignment_reward = (-dot - 0.5) * 0.5
        alignment_reward = torch.clamp(alignment_reward, min=0)
        rewards += self.cfg.rew_scale_alignment * alignment_reward

        # 캡 위에 있을 때 보너스
        above_cap_bonus = on_correct_side.float() * 0.5
        rewards += above_cap_bonus

        # =========================================================
        # 성공 조건 근접 보너스
        # =========================================================
        near_success = (
            (distance_to_cap < SUCCESS_DIST_TO_CAP * 2) &  # 14cm 이내
            (perpendicular_dist < SUCCESS_PERP_DIST * 2) &  # 2cm 이내
            on_correct_side
        )
        rewards[near_success] += self.cfg.rew_scale_ready_bonus * 0.5

        # =========================================================
        # V8 성공 조건 (수직거리 7cm)
        # =========================================================
        success_condition = (
            (distance_to_cap < SUCCESS_DIST_TO_CAP) &      # 캡까지 7cm 이내
            (perpendicular_dist < SUCCESS_PERP_DIST) &     # 펜 축에서 1cm 이내
            on_correct_side                                 # 캡 위에 있음
        )

        # 성공 조건 유지 카운트
        self.success_hold_count[success_condition] += 1
        self.success_hold_count[~success_condition] = 0

        # 30 스텝 유지하면 성공
        success = self.success_hold_count >= SUCCESS_HOLD_STEPS
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # =========================================================
        # 페널티
        # =========================================================
        # 펜 몸체 충돌 페널티
        collision = self._check_pen_collision()
        if collision.any():
            rewards[collision] += self.cfg.rew_scale_collision

        # 캡을 지나쳤을 때 페널티
        passed_cap = ~on_correct_side
        rewards[passed_cap] -= 1.0

        # 액션 페널티
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # V8: 관절 제한 페널티
        joint_limit_penalty = self._compute_joint_limit_penalty()
        rewards += self.cfg.rew_scale_joint_limit * joint_limit_penalty

        # =========================================================
        # 이전 거리 업데이트
        # =========================================================
        self.prev_distance_to_cap = distance_to_cap.clone()

        # =========================================================
        # TensorBoard 기록
        # =========================================================
        self._global_step += 1

        if "log" not in self.extras:
            self.extras["log"] = {}

        self.extras["log"]["Phase/success_hold_count_mean"] = self.success_hold_count.float().mean().item()
        self.extras["log"]["Phase/total_success"] = float(self.success_count.sum().item())
        self.extras["log"]["Phase/on_correct_side_ratio"] = on_correct_side.float().mean().item()

        self.extras["log"]["Metrics/dist_to_cap_mean"] = distance_to_cap.mean().item()
        self.extras["log"]["Metrics/perp_dist_mean"] = perpendicular_dist.mean().item()
        self.extras["log"]["Metrics/dot_mean"] = dot.mean().item()

        # V8: 관절 제한 페널티 로깅
        self.extras["log"]["Metrics/joint_limit_penalty"] = joint_limit_penalty.mean().item()

        # 콘솔 출력
        if self._global_step % self._phase_print_interval == 0:
            dot_val = dot.mean().item()
            on_correct_pct = on_correct_side.float().mean().item() * 100
            jl_penalty = joint_limit_penalty.mean().item()

            print(f"  [Step {self._global_step}] V8 APPROACH (7cm target)", flush=True)
            print(f"    → dist_cap={distance_to_cap.mean().item()*100:.2f}cm, "
                  f"perp={perpendicular_dist.mean().item()*100:.2f}cm, "
                  f"dot={dot_val:.3f}", flush=True)
            print(f"    → on_correct_side={on_correct_pct:.0f}%, "
                  f"success_hold={self.success_hold_count.float().mean().item():.1f}, "
                  f"jl_penalty={jl_penalty:.3f}", flush=True)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        success = self.success_hold_count >= SUCCESS_HOLD_STEPS
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """환경 리셋"""
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

        # 펜 리셋
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
        if self.cfg.pen_tilt_min is not None and self.cfg.pen_tilt_max is not None:
            tilt_min = self.cfg.pen_tilt_min
            tilt_max = self.cfg.pen_tilt_max
        else:
            tilt_min = 0.0
            tilt_max = CURRICULUM_TILT_MAX.get(self.cfg.curriculum_level, 0.0)
        tilt = sample_uniform(tilt_min, tilt_max, (len(env_ids),), device=self.device)
        azimuth = sample_uniform(0, 2 * 3.14159, (len(env_ids),), device=self.device)
        yaw = sample_uniform(
            self.cfg.pen_yaw_range[0], self.cfg.pen_yaw_range[1],
            (len(env_ids),), device=self.device
        )
        pen_quat = self._cone_angle_to_quat(tilt, azimuth, yaw)
        pen_state[:, 3:7] = pen_quat

        self.pen.write_root_pose_to_sim(pen_state[:, :7], env_ids)
        self.pen.write_root_velocity_to_sim(pen_state[:, 7:], env_ids)

        # 상태 리셋
        self.phase[env_ids] = PHASE_APPROACH
        self.phase_step_count[env_ids] = 0
        self.success_hold_count[env_ids] = 0

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
        """원뿔 각도 → 쿼터니언"""
        # Y축 기준 tilt 회전
        q1_w = torch.cos(tilt * 0.5)
        q1_x = torch.zeros_like(tilt)
        q1_y = torch.sin(tilt * 0.5)
        q1_z = torch.zeros_like(tilt)

        # Z축 기준 azimuth 회전
        q2_w = torch.cos(azimuth * 0.5)
        q2_x = torch.zeros_like(azimuth)
        q2_y = torch.zeros_like(azimuth)
        q2_z = torch.sin(azimuth * 0.5)

        # Z축 기준 yaw 회전
        q3_w = torch.cos(yaw * 0.5)
        q3_x = torch.zeros_like(yaw)
        q3_y = torch.zeros_like(yaw)
        q3_z = torch.sin(yaw * 0.5)

        # 쿼터니언 곱셈
        r1_w = q2_w * q1_w - q2_z * q1_y
        r1_x = q2_w * q1_x + q2_z * q1_z
        r1_y = q2_w * q1_y + q2_z * q1_x
        r1_z = q2_z * q1_w + q2_w * q1_z

        w = q3_w * r1_w - q3_z * r1_z
        x = q3_w * r1_x + q3_z * r1_y
        y = q3_w * r1_y - q3_z * r1_x
        z = q3_w * r1_z + q3_z * r1_w

        return torch.stack([w, x, y, z], dim=-1)

    def get_phase_stats(self) -> dict:
        """통계"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
        perpendicular_dist, axis_distance, on_correct_side = self._compute_axis_metrics()

        near_success = (
            (distance_to_cap < SUCCESS_DIST_TO_CAP) &
            (perpendicular_dist < SUCCESS_PERP_DIST) &
            on_correct_side
        ).sum().item()

        return {
            "total_success": self.success_count.sum().item(),
            "near_success": near_success,
            "success_hold_mean": self.success_hold_count.float().mean().item(),
            "on_correct_side": on_correct_side.sum().item(),
            "curriculum_level": self.cfg.curriculum_level,
        }


@configclass
class E0509IKEnvV8Cfg_PLAY(E0509IKEnvV8Cfg):
    """V8 테스트용 설정"""

    def __post_init__(self):
        self.scene.num_envs = 50


# Curriculum Level별 설정
@configclass
class E0509IKEnvV8Cfg_L0(E0509IKEnvV8Cfg):
    """Level 0: 펜 수직"""
    curriculum_level = 0


@configclass
class E0509IKEnvV8Cfg_L1(E0509IKEnvV8Cfg):
    """Level 1: 펜 10° 기울기"""
    curriculum_level = 1


@configclass
class E0509IKEnvV8Cfg_L2(E0509IKEnvV8Cfg):
    """Level 2: 펜 20° 기울기"""
    curriculum_level = 2


@configclass
class E0509IKEnvV8Cfg_L3(E0509IKEnvV8Cfg):
    """Level 3: 펜 30° 기울기"""
    curriculum_level = 3
