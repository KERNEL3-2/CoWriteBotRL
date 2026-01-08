"""
E0509 OSC 환경 (Operational Space Control)

=== OSC vs IK 차이점 ===
- IK: 관절 위치 타겟 → set_joint_position_target()
- OSC: 관절 토크 타겟 → set_joint_effort_target()

=== OSC 핵심 컨셉 ===
- RL: 위치(x, y, z)만 제어 → 3DoF
- 자세: 펜 축 기반 자동 계산
- 제어: 임피던스 기반 토크 제어 (Sim2Real 친화적)
- GRASP 제거: 실제 로봇에서 별도 처리

=== 성공 조건 (V3 수정) ===
- axis_distance ≈ -7cm (캡 위 7cm 목표, ±2cm 허용)
- 펜 축 정렬 (perp_dist < 1cm)
- 캡 위에 있음 (on_correct_side)
- 10 스텝 유지
- 목표보다 가까이 가면 패널티 (axis_distance 기반)

=== 관절 제한 (DART platform 기준) ===
- J1: ±360°
- J2: ±95°  (자가 충돌 방지)
- J3: ±135°
- J4: ±360°
- J5: ±135°
- J6: ±360°
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

# OSC Controller
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
import isaaclab.utils.math as math_utils


# =============================================================================
# 경로 및 상수
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")
PEN_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "pen.usd")

PEN_LENGTH = 0.1207  # 120.7mm

# 성공 조건 (V2: 7cm로 변경)
SUCCESS_DIST_TO_CAP = 0.07       # 캡까지 거리 < 7cm (기존 3cm)
SUCCESS_PERP_DIST = 0.01         # 펜 축에서 벗어난 거리 < 1cm
SUCCESS_HOLD_STEPS = 10          # 10 스텝 유지하면 성공 (기존 30)


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class E0509OSCEnvCfg(DirectRLEnvCfg):
    """E0509 OSC 환경 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 15.0
    action_scale = 0.05       # OSC는 위치 변화에 더 민감할 수 있음
    action_space = 3          # [Δx, Δy, Δz] 위치만!
    observation_space = 27    # V7과 동일
    state_space = 0

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

    # 로봇 설정 (OSC: stiffness=0, damping=0)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # OSC에서 gravity compensation 사용
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
            # OSC: 토크 직접 제어를 위해 stiffness=0, damping=0
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=3.14,
                stiffness=0.0,    # OSC 핵심!
                damping=0.0,      # OSC 핵심!
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_rh_.*"],
                effort_limit=50.0,
                velocity_limit=1.0,
                stiffness=2000.0,  # 그리퍼는 위치 제어 유지
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

    # 펜 캡 위치 범위 (sim2real pen_workspace.py와 동일)
    pen_cap_pos_range = {
        "x": (0.25, 0.55),     # 25~55cm
        "y": (-0.12, 0.12),    # -12~12cm
        "z": (0.22, 0.35),     # 22~35cm
    }

    # 펜 기울기 제한 (sim2real pen_workspace.py와 동일)
    pen_tilt_max = 0.79  # 45도
    pen_yaw_range = (-3.14, 3.14)

    # 성공 조건
    success_dist_to_cap = SUCCESS_DIST_TO_CAP   # 캡까지 거리 < 7cm
    success_perp_dist = SUCCESS_PERP_DIST       # 펜 축에서 벗어난 거리 < 1cm
    success_hold_steps = SUCCESS_HOLD_STEPS     # 10 스텝 유지

    # OSC 설정
    osc_motion_stiffness = 150.0      # 위치 강성
    osc_motion_damping_ratio = 1.0    # 임계 감쇠
    osc_inertial_decoupling = True    # 관성 디커플링

    # EE 설정
    ee_body_name = "link_6"
    ee_offset_pos = [0.0, 0.0, 0.15]

    # ==========================================================================
    # 보상 스케일 (V3: axis_distance 기반 목표 거리)
    # ==========================================================================
    # 목표: 펜 축 위에서 캡으로부터 7cm 거리 유지
    target_axis_distance = -0.07  # 캡 위 7cm (음수 = 캡 위)

    # 축 방향 목표 거리 보상
    rew_scale_target_dist = 15.0       # 목표 거리(7cm)에 가까울수록 보상
    rew_scale_target_dist_penalty = -10.0  # 목표에서 벗어나면 패널티

    # 펜 축 정렬 보상 (perp_dist)
    rew_scale_perp_dist = -15.0        # 축에서 벗어나면 페널티 (강화: -8 → -15)
    rew_scale_perp_exp = 8.0           # 축 위에 있으면 보상 (강화: 5 → 8)

    # 자세 정렬 (V3.1: 강화)
    rew_scale_alignment = 20.0         # 강화: 5 → 20 (dot 정렬 중요)

    # 성공/기타
    rew_scale_ready_bonus = 10.0
    rew_scale_success = 100.0
    rew_scale_action = -0.01
    rew_scale_collision = -50.0
    rew_scale_too_close = -15.0        # 목표보다 가까이 가면 패널티 (강화: -10 → -15)


class E0509OSCEnv(DirectRLEnv):
    """E0509 OSC 환경 (Operational Space Control)"""

    cfg: E0509OSCEnvCfg

    def __init__(self, cfg: E0509OSCEnvCfg, render_mode: str | None = None, **kwargs):
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
        else:
            self._jacobi_body_idx = self._ee_body_idx

        print(f"[E0509OSCEnv] EE body: {body_names[0]} (idx={self._ee_body_idx})")
        print(f"[E0509OSCEnv] Arm joints: {self._arm_joint_names}")
        print(f"[E0509OSCEnv] Using Operational Space Control (OSC)")
        print(f"[E0509OSCEnv] V2: SUCCESS_DIST_TO_CAP = {SUCCESS_DIST_TO_CAP*100:.0f}cm, too_close penalty enabled")

        # OSC Controller
        self._osc_controller = OperationalSpaceController(
            cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],  # 상대 위치/자세 제어
                impedance_mode="fixed",
                inertial_dynamics_decoupling=self.cfg.osc_inertial_decoupling,
                gravity_compensation=True,  # 중력 보상 활성화
                motion_stiffness_task=self.cfg.osc_motion_stiffness,
                motion_damping_ratio_task=self.cfg.osc_motion_damping_ratio,
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

        # 상태 변수
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)
        self.success_hold_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.collision_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 디버그 출력
        self._global_step = 0
        self._phase_print_interval = 5000

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
        """
        pen_z = self._get_pen_z_axis()

        # 그리퍼 Z축 = -펜 Z축 (펜을 위에서 잡음)
        gripper_z = -pen_z

        # 그리퍼 X축: 펜 축과 월드 Z축의 외적
        world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        gripper_x = torch.cross(world_z, gripper_z, dim=-1)
        gripper_x_norm = torch.norm(gripper_x, dim=-1, keepdim=True)

        # 펜이 거의 수직일 때 → 월드 X축 사용
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

        # 회전 행렬 → 쿼터니언
        rot_matrix = torch.stack([gripper_x, gripper_y, gripper_z], dim=-1)
        target_quat = math_utils.quat_from_matrix(rot_matrix)

        return target_quat

    def _apply_action(self) -> None:
        """
        액션 적용 (OSC: 토크 제어)

        RL 액션: [Δx, Δy, Δz] (3차원)
        자세: 펜 축 기반 자동 계산
        출력: 관절 토크
        """
        # 현재 EE 상태
        ee_pos_b, ee_quat_b = self._compute_ee_pose()
        ee_vel_b = self._compute_ee_velocity()

        # RL 액션: 위치 변화
        pos_delta = self.actions * self.action_scale

        # 목표 자세: 펜 축 기반 자동 계산
        target_quat = self._compute_auto_orientation()

        # 현재 자세 → 목표 자세의 상대 회전
        quat_curr_inv = math_utils.quat_inv(ee_quat_b)
        quat_delta = math_utils.quat_mul(target_quat, quat_curr_inv)
        rot_delta_axis_angle = math_utils.axis_angle_from_quat(quat_delta)

        # 자세 변화 스케일링
        rot_scale = 0.3
        rot_delta_scaled = rot_delta_axis_angle * rot_scale

        # 6DoF 명령: [Δx, Δy, Δz, Δrx, Δry, Δrz]
        osc_command = torch.cat([pos_delta, rot_delta_scaled], dim=-1)

        # EE pose (7DoF: pos + quat)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        # OSC 명령 설정
        self._osc_controller.set_command(
            command=osc_command,
            current_ee_pose_b=ee_pose_b,
        )

        # 동역학 데이터 가져오기
        jacobian_b = self._compute_ee_jacobian()
        mass_matrix = self._get_mass_matrix()
        gravity = self._get_gravity_compensation()
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]

        # OSC 계산 → 관절 토크
        joint_efforts = self._osc_controller.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
        )

        # 그리퍼는 위치 제어 (항상 열린 상태)
        gripper_target = torch.zeros(self.num_envs, 4, device=self.device)
        self.robot.set_joint_position_target(gripper_target, joint_ids=self._gripper_joint_ids)

        # 팔은 토크 제어
        self.robot.set_joint_effort_target(joint_efforts, joint_ids=self._arm_joint_ids)

    def _compute_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """End-effector pose 계산 (base frame)"""
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

    def _compute_ee_velocity(self) -> torch.Tensor:
        """End-effector velocity 계산 (base frame)"""
        ee_vel_w = self.robot.data.body_vel_w[:, self._ee_body_idx, :]
        root_vel_w = self.robot.data.root_vel_w

        # 상대 속도 (world frame)
        relative_vel_w = ee_vel_w - root_vel_w

        # World → Base frame 변환
        ee_lin_vel_b = math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w, relative_vel_w[:, 0:3]
        )
        ee_ang_vel_b = math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w, relative_vel_w[:, 3:6]
        )

        return torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    def _compute_ee_jacobian(self) -> torch.Tensor:
        """End-effector Jacobian 계산 (base frame)"""
        jacobian_w = self.robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._arm_joint_ids
        ]

        base_rot = self.robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))

        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # EE offset 보정
        jacobian_b[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._ee_offset_pos),
            jacobian_b[:, 3:, :]
        )

        return jacobian_b

    def _get_mass_matrix(self) -> torch.Tensor:
        """관절 공간 질량 행렬"""
        mass_matrix_full = self.robot.root_physx_view.get_generalized_mass_matrices()
        # 팔 관절만 추출
        mass_matrix = mass_matrix_full[:, :6, :][:, :, :6]
        return mass_matrix

    def _get_gravity_compensation(self) -> torch.Tensor:
        """중력 보상 토크"""
        gravity_full = self.robot.root_physx_view.get_gravity_compensation_forces()
        gravity = gravity_full[:, self._arm_joint_ids]
        return gravity

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
        """펜 몸체와의 충돌 감지 (캡 영역 제외)

        펜 구조 (pen_axis 방향이 캡):
            캡 (cap) ← proj > 0
              ↑
        ======●====== pen_pos (중심)
              ↓
            몸체 ← proj < 0 (여기만 충돌 감지!)
        """
        grasp_pos = self._get_grasp_point()
        pen_pos = self.pen.data.root_pos_w
        pen_axis = self._get_pen_z_axis()

        grasp_to_pen = pen_pos - grasp_pos
        proj_length = torch.sum(grasp_to_pen * pen_axis, dim=-1)

        proj_vec = proj_length.unsqueeze(-1) * pen_axis
        perp_to_pen = grasp_to_pen - proj_vec
        perp_dist = torch.norm(perp_to_pen, dim=-1)

        # 펜 몸체만 충돌 감지 (캡 영역 제외)
        # proj_length < 0: 그리퍼가 캡 위에 있음 (정상 접근, 충돌 감지 X)
        # proj_length > 0: 그리퍼가 몸체 쪽에 있음 (충돌 감지 O)
        in_pen_body = (proj_length > 0) & (proj_length < PEN_LENGTH / 2)
        collision = in_pen_body & (perp_dist < 0.015)  # 1.5cm

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

        # Phase는 항상 0 (APPROACH만)
        phase_normalized = torch.zeros(self.num_envs, device=self.device)

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
        """보상 계산 (V3: axis_distance 기반 목표 거리)"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()

        perpendicular_dist, axis_distance, on_correct_side = self._compute_axis_metrics()
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        # 자세 정렬 계산
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()
        dot = torch.sum(gripper_z * pen_z, dim=-1)

        # ==========================================================================
        # V3: 축 방향 목표 거리 보상 (axis_distance 기반)
        # ==========================================================================
        # 목표: axis_distance ≈ -7cm (캡 위 7cm)
        target_axis_dist = self.cfg.target_axis_distance  # -0.07
        axis_dist_error = torch.abs(axis_distance - target_axis_dist)

        # 목표 거리에 가까울수록 보상 (지수적)
        rewards += self.cfg.rew_scale_target_dist * torch.exp(-axis_dist_error * 30.0)

        # 목표에서 벗어나면 선형 패널티
        rewards += self.cfg.rew_scale_target_dist_penalty * axis_dist_error

        # ==========================================================================
        # 펜 축 정렬 보상 (perp_dist) - 강화됨
        # ==========================================================================
        rewards += self.cfg.rew_scale_perp_dist * perpendicular_dist
        rewards += self.cfg.rew_scale_perp_exp * torch.exp(-perpendicular_dist * 50.0)

        # 자세 정렬 보상
        alignment_reward = (-dot - 0.5) * 0.5
        alignment_reward = torch.clamp(alignment_reward, min=0)
        rewards += self.cfg.rew_scale_alignment * alignment_reward

        # 캡 위에 있을 때 보너스
        above_cap_bonus = on_correct_side.float() * 0.5
        rewards += above_cap_bonus

        # ==========================================================================
        # 성공 조건: 목표 위치 근처에서 유지
        # ==========================================================================
        # 성공 조건: axis_distance가 목표 근처 + perp_dist 낮음 + 캡 위
        near_target = torch.abs(axis_distance - target_axis_dist) < 0.02  # 목표 ±2cm
        success_condition = (
            near_target &
            (perpendicular_dist < SUCCESS_PERP_DIST) &
            on_correct_side
        )

        # 성공 조건 근접 보너스
        near_success = (
            (torch.abs(axis_distance - target_axis_dist) < 0.04) &  # 목표 ±4cm
            (perpendicular_dist < SUCCESS_PERP_DIST * 2) &
            on_correct_side
        )
        rewards[near_success] += self.cfg.rew_scale_ready_bonus * 0.5

        self.success_hold_count[success_condition] += 1
        self.success_hold_count[~success_condition] = 0

        success = self.success_hold_count >= SUCCESS_HOLD_STEPS
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # ==========================================================================
        # V3: 목표보다 가까이 가면 패널티 (axis_distance 기반)
        # ==========================================================================
        # axis_distance > target (= 캡에 더 가까움) 이면 패널티
        too_close = (axis_distance > target_axis_dist) & on_correct_side
        too_close_amount = torch.clamp(axis_distance - target_axis_dist, min=0)
        rewards += self.cfg.rew_scale_too_close * too_close_amount * too_close.float() * 10.0

        # 페널티: 충돌 감지 및 저장 (에피소드 종료용)
        collision = self._check_pen_collision()
        self.collision_detected = collision  # 종료 조건에서 사용
        if collision.any():
            rewards[collision] += self.cfg.rew_scale_collision

        passed_cap = ~on_correct_side
        rewards[passed_cap] -= 1.0

        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # 이전 거리 업데이트
        self.prev_distance_to_cap = distance_to_cap.clone()

        # TensorBoard 로깅
        self._global_step += 1

        if "log" not in self.extras:
            self.extras["log"] = {}

        self.extras["log"]["OSC/success_hold_count_mean"] = self.success_hold_count.float().mean().item()
        self.extras["log"]["OSC/total_success"] = float(self.success_count.sum().item())
        self.extras["log"]["OSC/collision_count"] = float(collision.sum().item())
        self.extras["log"]["OSC/too_close_count"] = float(too_close.sum().item())
        self.extras["log"]["Metrics/dist_to_cap_mean"] = distance_to_cap.mean().item()
        self.extras["log"]["Metrics/perp_dist_mean"] = perpendicular_dist.mean().item()
        self.extras["log"]["Metrics/axis_dist_mean"] = axis_distance.mean().item()  # V3: 축 방향 거리
        self.extras["log"]["Metrics/axis_dist_error"] = axis_dist_error.mean().item()  # V3: 목표 오차
        self.extras["log"]["Metrics/dot_mean"] = dot.mean().item()

        # 콘솔 출력
        if self._global_step % self._phase_print_interval == 0:
            on_correct_pct = on_correct_side.float().mean().item() * 100
            collision_cnt = collision.sum().item()
            too_close_cnt = too_close.sum().item()
            axis_dist_mean = axis_distance.mean().item() * 100
            axis_err_mean = axis_dist_error.mean().item() * 100
            print(f"  [Step {self._global_step}] OSC V3 (목표: 캡 위 7cm)", flush=True)
            print(f"    → axis_dist={axis_dist_mean:.2f}cm (목표: -7cm), "
                  f"오차={axis_err_mean:.2f}cm, "
                  f"perp={perpendicular_dist.mean().item()*100:.2f}cm", flush=True)
            print(f"    → dot={dot.mean().item():.3f}, "
                  f"on_correct={on_correct_pct:.0f}%, "
                  f"success={self.success_count.sum().item()}, "
                  f"collision={collision_cnt}", flush=True)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        success = self.success_hold_count >= SUCCESS_HOLD_STEPS

        # 충돌 시 에피소드 즉시 종료 (학습 효율 향상)
        terminated = success | self.collision_detected

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

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

        # 펜 리셋 (특이점 회피를 위한 위치/각도 설정)
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

        # 펜 기울기 (특이점 회피를 위해 제한)
        tilt = sample_uniform(0.0, self.cfg.pen_tilt_max, (len(env_ids),), device=self.device)
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
        self.success_hold_count[env_ids] = 0
        self.collision_detected[env_ids] = False

        self._osc_controller.reset()

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

        # 쿼터니언 곱셈: q2 * q1
        r1_w = q2_w * q1_w - q2_z * q1_y
        r1_x = q2_w * q1_x + q2_z * q1_z
        r1_y = q2_w * q1_y + q2_z * q1_x
        r1_z = q2_z * q1_w + q2_w * q1_z

        # q3 * r1
        w = q3_w * r1_w - q3_z * r1_z
        x = q3_w * r1_x + q3_z * r1_y
        y = q3_w * r1_y - q3_z * r1_x
        z = q3_w * r1_z + q3_z * r1_w

        return torch.stack([w, x, y, z], dim=-1)


# =============================================================================
# 설정 변형
# =============================================================================
@configclass
class E0509OSCEnvCfg_PLAY(E0509OSCEnvCfg):
    """테스트용 설정"""

    def __post_init__(self):
        self.scene.num_envs = 50


@configclass
class E0509OSCEnvCfg_Soft(E0509OSCEnvCfg):
    """부드러운 동작을 위한 낮은 stiffness 설정

    Sim2Real 전이 시 실제 로봇의 임피던스 특성에 맞춤
    - stiffness: 150 → 60 (더 부드러운 반응)
    - action_scale: 0.05 → 0.03 (더 작은 이동량)
    """

    # OSC 설정 (부드러운 동작)
    osc_motion_stiffness = 60.0       # 150 → 60 (부드럽게)
    osc_motion_damping_ratio = 1.0    # 임계 감쇠 유지
    action_scale = 0.03               # 0.05 → 0.03 (더 작은 스텝)


@configclass
class E0509OSCEnvCfg_Soft_PLAY(E0509OSCEnvCfg_Soft):
    """Soft 버전 테스트용"""

    def __post_init__(self):
        self.scene.num_envs = 50
