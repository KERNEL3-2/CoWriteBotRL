"""
E0509 IK 환경 V3 (펜 축 기준 접근)

=== 목표 ===
펜이 기울어져도 동일하게 작동하는 일반화된 접근 방식
"위에서"가 아닌 "펜 축 방향에서" 접근하도록 유도

=== 핵심 개념 ===
- perpendicular_dist: 펜 축에서 벗어난 거리 (작아야 좋음)
- axis_distance: 펜 축 방향 거리 (양수 = 캡 앞에 있음)

=== 단계 (Phase) ===
1. APPROACH: 펜 축 방향에서 pre-grasp 위치로 접근
2. ALIGN: 위치 유지 + 자세 정렬
3. DESCEND: 정렬 유지하며 캡으로 하강
4. GRASP: 그리퍼 닫기

=== V3 변경사항 ===
- 펜 축 기준 접근 (perpendicular_dist 사용)
- 충돌 페널티 추가
- 단계 체류 페널티 추가
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg

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

# 단계 정의
PHASE_APPROACH = 0    # 펜 축 방향에서 접근
PHASE_ALIGN = 1       # 자세 정렬
PHASE_DESCEND = 2     # 하강
PHASE_GRASP = 3       # 그리퍼 닫기

# Pre-grasp 설정 (펜 축 방향 거리)
PRE_GRASP_AXIS_DIST = 0.07  # 캡에서 펜 축 방향으로 7cm

# 단계 전환 조건
APPROACH_TO_ALIGN_AXIS_DIST = 0.08      # 축 방향 거리 < 8cm
APPROACH_TO_ALIGN_PERP_DIST = 0.03      # 축에서 벗어난 거리 < 3cm
ALIGN_TO_DESCEND_DOT = -0.95            # 정렬 완료 (약 18도 이내)
DESCEND_TO_GRASP_DIST = 0.02            # 캡까지 거리 < 2cm

# 성공 조건
SUCCESS_DIST = 0.015    # 1.5cm
SUCCESS_DOT = -0.95     # dot < -0.95


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class E0509IKEnvV3Cfg(DirectRLEnvCfg):
    """E0509 IK 환경 V3 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 15.0
    action_scale = 0.02
    action_space = 6      # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
    observation_space = 36  # 기존 33 + perpendicular_dist(1) + axis_dist(1) + phase(1)
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

    # 로봇 설정
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=True,  # 충돌 감지 활성화
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
                "joint_2": -0.5,
                "joint_3": 1.0,
                "joint_4": 0.0,
                "joint_5": 1.57,
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

    # 접촉 센서 (그리퍼용)
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link_6",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/Pen"],
    )

    # 펜 캡 위치 범위
    pen_cap_pos_range = {
        "x": (0.3, 0.5),
        "y": (-0.15, 0.15),
        "z": (0.25, 0.40),
    }

    # 펜 방향 랜덤화 (V3: 일단 수직, 나중에 기울기 추가)
    pen_rot_range = {
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    # IK 컨트롤러 설정
    ik_method = "dls"
    ik_lambda = 0.05
    ee_body_name = "link_6"
    ee_offset_pos = [0.0, 0.0, 0.15]

    # ==========================================================================
    # 보상 스케일 (V3: 펜 축 기준)
    # ==========================================================================

    # APPROACH 단계: 펜 축 방향에서 접근
    rew_scale_axis_dist = -5.0           # 축 방향 거리 페널티
    rew_scale_perp_dist = -15.0          # 축에서 벗어난 거리 페널티 (강함!)
    rew_scale_approach_progress = 15.0   # 접근 진행 보상
    rew_scale_on_axis_bonus = 3.0        # 축 위에 있을 때 보너스

    # ALIGN 단계: 자세 정렬
    rew_scale_align_position_hold = -5.0
    rew_scale_align_orientation = 5.0
    rew_scale_exponential_align = 1.5
    exponential_align_threshold = 0.85
    exponential_align_scale = 12.0

    # DESCEND 단계: 하강
    rew_scale_descend_dist = -10.0
    rew_scale_descend_align = 3.0
    rew_scale_descend_progress = 15.0

    # GRASP 단계: 그리퍼 닫기
    rew_scale_grasp_close = 5.0
    rew_scale_grasp_hold = 10.0

    # 공통
    rew_scale_success = 150.0
    rew_scale_phase_transition = 25.0
    rew_scale_action = -0.01

    # 페널티
    rew_scale_collision = -20.0          # 펜 몸체 충돌 페널티
    rew_scale_phase_stall = -0.5         # 단계 체류 페널티 (스텝당)
    phase_stall_threshold = 100          # 이 스텝 이후부터 체류 페널티 적용
    rew_scale_wrong_side = -10.0         # 펜 뒤에서 접근 시 페널티


class E0509IKEnvV3(DirectRLEnv):
    """E0509 IK 환경 V3 (펜 축 기준 접근)"""

    cfg: E0509IKEnvV3Cfg

    def __init__(self, cfg: E0509IKEnvV3Cfg, render_mode: str | None = None, **kwargs):
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

        print(f"[E0509IKEnvV3] EE body: {body_names[0]} (idx={self._ee_body_idx})")
        print(f"[E0509IKEnvV3] Arm joints: {self._arm_joint_names}")

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

        # 상태 머신 (4단계)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 이전 거리
        self.prev_axis_distance = torch.zeros(self.num_envs, device=self.device)
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)

        # ALIGN 단계 목표 위치
        self.align_target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 그리퍼 상태
        self.gripper_closed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 성공 카운터
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _setup_scene(self):
        """씬 구성"""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.pen = RigidObject(self.cfg.pen_cfg)
        self.scene.rigid_objects["pen"] = self.pen

        # 접촉 센서
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """액션 전처리"""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """IK를 사용한 액션 적용"""
        ee_pos_curr, ee_quat_curr = self._compute_ee_pose()

        # GRASP 단계에서는 위치 고정, 그리퍼만 닫기
        if (self.phase == PHASE_GRASP).any():
            grasp_mask = (self.phase == PHASE_GRASP)
            self.actions[grasp_mask, :3] = 0.0  # 위치 변화 없음
            self.actions[grasp_mask, 3:] = 0.0  # 자세 변화 없음

        scaled_actions = self.actions * self.action_scale

        self._ik_controller.set_command(scaled_actions, ee_pos_curr, ee_quat_curr)

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

        # 그리퍼 제어
        gripper_target = torch.zeros(self.num_envs, 4, device=self.device)

        # GRASP 단계에서 그리퍼 닫기
        grasp_mask = (self.phase == PHASE_GRASP)
        if grasp_mask.any():
            gripper_target[grasp_mask] = 0.7  # 닫기
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
        """
        펜 축 기준 메트릭 계산

        Returns:
            perpendicular_dist: 펜 축에서 벗어난 거리
            axis_distance: 펜 축 방향 거리 (양수 = 캡 앞)
            on_correct_side: 캡 앞에 있는지 여부
        """
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        pen_axis = self._get_pen_z_axis()  # 펜 Z축 (캡 방향)

        # 그리퍼 → 캡 벡터
        grasp_to_cap = cap_pos - grasp_pos

        # 펜 축에 투영 (축 방향 거리)
        axis_distance = torch.sum(grasp_to_cap * pen_axis, dim=-1)

        # 펜 축에 수직인 거리 (축에서 벗어난 정도)
        projection = axis_distance.unsqueeze(-1) * pen_axis
        perpendicular_vec = grasp_to_cap - projection
        perpendicular_dist = torch.norm(perpendicular_vec, dim=-1)

        # 캡 앞에 있는지 (axis_distance가 음수면 캡 앞)
        # 왜냐하면 grasp_to_cap = cap - grasp, pen_axis는 캡 방향
        # 그리퍼가 캡 앞에 있으면 grasp_to_cap과 pen_axis가 반대 방향 → 음수
        on_correct_side = axis_distance < 0

        return perpendicular_dist, axis_distance, on_correct_side

    def _get_pregrasp_pos(self) -> torch.Tensor:
        """Pre-grasp 위치 (펜 축 방향으로 7cm)"""
        cap_pos = self._get_pen_cap_pos()
        pen_axis = self._get_pen_z_axis()
        return cap_pos + PRE_GRASP_AXIS_DIST * pen_axis

    def _check_pen_collision(self) -> torch.Tensor:
        """펜 몸체와의 충돌 감지"""
        # 접촉 센서 데이터 확인
        contact_forces = self.contact_sensor.data.net_forces_w
        contact_magnitude = torch.norm(contact_forces, dim=-1).squeeze(-1)

        # 충돌 여부 (force > threshold)
        collision = contact_magnitude > 1.0
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

        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        ee_pos, ee_quat = self._compute_ee_pose()
        ee_axis_angle = math_utils.axis_angle_from_quat(ee_quat)

        # 펜 축 메트릭 (V3 추가)
        perpendicular_dist, axis_distance, _ = self._compute_axis_metrics()

        # 현재 단계 (정규화)
        phase_normalized = self.phase.float() / 3.0

        obs = torch.cat([
            joint_pos,                           # 6
            joint_vel,                           # 6
            grasp_pos_local,                     # 3
            cap_pos_local,                       # 3
            rel_pos,                             # 3
            gripper_z,                           # 3
            pen_z,                               # 3
            ee_pos,                              # 3
            ee_axis_angle,                       # 3
            perpendicular_dist.unsqueeze(-1),   # 1 (V3)
            axis_distance.unsqueeze(-1),         # 1 (V3)
            phase_normalized.unsqueeze(-1),      # 1 (V3)
        ], dim=-1)  # 총 36

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """보상 계산 (V3: 펜 축 기준)"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        pregrasp_pos = self._get_pregrasp_pos()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        perpendicular_dist, axis_distance, on_correct_side = self._compute_axis_metrics()
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
        distance_to_pregrasp = torch.norm(grasp_pos - pregrasp_pos, dim=-1)
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        # 단계별 스텝 카운트 증가
        self.phase_step_count += 1

        # =========================================================
        # APPROACH 단계: 펜 축 방향에서 접근
        # =========================================================
        approach_mask = (self.phase == PHASE_APPROACH)
        if approach_mask.any():
            # 축에서 벗어난 거리 페널티 (강함!)
            rewards[approach_mask] += self.cfg.rew_scale_perp_dist * perpendicular_dist[approach_mask]

            # 축 방향 거리 페널티
            rewards[approach_mask] += self.cfg.rew_scale_axis_dist * torch.abs(axis_distance[approach_mask] + PRE_GRASP_AXIS_DIST)

            # 축 위에 있을 때 보너스
            on_axis = perpendicular_dist[approach_mask] < 0.03
            rewards[approach_mask] += self.cfg.rew_scale_on_axis_bonus * on_axis.float()

            # 접근 진행 보상
            progress = self.prev_axis_distance[approach_mask] - torch.abs(axis_distance[approach_mask] + PRE_GRASP_AXIS_DIST)
            rewards[approach_mask] += self.cfg.rew_scale_approach_progress * torch.clamp(progress, min=0)

            # 잘못된 방향 페널티 (캡 뒤에서 접근)
            wrong_side = ~on_correct_side[approach_mask]
            if wrong_side.any():
                approach_indices = torch.where(approach_mask)[0]
                wrong_indices = approach_indices[wrong_side]
                rewards[wrong_indices] += self.cfg.rew_scale_wrong_side

        # =========================================================
        # ALIGN 단계: 자세 정렬
        # =========================================================
        align_mask = (self.phase == PHASE_ALIGN)
        if align_mask.any():
            # 위치 유지 페널티
            distance_from_target = torch.norm(grasp_pos[align_mask] - self.align_target_pos[align_mask], dim=-1)
            rewards[align_mask] += self.cfg.rew_scale_align_position_hold * distance_from_target

            # 정렬 보상
            align_quality = (-dot_product[align_mask] - 0.5) / 0.5
            rewards[align_mask] += self.cfg.rew_scale_align_orientation * torch.clamp(align_quality, min=0)

            # 지수적 정렬 보너스
            align_value = -dot_product[align_mask]
            exponential_bonus = torch.where(
                align_value > self.cfg.exponential_align_threshold,
                torch.exp((align_value - self.cfg.exponential_align_threshold) * self.cfg.exponential_align_scale),
                torch.ones_like(align_value)
            )
            rewards[align_mask] += self.cfg.rew_scale_exponential_align * exponential_bonus

        # =========================================================
        # DESCEND 단계: 하강
        # =========================================================
        descend_mask = (self.phase == PHASE_DESCEND)
        if descend_mask.any():
            # 캡까지 거리 페널티
            rewards[descend_mask] += self.cfg.rew_scale_descend_dist * distance_to_cap[descend_mask]

            # 정렬 유지 보상
            align_maintain = -dot_product[descend_mask]
            rewards[descend_mask] += self.cfg.rew_scale_descend_align * align_maintain

            # 하강 진행 보상
            descend_progress = self.prev_distance_to_cap[descend_mask] - distance_to_cap[descend_mask]
            rewards[descend_mask] += self.cfg.rew_scale_descend_progress * torch.clamp(descend_progress, min=0)

            # 정렬 풀리면 페널티
            align_lost = dot_product[descend_mask] > -0.9
            if align_lost.any():
                descend_indices = torch.where(descend_mask)[0]
                lost_indices = descend_indices[align_lost]
                rewards[lost_indices] -= 5.0

        # =========================================================
        # GRASP 단계: 그리퍼 닫기
        # =========================================================
        grasp_mask = (self.phase == PHASE_GRASP)
        if grasp_mask.any():
            # 그리퍼 닫기 보상
            gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]
            gripper_closed_amount = gripper_pos.mean(dim=-1)
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_close * gripper_closed_amount[grasp_mask]

            # 위치 유지 보상
            hold_dist = distance_to_cap[grasp_mask]
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_hold * (1.0 - hold_dist)

        # =========================================================
        # 단계 전환
        # =========================================================
        # APPROACH → ALIGN
        transition_to_align = approach_mask & (perpendicular_dist < APPROACH_TO_ALIGN_PERP_DIST) & (torch.abs(axis_distance) < APPROACH_TO_ALIGN_AXIS_DIST)
        if transition_to_align.any():
            self.phase[transition_to_align] = PHASE_ALIGN
            rewards[transition_to_align] += self.cfg.rew_scale_phase_transition
            self.align_target_pos[transition_to_align] = grasp_pos[transition_to_align].clone()
            self.phase_step_count[transition_to_align] = 0

        # ALIGN → DESCEND
        transition_to_descend = align_mask & (dot_product < ALIGN_TO_DESCEND_DOT)
        if transition_to_descend.any():
            self.phase[transition_to_descend] = PHASE_DESCEND
            rewards[transition_to_descend] += self.cfg.rew_scale_phase_transition
            self.prev_distance_to_cap[transition_to_descend] = distance_to_cap[transition_to_descend]
            self.phase_step_count[transition_to_descend] = 0

        # DESCEND → GRASP
        transition_to_grasp = descend_mask & (distance_to_cap < DESCEND_TO_GRASP_DIST) & (dot_product < SUCCESS_DOT)
        if transition_to_grasp.any():
            self.phase[transition_to_grasp] = PHASE_GRASP
            rewards[transition_to_grasp] += self.cfg.rew_scale_phase_transition
            self.phase_step_count[transition_to_grasp] = 0

        # =========================================================
        # 성공 보상
        # =========================================================
        success = (distance_to_cap < SUCCESS_DIST) & (dot_product < SUCCESS_DOT) & self.gripper_closed
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # =========================================================
        # 페널티
        # =========================================================
        # 충돌 페널티
        collision = self._check_pen_collision()
        # GRASP/DESCEND 단계가 아닐 때만 충돌 페널티
        collision_penalty_mask = collision & (self.phase < PHASE_DESCEND)
        if collision_penalty_mask.any():
            rewards[collision_penalty_mask] += self.cfg.rew_scale_collision

        # 단계 체류 페널티 (threshold 스텝 이후부터)
        stall_mask = self.phase_step_count > self.cfg.phase_stall_threshold
        if stall_mask.any():
            stall_penalty = (self.phase_step_count[stall_mask] - self.cfg.phase_stall_threshold).float()
            rewards[stall_mask] += self.cfg.rew_scale_phase_stall * stall_penalty

        # 액션 페널티
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # =========================================================
        # 이전 거리 업데이트
        # =========================================================
        self.prev_axis_distance = torch.abs(axis_distance + PRE_GRASP_AXIS_DIST)
        self.prev_distance_to_cap[descend_mask] = distance_to_cap[descend_mask]

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        distance = torch.norm(grasp_pos - cap_pos, dim=-1)
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)

        success = (distance < SUCCESS_DIST) & (dot_product < SUCCESS_DOT) & self.gripper_closed
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

        roll = sample_uniform(
            self.cfg.pen_rot_range["roll"][0], self.cfg.pen_rot_range["roll"][1],
            (len(env_ids),), device=self.device
        )
        pitch = sample_uniform(
            self.cfg.pen_rot_range["pitch"][0], self.cfg.pen_rot_range["pitch"][1],
            (len(env_ids),), device=self.device
        )
        yaw = sample_uniform(
            self.cfg.pen_rot_range["yaw"][0], self.cfg.pen_rot_range["yaw"][1],
            (len(env_ids),), device=self.device
        )
        pen_quat = self._euler_to_quat(roll, pitch, yaw)
        pen_state[:, 3:7] = pen_quat

        self.pen.write_root_pose_to_sim(pen_state[:, :7], env_ids)
        self.pen.write_root_velocity_to_sim(pen_state[:, 7:], env_ids)

        # 상태 리셋
        self.phase[env_ids] = PHASE_APPROACH
        self.phase_step_count[env_ids] = 0
        self.gripper_closed[env_ids] = False

        self._ik_controller.reset(env_ids_tensor)

        # 초기 거리
        grasp_pos = self._get_grasp_point()
        _, axis_distance, _ = self._compute_axis_metrics()
        cap_pos = self._get_pen_cap_pos()

        self.prev_axis_distance[env_ids] = torch.abs(axis_distance[env_ids] + PRE_GRASP_AXIS_DIST)
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

    def get_phase_stats(self) -> dict:
        """단계별 통계"""
        current_phases = torch.bincount(self.phase, minlength=4)
        return {
            "approach": current_phases[0].item(),
            "align": current_phases[1].item(),
            "descend": current_phases[2].item(),
            "grasp": current_phases[3].item(),
            "total_success": self.success_count.sum().item(),
        }


@configclass
class E0509IKEnvV3Cfg_PLAY(E0509IKEnvV3Cfg):
    """테스트용 설정"""

    def __post_init__(self):
        self.scene.num_envs = 50
