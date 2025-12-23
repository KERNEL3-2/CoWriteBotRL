"""
E0509 IK 환경 V5.4 (Hybrid RL + TCP + LIFT + Good Grasp)

=== V5.4 변경사항 (V5.3 대비) ===
1. PHASE_LIFT 추가 (6단계): 펜 들어올리기 + 성공 종료
   - GRASP에서 Good Grasp + 30스텝 유지 후 LIFT로 전환
   - LIFT: 월드 Z축 방향으로 5cm 상승
   - LIFT 완료 + Good Grasp 유지 시 성공

2. Good Grasp 조건 (펜이 그리퍼에 제대로 잡힌 상태):
   - perp_dist < 8mm (펜 축과 grasp point 일치)
   - dot < -0.98 (자세 정렬)
   - gripper_amount: 0.8 ~ 1.05 (펜에 걸린 상태, 완전히 닫힌 게 아님)

3. 그리퍼 닫기 강화: gripper_target 0.7 → 1.0

=== Curriculum Levels ===
Level 0: 펜 수직 (tilt_max = 0°)     - 기본 동작 학습
Level 1: 펜 10° 기울기               - 약간의 변형 적응
Level 2: 펜 20° 기울기               - 중간 난이도
Level 3: 펜 30° 기울기               - 최종 목표

=== 단계 (Phase) - 6단계 ===
0. APPROACH: 펜 축 방향에서 접근 + 자세 정렬 시작 (RL)
1. ALIGN: 위치 미세조정 + 자세 정렬 완료 (RL)
2. FINE_ALIGN: 정밀 자세 정렬 (TCP)
3. DESCEND: 펜 캡 방향 하강 + 자세 유지 (TCP)
4. GRASP: 그리퍼 닫기 (Good Grasp + 30스텝) (TCP)
5. LIFT: 펜 들어올리기 5cm → 성공 종료 (TCP)
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

# 단계 정의 (V5.4: LIFT 추가)
PHASE_APPROACH = 0    # RL: 펜 축 방향에서 접근
PHASE_ALIGN = 1       # RL: 대략적 자세 정렬
PHASE_FINE_ALIGN = 2  # TCP: 정밀 자세 정렬 (dot → -1)
PHASE_DESCEND = 3     # TCP: 수직 하강
PHASE_GRASP = 4       # TCP: 그리퍼 닫기
PHASE_LIFT = 5        # TCP: 펜 들어올리기 + Home 복귀

# Pre-grasp 설정
PRE_GRASP_AXIS_DIST = 0.07  # 캡에서 펜 축 방향으로 7cm

# 단계 전환 조건 (V5.2: 위치 + 자세 동시 체크)
APPROACH_TO_ALIGN_AXIS_DIST = 0.10      # 축 방향 거리 < 10cm
APPROACH_TO_ALIGN_PERP_DIST = 0.05      # 축에서 벗어난 거리 < 5cm
ALIGN_TO_FINE_ALIGN_DOT = -0.85         # 대략적 정렬
ALIGN_TO_FINE_ALIGN_PERP_DIST = 0.04    # V5.2: 축 거리 < 4cm 추가
FINE_ALIGN_TO_DESCEND_DOT = -0.98       # 정밀 정렬 완료
FINE_ALIGN_TO_DESCEND_PERP_DIST = 0.03  # V5.2: 축 거리 < 3cm 추가
DESCEND_TO_GRASP_DIST = 0.02            # 캡까지 거리 < 2cm

# 성공 조건 (V5.4: 강화)
SUCCESS_DIST = 0.005    # 5mm (거의 일치)
SUCCESS_DOT = -0.99     # dot < -0.99 (거의 -1)

# Fine Align TCP 제어 설정
FINE_ALIGN_ROTATION_SPEED = 0.03  # rad/step
DESCEND_SPEED = 0.002             # m/step
DESCEND_POSE_CORRECTION_GAIN = 1.5  # 자세 보정 강화

# GRASP/LIFT 설정 (V5.4)
GRIPPER_CLOSE_TARGET = 1.1        # 그리퍼 닫기 목표 (확실히 닫으려고 시도)
LIFT_HEIGHT = 0.05                # 들어올리기 높이 5cm
LIFT_SPEED = 0.003                # 들어올리기 속도

# Good Grasp 조건 (V5.4)
# 펜 두께 16-17mm → 그리퍼가 완전히 안 닫히고 펜에 걸린 상태
# - 닫기 명령 1.1 보내도 펜이 있으면 ~0.9에서 멈춤
# - 펜이 없으면 1.1까지 완전히 닫힘 → good_grasp 실패
GOOD_GRASP_GRIPPER_MIN = 0.8      # 그리퍼가 어느정도 닫힘
GOOD_GRASP_GRIPPER_MAX = 1.05     # 완전히 닫힌 건 아님 (펜이 있음)
GOOD_GRASP_PERP_DIST = 0.008      # 펜 축과 grasp point 거의 일치 (8mm)
GOOD_GRASP_DOT = -0.98            # 자세 정렬

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
class E0509IKEnvV5Cfg(DirectRLEnvCfg):
    """E0509 IK 환경 V5 설정 (Hybrid RL + TCP + Curriculum)"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 15.0
    action_scale = 0.02
    action_space = 6      # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
    observation_space = 37  # 기존 36 + fine_align_active(1)
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
    # 보상 스케일 (V5.6 - 점진적 증가)
    # ==========================================================================
    # 목표: 다음 phase로 갈수록 보상 증가 → 방향성 명확
    # APPROACH(3-4) < ALIGN(4-5) < FINE_ALIGN(5-6) < DESCEND(6-7) < GRASP(8-10)

    # APPROACH 단계 (목표 ~3-4/step)
    rew_scale_axis_dist = -2.0
    rew_scale_perp_dist = -5.0
    rew_scale_approach_progress = 3.0   # V5.6: 10.0 → 3.0
    rew_scale_on_axis_bonus = 1.5       # V5.6: 2.0 → 1.5

    # ALIGN 단계 (목표 ~4-5/step)
    rew_scale_align_position_hold = -3.0
    rew_scale_align_orientation = 2.0   # V5.6: 3.0 → 2.0
    rew_scale_exponential_align = 0.3   # V5.6: 0.5 → 0.3
    exponential_align_threshold = 0.80
    exponential_align_scale = 10.0

    # FINE_ALIGN (목표 ~5-6/step)
    rew_scale_fine_align = 6.0     # V5.6: 2.0 → 6.0

    # DESCEND (목표 ~6-7/step)
    rew_scale_descend = 7.0        # V5.6: 3.0 → 7.0

    # GRASP (목표 ~8-10/step)
    rew_scale_grasp_close = 4.0    # V5.6: 5.0 → 4.0
    rew_scale_grasp_hold = 6.0     # V5.6: 10.0 → 6.0 (합계 ~10)

    # 공통
    rew_scale_success = 100.0
    rew_scale_phase_transition = 50.0   # V5.6: 15.0 → 50.0 (크게 높임)
    rew_scale_action = -0.005

    # 페널티
    rew_scale_collision = -5.0
    rew_scale_wrong_side = 0.0
    rew_scale_phase_stagnation = -0.01  # V5.6: 체류 패널티 추가


class E0509IKEnvV5(DirectRLEnv):
    """E0509 IK 환경 V5 (Hybrid RL + TCP + Curriculum Learning)"""

    cfg: E0509IKEnvV5Cfg

    def __init__(self, cfg: E0509IKEnvV5Cfg, render_mode: str | None = None, **kwargs):
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

        print(f"[E0509IKEnvV4] EE body: {body_names[0]} (idx={self._ee_body_idx})")
        print(f"[E0509IKEnvV4] Arm joints: {self._arm_joint_names}")

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

        # 상태 머신 (5단계)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 이전 거리
        self.prev_axis_distance = torch.zeros(self.num_envs, device=self.device)
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)

        # ALIGN 단계 목표 위치
        self.align_target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # FINE_ALIGN 목표 위치 (고정)
        self.fine_align_target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 그리퍼 상태
        self.gripper_closed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # LIFT 단계 상태 (V5.4)
        self.lift_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.lift_complete = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.grasp_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 성공 카운터
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Phase 출력용 global step 카운터
        self._global_step = 0
        self._phase_print_interval = 5000  # 5000 step마다 출력 (약 50 iter)

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

    def _apply_action(self) -> None:
        """액션 적용 (Hybrid: RL + TCP 제어)"""
        ee_pos_curr, ee_quat_curr = self._compute_ee_pose()

        # RL 단계 (APPROACH, ALIGN)
        rl_mask = (self.phase == PHASE_APPROACH) | (self.phase == PHASE_ALIGN)

        # TCP 제어 단계 (FINE_ALIGN, DESCEND, GRASP)
        tcp_mask = ~rl_mask

        # ============================================================
        # RL 단계: 학습된 정책 사용
        # ============================================================
        scaled_actions = self.actions * self.action_scale

        # TCP 단계에서는 deterministic 액션 사용
        if tcp_mask.any():
            tcp_actions = self._compute_tcp_actions(ee_pos_curr, ee_quat_curr)
            scaled_actions[tcp_mask] = tcp_actions[tcp_mask]

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

        # GRASP/LIFT 단계에서 그리퍼 닫기 (V5.4: 1.0으로 강화)
        grasp_or_lift_mask = (self.phase == PHASE_GRASP) | (self.phase == PHASE_LIFT)
        if grasp_or_lift_mask.any():
            gripper_target[grasp_or_lift_mask] = GRIPPER_CLOSE_TARGET  # V5.4: 0.7 → 1.0
            self.gripper_closed[grasp_or_lift_mask] = True

        full_target = torch.zeros(self.num_envs, 10, device=self.device)
        full_target[:, :6] = joint_pos_target
        full_target[:, 6:] = gripper_target

        self.robot.set_joint_position_target(full_target)

    def _compute_tcp_actions(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
        """TCP 제어 액션 계산 (FINE_ALIGN, DESCEND, GRASP)"""
        tcp_actions = torch.zeros(self.num_envs, 6, device=self.device)

        pen_z = self._get_pen_z_axis()
        gripper_z = self._get_gripper_z_axis()
        cap_pos = self._get_pen_cap_pos()
        grasp_pos = self._get_grasp_point()

        # ============================================================
        # FINE_ALIGN: 그리퍼 z축을 펜 z축과 정렬 (dot → -1)
        # ============================================================
        fine_align_mask = (self.phase == PHASE_FINE_ALIGN)
        if fine_align_mask.any():
            # 목표: gripper_z와 pen_z가 반대 방향 (dot = -1)
            # 회전축: gripper_z × (-pen_z)
            target_z = -pen_z[fine_align_mask]
            current_z = gripper_z[fine_align_mask]

            # 회전축 계산 (cross product)
            rotation_axis = torch.cross(current_z, target_z, dim=-1)
            rotation_axis_norm = torch.norm(rotation_axis, dim=-1, keepdim=True) + 1e-6
            rotation_axis = rotation_axis / rotation_axis_norm

            # 회전 각도 (dot product로부터)
            dot = torch.sum(current_z * target_z, dim=-1, keepdim=True)
            angle = torch.acos(torch.clamp(dot, -1.0, 1.0))

            # 회전 속도 제한
            rotation_delta = torch.clamp(angle, max=FINE_ALIGN_ROTATION_SPEED)
            rotation_vec = rotation_axis * rotation_delta

            # 위치 고정 (fine_align_target_pos 유지)
            pos_error = self.fine_align_target_pos[fine_align_mask] - grasp_pos[fine_align_mask]
            tcp_actions[fine_align_mask, :3] = pos_error * 0.5  # 위치 보정

            # 회전만 적용
            tcp_actions[fine_align_mask, 3:] = rotation_vec.squeeze(-1) if rotation_vec.dim() > 2 else rotation_vec

        # ============================================================
        # DESCEND: 펜 캡 방향으로 이동 + Z축 정렬 유지 (V5.1 수정)
        # ============================================================
        descend_mask = (self.phase == PHASE_DESCEND)
        if descend_mask.any():
            # V5.1 수정: 펜 캡 방향으로 이동 (월드 Z축이 아닌!)
            # 문제: 그리퍼가 펜 캡 "위"가 아닌 "옆"에 있을 때 월드 Z축으로 내려가면 펜에 가까워지지 않음
            # 해결: 항상 펜 캡 방향으로 이동
            to_cap = cap_pos[descend_mask] - grasp_pos[descend_mask]
            to_cap_dist = torch.norm(to_cap, dim=-1, keepdim=True) + 1e-6
            to_cap_normalized = to_cap / to_cap_dist
            tcp_actions[descend_mask, :3] = to_cap_normalized * DESCEND_SPEED

            # Z축 정렬 유지 (자세 보정)
            target_z = -pen_z[descend_mask]
            current_z = gripper_z[descend_mask]
            rotation_axis = torch.cross(current_z, target_z, dim=-1)
            rotation_axis_norm = torch.norm(rotation_axis, dim=-1, keepdim=True) + 1e-6
            rotation_axis = rotation_axis / rotation_axis_norm
            dot = torch.sum(current_z * target_z, dim=-1, keepdim=True)
            angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
            # 자세 보정 속도 (DESCEND_POSE_CORRECTION_GAIN)
            rotation_delta = torch.clamp(angle, max=FINE_ALIGN_ROTATION_SPEED * DESCEND_POSE_CORRECTION_GAIN)
            tcp_actions[descend_mask, 3:] = (rotation_axis * rotation_delta).squeeze(-1) if (rotation_axis * rotation_delta).dim() > 2 else rotation_axis * rotation_delta

        # ============================================================
        # GRASP: 위치 고정
        # ============================================================
        grasp_mask = (self.phase == PHASE_GRASP)
        if grasp_mask.any():
            tcp_actions[grasp_mask] = 0.0  # 위치/자세 고정
            self.grasp_step_count[grasp_mask] += 1  # V5.4: GRASP 스텝 카운트

        # ============================================================
        # LIFT: 펜 들어올리기 (V5.4)
        # ============================================================
        lift_mask = (self.phase == PHASE_LIFT)
        if lift_mask.any():
            # 월드 Z축 방향으로 상승
            lift_direction = torch.zeros(lift_mask.sum(), 3, device=self.device)
            lift_direction[:, 2] = 1.0  # 월드 Z축 상승

            # 현재 높이 계산
            current_pos = grasp_pos[lift_mask]
            start_pos = self.lift_start_pos[lift_mask]
            current_lift_height = current_pos[:, 2] - start_pos[:, 2]

            # 목표 높이까지 상승
            remaining_height = LIFT_HEIGHT - current_lift_height
            still_lifting = remaining_height > 0.001  # 1mm 이상 남음

            # 상승 명령
            tcp_actions[lift_mask, :3] = lift_direction * LIFT_SPEED

            # LIFT 완료 체크
            lift_complete_now = ~still_lifting
            if lift_complete_now.any():
                lift_indices = torch.where(lift_mask)[0]
                complete_indices = lift_indices[lift_complete_now]
                self.lift_complete[complete_indices] = True
                tcp_actions[complete_indices] = 0.0  # 완료 후 정지

            # 자세 유지 (펜 축 정렬)
            target_z = -pen_z[lift_mask]
            current_z = gripper_z[lift_mask]
            rotation_axis = torch.cross(current_z, target_z, dim=-1)
            rotation_axis_norm = torch.norm(rotation_axis, dim=-1, keepdim=True) + 1e-6
            rotation_axis = rotation_axis / rotation_axis_norm
            dot = torch.sum(current_z * target_z, dim=-1, keepdim=True)
            angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
            rotation_delta = torch.clamp(angle, max=FINE_ALIGN_ROTATION_SPEED)
            tcp_actions[lift_mask, 3:] = (rotation_axis * rotation_delta).squeeze(-1) if (rotation_axis * rotation_delta).dim() > 2 else rotation_axis * rotation_delta

        return tcp_actions

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

    def _get_pregrasp_pos(self) -> torch.Tensor:
        """Pre-grasp 위치"""
        cap_pos = self._get_pen_cap_pos()
        pen_axis = self._get_pen_z_axis()
        return cap_pos + PRE_GRASP_AXIS_DIST * pen_axis

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

        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        ee_pos, ee_quat = self._compute_ee_pose()
        ee_axis_angle = math_utils.axis_angle_from_quat(ee_quat)

        perpendicular_dist, axis_distance, _ = self._compute_axis_metrics()

        # 현재 단계 (정규화)
        phase_normalized = self.phase.float() / 5.0  # V5.4: 6단계 (0~5)

        # TCP 제어 활성화 여부
        tcp_active = ((self.phase >= PHASE_FINE_ALIGN) & (self.phase <= PHASE_LIFT)).float()  # V5.4: LIFT 포함

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
            perpendicular_dist.unsqueeze(-1),   # 1
            axis_distance.unsqueeze(-1),         # 1
            phase_normalized.unsqueeze(-1),      # 1
            tcp_active.unsqueeze(-1),            # 1 (V4 추가)
        ], dim=-1)  # 총 37

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """보상 계산"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        pregrasp_pos = self._get_pregrasp_pos()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        perpendicular_dist, axis_distance, on_correct_side = self._compute_axis_metrics()
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        self.phase_step_count += 1

        # =========================================================
        # APPROACH 단계 (RL) - V5.3: 위치 + 자세 동시 reward
        # =========================================================
        approach_mask = (self.phase == PHASE_APPROACH)
        if approach_mask.any():
            # 위치 reward (기존)
            rewards[approach_mask] += self.cfg.rew_scale_perp_dist * perpendicular_dist[approach_mask]
            rewards[approach_mask] += self.cfg.rew_scale_axis_dist * torch.abs(axis_distance[approach_mask] + PRE_GRASP_AXIS_DIST)

            on_axis = perpendicular_dist[approach_mask] < 0.05
            rewards[approach_mask] += self.cfg.rew_scale_on_axis_bonus * on_axis.float()

            progress = self.prev_axis_distance[approach_mask] - torch.abs(axis_distance[approach_mask] + PRE_GRASP_AXIS_DIST)
            rewards[approach_mask] += self.cfg.rew_scale_approach_progress * torch.clamp(progress, min=0)

            # V5.6 수정: 자세 정렬 reward (위치 조건부 + exponential + 낮은 가중치)
            # 거리가 가까울수록 자세 reward 가중치가 exponential하게 증가
            approach_align_quality = (-dot_product[approach_mask] - 0.5) / 0.5

            # 기준 거리 10cm, 가까울수록 가중치 증가 (max 2.0으로 제한)
            base_dist = 0.1
            scale = 20.0
            position_weight = torch.exp((base_dist - perpendicular_dist[approach_mask]) * scale)
            position_weight = torch.clamp(position_weight, min=0, max=2.0)  # V5.6: 5.0 → 2.0

            rewards[approach_mask] += self.cfg.rew_scale_align_orientation * 0.5 * torch.clamp(approach_align_quality, min=0) * position_weight

            # V5.6: Phase 체류 패널티 (오래 머물수록 페널티 증가)
            rewards[approach_mask] += self.cfg.rew_scale_phase_stagnation * self.phase_step_count[approach_mask].float()

        # =========================================================
        # ALIGN 단계 (RL) - V5.6: 보상 균등화 + 체류 패널티
        # =========================================================
        align_mask = (self.phase == PHASE_ALIGN)
        if align_mask.any():
            # 위치 reward (펜 축으로 이동 유도)
            rewards[align_mask] += self.cfg.rew_scale_perp_dist * perpendicular_dist[align_mask]

            # V5.6 수정: 자세 정렬 reward (위치 조건부 + exponential + 낮은 가중치)
            align_quality = (-dot_product[align_mask] - 0.5) / 0.5

            # 거리가 가까울수록 자세 reward 가중치가 exponential하게 증가 (max 2.0)
            base_dist = 0.05  # ALIGN은 이미 가까우므로 기준 5cm
            scale = 30.0
            position_weight = torch.exp((base_dist - perpendicular_dist[align_mask]) * scale)
            position_weight = torch.clamp(position_weight, min=0, max=2.0)  # V5.6: 5.0 → 2.0

            rewards[align_mask] += self.cfg.rew_scale_align_orientation * torch.clamp(align_quality, min=0) * position_weight

            # 정밀 정렬 보너스 (위치 조건부)
            align_value = -dot_product[align_mask]
            exponential_bonus = torch.where(
                align_value > self.cfg.exponential_align_threshold,
                torch.exp((align_value - self.cfg.exponential_align_threshold) * self.cfg.exponential_align_scale),
                torch.ones_like(align_value)
            )
            rewards[align_mask] += self.cfg.rew_scale_exponential_align * exponential_bonus * position_weight

            # 펜 축에 가까워지면 보너스
            on_axis_align = perpendicular_dist[align_mask] < 0.04
            rewards[align_mask] += self.cfg.rew_scale_on_axis_bonus * on_axis_align.float()

            # V5.6: Phase 체류 패널티 (오래 머물수록 페널티 증가)
            rewards[align_mask] += self.cfg.rew_scale_phase_stagnation * self.phase_step_count[align_mask].float()

        # =========================================================
        # FINE_ALIGN 단계 (TCP) - 고정 보상
        # =========================================================
        fine_align_mask = (self.phase == PHASE_FINE_ALIGN)
        if fine_align_mask.any():
            # 정렬 개선에 따른 보상
            align_progress = -dot_product[fine_align_mask]  # 클수록 좋음 (최대 1)
            rewards[fine_align_mask] += self.cfg.rew_scale_fine_align * align_progress

        # =========================================================
        # DESCEND 단계 (TCP) - 고정 보상
        # =========================================================
        descend_mask = (self.phase == PHASE_DESCEND)
        if descend_mask.any():
            # 하강 진행에 따른 보상
            descend_progress = self.prev_distance_to_cap[descend_mask] - distance_to_cap[descend_mask]
            rewards[descend_mask] += self.cfg.rew_scale_descend * torch.clamp(descend_progress * 100, min=0, max=1)

        # =========================================================
        # GRASP 단계 (TCP)
        # =========================================================
        grasp_mask = (self.phase == PHASE_GRASP)
        if grasp_mask.any():
            gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]
            gripper_closed_amount = gripper_pos.mean(dim=-1)
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_close * gripper_closed_amount[grasp_mask]
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_hold * (1.0 - distance_to_cap[grasp_mask])

        # =========================================================
        # 단계 전환
        # =========================================================
        # APPROACH → ALIGN (완화된 조건)
        transition_to_align = approach_mask & (perpendicular_dist < APPROACH_TO_ALIGN_PERP_DIST) & (torch.abs(axis_distance) < APPROACH_TO_ALIGN_AXIS_DIST)
        if transition_to_align.any():
            self.phase[transition_to_align] = PHASE_ALIGN
            rewards[transition_to_align] += self.cfg.rew_scale_phase_transition
            self.align_target_pos[transition_to_align] = grasp_pos[transition_to_align].clone()
            self.phase_step_count[transition_to_align] = 0

        # ALIGN → FINE_ALIGN (V5.2: 자세 + 위치 동시 체크)
        transition_to_fine_align = (
            align_mask &
            (dot_product < ALIGN_TO_FINE_ALIGN_DOT) &
            (perpendicular_dist < ALIGN_TO_FINE_ALIGN_PERP_DIST)
        )
        if transition_to_fine_align.any():
            self.phase[transition_to_fine_align] = PHASE_FINE_ALIGN
            rewards[transition_to_fine_align] += self.cfg.rew_scale_phase_transition
            self.fine_align_target_pos[transition_to_fine_align] = grasp_pos[transition_to_fine_align].clone()
            self.phase_step_count[transition_to_fine_align] = 0

        # FINE_ALIGN → DESCEND (V5.2: 자세 + 위치 동시 체크)
        transition_to_descend = (
            fine_align_mask &
            (dot_product < FINE_ALIGN_TO_DESCEND_DOT) &
            (perpendicular_dist < FINE_ALIGN_TO_DESCEND_PERP_DIST)
        )
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
            self.grasp_step_count[transition_to_grasp] = 0  # V5.4: GRASP 스텝 초기화

        # GRASP → LIFT (V5.4: Good Grasp 조건)
        # Good Grasp 조건:
        # 1. perp_dist < 8mm (펜 축과 grasp point 일치)
        # 2. dot < -0.98 (자세 정렬)
        # 3. 그리퍼가 0.8~1.05 사이 (펜에 걸린 상태)
        # 4. 30 스텝 이상 유지
        GRASP_HOLD_STEPS = 30
        gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]
        gripper_closed_amount = gripper_pos.mean(dim=-1)

        good_grasp = (
            (perpendicular_dist < GOOD_GRASP_PERP_DIST) &
            (dot_product < GOOD_GRASP_DOT) &
            (gripper_closed_amount > GOOD_GRASP_GRIPPER_MIN) &
            (gripper_closed_amount < GOOD_GRASP_GRIPPER_MAX)
        )

        transition_to_lift = grasp_mask & good_grasp & (self.grasp_step_count >= GRASP_HOLD_STEPS)
        if transition_to_lift.any():
            self.phase[transition_to_lift] = PHASE_LIFT
            rewards[transition_to_lift] += self.cfg.rew_scale_phase_transition * 2.0  # 보너스
            self.phase_step_count[transition_to_lift] = 0
            # LIFT 시작 위치 저장
            self.lift_start_pos[transition_to_lift] = grasp_pos[transition_to_lift]

        # =========================================================
        # 성공 보상 (V5.4: LIFT 완료 + Good Grasp 유지)
        # =========================================================
        lift_mask = (self.phase == PHASE_LIFT)
        success = lift_mask & self.lift_complete & good_grasp
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # =========================================================
        # 페널티
        # =========================================================
        collision = self._check_pen_collision()
        collision_penalty_mask = collision & (self.phase < PHASE_DESCEND)
        if collision_penalty_mask.any():
            rewards[collision_penalty_mask] += self.cfg.rew_scale_collision

        # 액션 페널티 (RL 단계만)
        rl_mask = (self.phase == PHASE_APPROACH) | (self.phase == PHASE_ALIGN)
        rewards[rl_mask] += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions[rl_mask]), dim=-1)

        # =========================================================
        # 이전 거리 업데이트
        # =========================================================
        self.prev_axis_distance = torch.abs(axis_distance + PRE_GRASP_AXIS_DIST)
        self.prev_distance_to_cap[descend_mask] = distance_to_cap[descend_mask]

        # =========================================================
        # Phase 분포 출력 (N step마다)
        # =========================================================
        self._global_step += 1
        if self._global_step % self._phase_print_interval == 0:
            phase_stats = self.get_phase_stats()
            print(f"  [Step {self._global_step}] Phase: "
                  f"APP:{phase_stats['approach']} ALN:{phase_stats['align']} "
                  f"FINE:{phase_stats['fine_align']} DESC:{phase_stats['descend']} "
                  f"GRP:{phase_stats['grasp']} LIFT:{phase_stats['lift']} "
                  f"| Success:{phase_stats['total_success']}", flush=True)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건 (V5.4: LIFT 완료 + Good Grasp)"""
        # Good Grasp 조건 계산
        perpendicular_dist, _, _ = self._compute_axis_metrics()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)

        gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]
        gripper_closed_amount = gripper_pos.mean(dim=-1)

        good_grasp = (
            (perpendicular_dist < GOOD_GRASP_PERP_DIST) &
            (dot_product < GOOD_GRASP_DOT) &
            (gripper_closed_amount > GOOD_GRASP_GRIPPER_MIN) &
            (gripper_closed_amount < GOOD_GRASP_GRIPPER_MAX)
        )

        # V5.4: LIFT 완료 + Good Grasp가 성공 조건
        lift_mask = (self.phase == PHASE_LIFT)
        success = lift_mask & self.lift_complete & good_grasp
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

        # Z축 기준 원뿔 각도로 펜 방향 설정 (V5: Curriculum Learning 적용)
        # θ (tilt): 0 ~ max_tilt (curriculum_level에 따라 결정), φ (azimuth): 0 ~ 2π
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

        # 상태 리셋
        self.phase[env_ids] = PHASE_APPROACH
        self.phase_step_count[env_ids] = 0
        self.gripper_closed[env_ids] = False

        # V5.4: LIFT 상태 리셋
        self.lift_start_pos[env_ids] = 0.0
        self.lift_complete[env_ids] = False
        self.grasp_step_count[env_ids] = 0

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
        """단계별 통계 (V5.4: LIFT 추가)"""
        current_phases = torch.bincount(self.phase, minlength=6)  # V5.4: 6단계
        return {
            "approach": current_phases[0].item(),
            "align": current_phases[1].item(),
            "fine_align": current_phases[2].item(),
            "descend": current_phases[3].item(),
            "grasp": current_phases[4].item(),
            "lift": current_phases[5].item(),  # V5.4
            "total_success": self.success_count.sum().item(),
            "curriculum_level": self.cfg.curriculum_level,
        }


@configclass
class E0509IKEnvV5Cfg_PLAY(E0509IKEnvV5Cfg):
    """테스트용 설정"""

    def __post_init__(self):
        self.scene.num_envs = 50


# Curriculum Level별 설정 (학습용)
@configclass
class E0509IKEnvV5Cfg_L0(E0509IKEnvV5Cfg):
    """Level 0: 펜 수직 (기본 동작 학습)"""
    curriculum_level = 0


@configclass
class E0509IKEnvV5Cfg_L1(E0509IKEnvV5Cfg):
    """Level 1: 펜 10° 기울기"""
    curriculum_level = 1


@configclass
class E0509IKEnvV5Cfg_L2(E0509IKEnvV5Cfg):
    """Level 2: 펜 20° 기울기"""
    curriculum_level = 2


@configclass
class E0509IKEnvV5Cfg_L3(E0509IKEnvV5Cfg):
    """Level 3: 펜 30° 기울기 (최종 목표)"""
    curriculum_level = 3
