"""
E0509 IK 환경 (Task Space Control)

=== 목표 ===
IK(Inverse Kinematics)를 사용하여 Task Space에서 직접 제어
그리퍼가 위에서 아래로 내려다보는 자세로 접근하도록 유도

=== 액션 공간 ===
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw] (6차원)
- 상대적인 end-effector 위치/자세 변화량

=== 장점 ===
- 직관적인 액션 공간 (Task space)
- 초기 자세를 강제할 수 있음 (위에서 내려다보기)
- Sim2Real 전환 용이

=== 사용법 ===
python train_ik.py --headless --num_envs 4096
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

# 단계 정의 (PRE_GRASP → DESCEND)
PHASE_PRE_GRASP = 0   # 펜캡 위 7cm + 정렬
PHASE_DESCEND = 1     # 수직 하강

# Pre-grasp 설정
PRE_GRASP_HEIGHT = 0.07  # 펜캡 위 7cm

# 단계 전환 조건
PRE_GRASP_TO_DESCEND_DIST = 0.03   # pre-grasp 위치에서 3cm 이내
PRE_GRASP_TO_DESCEND_DOT = -0.95   # 정렬 조건 (약 18도 이내)

# 성공 조건
SUCCESS_DIST = 0.02     # 2cm
SUCCESS_DOT = -0.95     # dot < -0.95


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class E0509IKEnvCfg(DirectRLEnvCfg):
    """E0509 IK 환경 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 12.0
    action_scale = 0.02   # Task space에서의 스케일 (미터/라디안)
    action_space = 6      # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
    observation_space = 33  # 기존 27 + ee_pos(3) + ee_quat(3, axis-angle)
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

    # 로봇 설정 (IK에 맞는 높은 PD gain)
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
                "joint_2": -0.5,      # 어깨 약간 앞으로
                "joint_3": 1.0,       # 팔꿈치 굽힘
                "joint_4": 0.0,
                "joint_5": 1.57,      # 손목 90도 - 그리퍼가 아래를 향함
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
                stiffness=800.0,      # IK 추종을 위해 높은 강성
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

    # 목표 위치 범위 (펜 리셋용)
    pen_pos_range = {
        "x": (0.3, 0.5),
        "y": (-0.20, 0.20),
        "z": (0.20, 0.50),
    }

    # 펜 방향 랜덤화 (1단계: 고정)
    pen_rot_range = {
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    # IK 컨트롤러 설정
    ik_method = "dls"  # damped least squares
    ik_lambda = 0.05   # damping coefficient

    # End-effector body name (link_6)
    ee_body_name = "link_6"

    # End-effector offset (그리퍼 끝까지의 오프셋)
    ee_offset_pos = [0.0, 0.0, 0.15]  # Z 방향으로 15cm (그리퍼 길이)

    # 보상 스케일 (기존과 동일)
    rew_scale_pregrasp_dist = -8.0
    rew_scale_pregrasp_progress = 20.0
    rew_scale_pregrasp_align = 1.5
    rew_scale_descend_dist = -10.0
    rew_scale_descend_align = 2.0
    rew_scale_descend_progress = 15.0
    rew_scale_success = 100.0
    rew_scale_phase_transition = 15.0
    rew_scale_action = -0.01
    rew_scale_wrong_direction = -2.0
    rew_scale_exponential_align = 0.3
    exponential_align_threshold = 0.9
    exponential_align_scale = 10.0


class E0509IKEnv(DirectRLEnv):
    """E0509 IK 환경 (Task Space Control)"""

    cfg: E0509IKEnvCfg

    def __init__(self, cfg: E0509IKEnvCfg, render_mode: str | None = None, **kwargs):
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

        # Fixed-base이므로 jacobian body index 조정
        if self.robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._arm_joint_ids]

        print(f"[E0509IKEnv] EE body: {body_names[0]} (idx={self._ee_body_idx})")
        print(f"[E0509IKEnv] Jacobian body idx: {self._jacobi_body_idx}")
        print(f"[E0509IKEnv] Arm joints: {self._arm_joint_names}")

        # IK Controller 생성
        self._ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,  # Delta pose mode
                ik_method=self.cfg.ik_method,
                ik_params={"lambda_val": self.cfg.ik_lambda} if self.cfg.ik_method == "dls" else None,
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

        # EE offset (그리퍼 끝 위치)
        self._ee_offset_pos = torch.tensor(
            self.cfg.ee_offset_pos, device=self.device
        ).repeat(self.num_envs, 1)
        self._ee_offset_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device  # Identity quaternion
        ).repeat(self.num_envs, 1)

        # 액션 스케일
        self.action_scale = self.cfg.action_scale

        # 관절 한계 (소프트 리밋 사용)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :6, 0]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :6, 1]

        # 상태 머신
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 이전 거리 (progress 보상용)
        self.prev_distance_to_pregrasp = torch.zeros(self.num_envs, device=self.device)
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)

        # 성공 카운터
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

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
        """IK를 사용한 액션 적용"""
        # 현재 EE pose 계산 (root frame 기준)
        ee_pos_curr, ee_quat_curr = self._compute_ee_pose()

        # 액션 스케일 적용
        scaled_actions = self.actions * self.action_scale

        # IK 컨트롤러에 명령 설정 (상대 모드)
        self._ik_controller.set_command(scaled_actions, ee_pos_curr, ee_quat_curr)

        # Jacobian 계산
        jacobian = self._compute_ee_jacobian()

        # 현재 관절 위치
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]

        # IK로 목표 관절 위치 계산
        joint_pos_target = self._ik_controller.compute(
            ee_pos_curr, ee_quat_curr, jacobian, joint_pos
        )

        # 관절 한계 클램핑
        joint_pos_target = torch.clamp(
            joint_pos_target,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # 전체 관절 타겟 설정
        full_target = torch.zeros(self.num_envs, 10, device=self.device)
        full_target[:, :6] = joint_pos_target
        full_target[:, 6:] = 0.0  # 그리퍼 열림

        self.robot.set_joint_position_target(full_target)

    def _compute_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """End-effector pose 계산 (root frame 기준, offset 포함)"""
        # EE body pose (world frame)
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        # Root pose (world frame)
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        # EE pose in root frame
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # Offset 적용 (그리퍼 끝 위치)
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._ee_offset_pos, self._ee_offset_rot
        )

        return ee_pos_b, ee_quat_b

    def _compute_ee_jacobian(self) -> torch.Tensor:
        """End-effector Jacobian 계산 (root frame 기준)"""
        # World frame Jacobian
        jacobian_w = self.robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]

        # Root frame으로 변환
        base_rot = self.robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))

        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # Offset 보정
        jacobian_b[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._ee_offset_pos),
            jacobian_b[:, 3:, :]
        )

        return jacobian_b

    def _get_observations(self) -> dict:
        """관찰값 계산"""
        # 관절 상태
        joint_pos = self.robot.data.joint_pos[:, :6]
        joint_vel = self.robot.data.joint_vel[:, :6]

        # 그리퍼 위치 (grasp point)
        grasp_pos = self._get_grasp_point()
        grasp_pos_local = grasp_pos - self.scene.env_origins

        # 펜 캡 위치
        cap_pos = self._get_pen_cap_pos()
        cap_pos_local = cap_pos - self.scene.env_origins

        # 상대 위치
        rel_pos = cap_pos - grasp_pos

        # 축 방향
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        # EE pose (추가)
        ee_pos, ee_quat = self._compute_ee_pose()
        # Quaternion을 axis-angle로 변환 (3D)
        ee_axis_angle = math_utils.axis_angle_from_quat(ee_quat)

        # 관찰 결합
        obs = torch.cat([
            joint_pos,       # 6
            joint_vel,       # 6
            grasp_pos_local, # 3
            cap_pos_local,   # 3
            rel_pos,         # 3
            gripper_z,       # 3
            pen_z,           # 3
            ee_pos,          # 3 (추가)
            ee_axis_angle,   # 3 (추가)
        ], dim=-1)           # 총 33

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Pre-grasp 방식 단계별 보상 계산 (기존과 동일)"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        pregrasp_pos = self._get_pregrasp_pos()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        distance_to_pregrasp = torch.norm(grasp_pos - pregrasp_pos, dim=-1)
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        # 지수적 정렬 보너스
        align_value = -dot_product
        exponential_bonus = torch.where(
            align_value > self.cfg.exponential_align_threshold,
            torch.exp((align_value - self.cfg.exponential_align_threshold) * self.cfg.exponential_align_scale),
            torch.ones_like(align_value)
        )

        # PRE_GRASP 단계
        pregrasp_mask = (self.phase == PHASE_PRE_GRASP)
        if pregrasp_mask.any():
            rewards[pregrasp_mask] += self.cfg.rew_scale_pregrasp_dist * distance_to_pregrasp[pregrasp_mask]
            progress = self.prev_distance_to_pregrasp[pregrasp_mask] - distance_to_pregrasp[pregrasp_mask]
            rewards[pregrasp_mask] += self.cfg.rew_scale_pregrasp_progress * torch.clamp(progress, min=0)
            align_quality = (-dot_product[pregrasp_mask] - 0.5) / 0.5
            rewards[pregrasp_mask] += self.cfg.rew_scale_pregrasp_align * torch.clamp(align_quality, min=0)
            rewards[pregrasp_mask] += self.cfg.rew_scale_exponential_align * exponential_bonus[pregrasp_mask]

        # DESCEND 단계
        descend_mask = (self.phase == PHASE_DESCEND)
        if descend_mask.any():
            rewards[descend_mask] += self.cfg.rew_scale_descend_dist * distance_to_cap[descend_mask]
            align_maintain = -dot_product[descend_mask]
            rewards[descend_mask] += self.cfg.rew_scale_descend_align * align_maintain
            rewards[descend_mask] += self.cfg.rew_scale_exponential_align * exponential_bonus[descend_mask]
            descend_progress = self.prev_distance_to_cap[descend_mask] - distance_to_cap[descend_mask]
            rewards[descend_mask] += self.cfg.rew_scale_descend_progress * torch.clamp(descend_progress, min=0)

            align_lost = dot_product[descend_mask] > -0.9
            if align_lost.any():
                descend_indices = torch.where(descend_mask)[0]
                lost_indices = descend_indices[align_lost]
                rewards[lost_indices] -= 5.0

        # 단계 전환
        transition_to_descend = pregrasp_mask & (distance_to_pregrasp < PRE_GRASP_TO_DESCEND_DIST) & (dot_product < PRE_GRASP_TO_DESCEND_DOT)
        if transition_to_descend.any():
            self.phase[transition_to_descend] = PHASE_DESCEND
            rewards[transition_to_descend] += self.cfg.rew_scale_phase_transition
            self.prev_distance_to_cap[transition_to_descend] = distance_to_cap[transition_to_descend]

        # 성공 보상
        success = (distance_to_cap < SUCCESS_DIST) & (dot_product < SUCCESS_DOT)
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # 액션 페널티
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # 반대 방향 페널티
        wrong_direction_mask = dot_product > 0
        if wrong_direction_mask.any():
            rewards[wrong_direction_mask] += self.cfg.rew_scale_wrong_direction * dot_product[wrong_direction_mask]

        # 이전 거리 업데이트
        self.prev_distance_to_pregrasp = distance_to_pregrasp.clone()
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

        success = (distance < SUCCESS_DIST) & (dot_product < SUCCESS_DOT)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """환경 리셋"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        super()._reset_idx(env_ids)

        # 로봇 리셋 (위에서 내려다보는 초기 자세)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()

        # 약간의 랜덤 노이즈 추가 (초기 자세 근처에서)
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

        # 펜 리셋 (랜덤 위치)
        pen_state = self.pen.data.default_root_state[env_ids].clone()

        pen_pos_x = sample_uniform(
            self.cfg.pen_pos_range["x"][0], self.cfg.pen_pos_range["x"][1],
            (len(env_ids),), device=self.device
        )
        pen_pos_y = sample_uniform(
            self.cfg.pen_pos_range["y"][0], self.cfg.pen_pos_range["y"][1],
            (len(env_ids),), device=self.device
        )
        pen_pos_z = sample_uniform(
            self.cfg.pen_pos_range["z"][0], self.cfg.pen_pos_range["z"][1],
            (len(env_ids),), device=self.device
        )

        pen_state[:, 0] = self.scene.env_origins[env_ids, 0] + pen_pos_x
        pen_state[:, 1] = self.scene.env_origins[env_ids, 1] + pen_pos_y
        pen_state[:, 2] = self.scene.env_origins[env_ids, 2] + pen_pos_z

        # 펜 방향 랜덤화
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

        # 단계 리셋
        self.phase[env_ids] = PHASE_PRE_GRASP

        # IK 컨트롤러 리셋
        self._ik_controller.reset(env_ids_tensor)

        # 이전 거리 초기화
        grasp_pos = self._get_grasp_point()
        pregrasp_pos = self._get_pregrasp_pos()
        cap_pos = self._get_pen_cap_pos()

        self.prev_distance_to_pregrasp[env_ids] = torch.norm(
            grasp_pos[env_ids] - pregrasp_pos[env_ids], dim=-1
        )
        self.prev_distance_to_cap[env_ids] = torch.norm(
            grasp_pos[env_ids] - cap_pos[env_ids], dim=-1
        )

    # =============================================================================
    # 헬퍼 함수들
    # =============================================================================
    def _get_grasp_point(self) -> torch.Tensor:
        """그리퍼 잡기 포인트 계산"""
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

    def _get_pregrasp_pos(self) -> torch.Tensor:
        """Pre-grasp 위치"""
        cap_pos = self._get_pen_cap_pos()
        pen_z = self._get_pen_z_axis()
        return cap_pos + PRE_GRASP_HEIGHT * pen_z

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
        """오일러 각도를 쿼터니언으로 변환"""
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
        """단계별 통계 반환"""
        current_phases = torch.bincount(self.phase, minlength=2)
        return {
            "pre_grasp": current_phases[0].item(),
            "descend": current_phases[1].item(),
            "total_success": self.success_count.sum().item(),
        }


@configclass
class E0509IKEnvCfg_PLAY(E0509IKEnvCfg):
    """테스트용 설정"""

    def __post_init__(self):
        self.scene.num_envs = 50
