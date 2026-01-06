"""
E0509 전문가 궤적 생성용 환경

IK 방식처럼 펜 축 기준으로 접근하는 전문가 정책용 환경.
OSC 환경과 분리하여 전문가 궤적 수집에 최적화.

특징:
- 펜 축 방향으로 접근 (충돌 회피)
- 자동 자세 정렬
- 간단한 성공/실패 판정
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

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG


# =============================================================================
# 경로 및 상수
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")
PEN_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "pen.usd")

PEN_LENGTH = 0.1207  # 120.7mm

# 성공 조건
SUCCESS_DIST_TO_CAP = 0.03       # 캡까지 거리 < 3cm
SUCCESS_HOLD_STEPS = 30          # 30 스텝 유지하면 성공


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class E0509ExpertEnvCfg(DirectRLEnvCfg):
    """전문가 궤적 생성용 환경 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 15.0
    action_scale = 0.05
    action_space = 3          # [Δx, Δy, Δz]
    observation_space = 30    # 기존 27 + 추가 정보 3
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

    # 로봇 설정 (IK 방식: 위치 제어)
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
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.57,
                "joint_4": 0.0,
                "joint_5": 1.57,
                "joint_6": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=3.14,
                stiffness=400.0,   # IK: 높은 stiffness
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_rh_.*"],
                effort_limit=10.0,
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
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 보상 스케일
    rew_scale_distance = 1.0
    rew_scale_success = 10.0


# =============================================================================
# 환경 클래스
# =============================================================================
class E0509ExpertEnv(DirectRLEnv):
    """전문가 궤적 생성용 환경"""

    cfg: E0509ExpertEnvCfg

    def __init__(self, cfg: E0509ExpertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스
        self._arm_joint_ids = list(range(6))
        self._gripper_joint_ids = list(range(6, 10))

        # EE 바디 인덱스 (그리퍼 베이스 사용)
        self._ee_body_idx = self.robot.find_bodies("gripper_rh_p12_rn_base")[0][0]

        # 손가락 바디 인덱스
        self._finger_l1_idx = self.robot.find_bodies("gripper_rh_p12_rn_l1")[0][0]
        self._finger_r1_idx = self.robot.find_bodies("gripper_rh_p12_rn_r1")[0][0]
        self._finger_l2_idx = self.robot.find_bodies("gripper_rh_p12_rn_l2")[0][0]
        self._finger_r2_idx = self.robot.find_bodies("gripper_rh_p12_rn_r2")[0][0]

        # EE 오프셋
        self._ee_offset_pos = torch.zeros(3, device=self.device)
        self._ee_offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # 상태 변수
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)
        self.success_hold_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        # 액션 스케일
        self.action_scale = self.cfg.action_scale

        # IK 관련
        self._setup_ik()

        # 시각화 마커 설정
        self._setup_markers()

    def _setup_markers(self):
        """디버그 시각화 마커 설정"""
        # Grasp point 마커 (녹색 구)
        grasp_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/GraspMarker",
            markers={
                "grasp": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        self._grasp_marker = VisualizationMarkers(grasp_marker_cfg)

        # Pen cap 마커 (빨간 구)
        cap_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CapMarker",
            markers={
                "cap": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self._cap_marker = VisualizationMarkers(cap_marker_cfg)

        # 접근 목표 마커 (파란 구)
        approach_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ApproachMarker",
            markers={
                "approach": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        )
        self._approach_marker = VisualizationMarkers(approach_marker_cfg)

    def _update_markers(self):
        """마커 위치 업데이트"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        pen_z = self._get_pen_z_axis()
        approach_pos = cap_pos - pen_z * 0.05  # 접근 위치 (캡에서 5cm)

        # 마커 업데이트
        grasp_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        self._grasp_marker.visualize(grasp_pos, grasp_quat)
        self._cap_marker.visualize(cap_pos, grasp_quat)
        self._approach_marker.visualize(approach_pos, grasp_quat)

    def _setup_scene(self):
        """씬 구성"""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.pen = RigidObject(self.cfg.pen_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["pen"] = self.pen

        # 라이트
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_ik(self):
        """IK 설정"""
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        )
        self._ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # Jacobian 인덱스 (그리퍼 베이스 사용)
        self._jacobi_body_idx = self._ee_body_idx - 1

        # 목표 포즈 저장
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat[:, 0] = 1.0  # 단위 쿼터니언

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """물리 시뮬레이션 전 처리"""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """액션 적용 (IK 기반)"""
        # 현재 EE 상태
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # 위치 변화량
        pos_delta = self.actions * self.action_scale

        # 목표 자세: 펜 축 기반 자동 계산
        target_quat = self._compute_auto_orientation()

        # 상대 회전 (axis-angle)
        quat_curr_inv = math_utils.quat_inv(ee_quat_b)
        quat_delta = math_utils.quat_mul(target_quat, quat_curr_inv)
        rot_delta = math_utils.axis_angle_from_quat(quat_delta) * 0.5

        # IK 명령: 상대 위치 + 상대 회전
        command = torch.cat([pos_delta, rot_delta], dim=-1)

        # Jacobian
        jacobian_w = self.robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._arm_joint_ids
        ]
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # 현재 관절 위치
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]

        # IK 계산 - 상대 위치/회전 명령 설정
        self._ik_controller.set_command(command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_target = self._ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian_b, joint_pos
        )

        # 관절 위치 타겟 설정
        self.robot.set_joint_position_target(joint_pos_target, joint_ids=self._arm_joint_ids)

        # 그리퍼는 열린 상태 유지
        gripper_target = torch.zeros(self.num_envs, 4, device=self.device)
        self.robot.set_joint_position_target(gripper_target, joint_ids=self._gripper_joint_ids)

    def _compute_auto_orientation(self) -> torch.Tensor:
        """펜 축 기반 자동 자세 계산"""
        pen_z = self._get_pen_z_axis()

        # 그리퍼 Z축 = -펜 Z축
        gripper_z = -pen_z

        # 그리퍼 X축: 펜 축과 월드 Z축의 외적
        world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        gripper_x = torch.cross(world_z, gripper_z, dim=-1)
        gripper_x_norm = torch.norm(gripper_x, dim=-1, keepdim=True)

        # 펜이 수직일 때
        nearly_vertical = gripper_x_norm.squeeze(-1) < 0.1
        world_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        gripper_x = torch.where(
            nearly_vertical.unsqueeze(-1).expand(-1, 3),
            world_x,
            gripper_x / (gripper_x_norm + 1e-6)
        )

        gripper_y = torch.cross(gripper_z, gripper_x, dim=-1)
        gripper_y = gripper_y / (torch.norm(gripper_y, dim=-1, keepdim=True) + 1e-6)

        rot_matrix = torch.stack([gripper_x, gripper_y, gripper_z], dim=-1)
        return math_utils.quat_from_matrix(rot_matrix)

    def _get_pen_z_axis(self) -> torch.Tensor:
        """펜의 Z축 (길이 방향) 가져오기"""
        pen_quat = self.pen.data.root_quat_w
        z_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        return math_utils.quat_apply(pen_quat, z_local)

    def _get_pen_cap_pos(self) -> torch.Tensor:
        """펜 캡 위치"""
        pen_pos = self.pen.data.root_pos_w
        pen_z = self._get_pen_z_axis()
        return pen_pos + pen_z * (PEN_LENGTH / 2)

    def _get_grasp_point(self) -> torch.Tensor:
        """그립 포인트 위치 (손가락 중심)"""
        # 손가락 위치
        l1 = self.robot.data.body_pos_w[:, self._finger_l1_idx]
        r1 = self.robot.data.body_pos_w[:, self._finger_r1_idx]
        l2 = self.robot.data.body_pos_w[:, self._finger_l2_idx]
        r2 = self.robot.data.body_pos_w[:, self._finger_r2_idx]

        # 베이스 중심과 팁 중심
        base_center = (l1 + r1) / 2.0
        tip_center = (l2 + r2) / 2.0

        # 손가락 방향
        finger_dir = tip_center - base_center
        finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

        # 베이스에서 2cm 앞 (팁 방향)
        return base_center + finger_dir * 0.02

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

        distance_to_cap = torch.norm(rel_pos, dim=-1)

        # 펜 축 방향 접근 정보 추가
        # 그리퍼가 펜 축 방향에서 얼마나 벗어났는지
        approach_dir = -pen_z  # 접근 방향 (펜 축 반대)
        approach_offset = grasp_pos - cap_pos
        approach_dist_along_axis = torch.sum(approach_offset * approach_dir, dim=-1)  # 축 방향 거리

        obs = torch.cat([
            joint_pos,                              # 6
            joint_vel,                              # 6
            grasp_pos_local,                        # 3
            cap_pos_local,                          # 3
            rel_pos,                                # 3
            pen_z,                                  # 3
            distance_to_cap.unsqueeze(-1),          # 1
            approach_dir,                           # 3 (추가: 접근 방향)
            approach_dist_along_axis.unsqueeze(-1), # 1 (추가: 축 방향 거리)
            torch.zeros(self.num_envs, 1, device=self.device),  # 1 (phase placeholder)
        ], dim=-1)  # 총 30

        self.prev_distance_to_cap = distance_to_cap.clone()

        # 시각화 마커 업데이트
        self._update_markers()

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """보상 계산"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        distance_to_cap = torch.norm(cap_pos - grasp_pos, dim=-1)

        # 거리 보상
        rewards = -self.cfg.rew_scale_distance * distance_to_cap

        # 성공 보상
        success = distance_to_cap < SUCCESS_DIST_TO_CAP
        rewards += self.cfg.rew_scale_success * success.float()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        distance_to_cap = torch.norm(cap_pos - grasp_pos, dim=-1)

        # 성공 조건
        near_cap = distance_to_cap < SUCCESS_DIST_TO_CAP
        self.success_hold_count = torch.where(
            near_cap,
            self.success_hold_count + 1,
            torch.zeros_like(self.success_hold_count)
        )
        success = self.success_hold_count >= SUCCESS_HOLD_STEPS

        # 시간 초과
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        """환경 리셋"""
        super()._reset_idx(env_ids)

        # 로봇 리셋
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 펜 랜덤 위치
        num_resets = len(env_ids)

        # 위치: 로봇 앞 랜덤
        pen_pos = torch.zeros(num_resets, 3, device=self.device)
        pen_pos[:, 0] = sample_uniform(0.35, 0.55, (num_resets,), device=self.device)
        pen_pos[:, 1] = sample_uniform(-0.15, 0.15, (num_resets,), device=self.device)
        pen_pos[:, 2] = 0.02

        # 자세: 랜덤 회전 (테이블 위에 눕힘)
        roll = sample_uniform(-0.3, 0.3, (num_resets,), device=self.device)
        pitch = sample_uniform(1.4, 1.7, (num_resets,), device=self.device)  # ~90도 눕힘
        yaw = sample_uniform(-3.14, 3.14, (num_resets,), device=self.device)
        pen_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # 환경 원점 오프셋
        pen_pos += self.scene.env_origins[env_ids]

        # 펜 상태 설정
        pen_vel = torch.zeros(num_resets, 6, device=self.device)
        self.pen.write_root_pose_to_sim(torch.cat([pen_pos, pen_quat], dim=-1), env_ids=env_ids)
        self.pen.write_root_velocity_to_sim(pen_vel, env_ids=env_ids)

        # 상태 리셋
        self.success_hold_count[env_ids] = 0

        # 거리 초기화
        grasp_pos = self._get_grasp_point()[env_ids]
        cap_pos = self._get_pen_cap_pos()[env_ids]
        self.prev_distance_to_cap[env_ids] = torch.norm(cap_pos - grasp_pos, dim=-1)


# =============================================================================
# Play용 설정
# =============================================================================
@configclass
class E0509ExpertEnvCfg_PLAY(E0509ExpertEnvCfg):
    """Play용 설정"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 30.0
