"""
E0509 Direct 환경 (단계별 상태 머신)

=== 목표 ===
Manager-Based와 비교하기 위한 Direct 구현
단계별로 다른 보상을 적용하여 복잡한 작업 학습

=== 단계 (Phase) ===
1. APPROACH: 펜 캡에 접근 (거리 < 10cm 시 전환)
2. ALIGN: z축 정렬 (dot < -0.8 시 전환)
3. GRASP: 잡기 시도 (거리 < 2cm & dot < -0.9 시 성공)

=== 사용법 ===
python train_direct.py --headless --num_envs 4096
"""
from __future__ import annotations

import os
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.actuators import ImplicitActuatorCfg


# =============================================================================
# 경로 및 상수
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")
PEN_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "pen.usd")

PEN_LENGTH = 0.1207  # 120.7mm

# 단계 정의
PHASE_APPROACH = 0
PHASE_ALIGN = 1
PHASE_GRASP = 2

# 단계 전환 조건
APPROACH_TO_ALIGN_DIST = 0.10  # 10cm 이내면 align 단계로
ALIGN_TO_GRASP_DOT = -0.8      # dot < -0.8이면 grasp 단계로
SUCCESS_DIST = 0.02            # 2cm
SUCCESS_DOT = -0.9             # dot < -0.9


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class E0509DirectEnvCfg(DirectRLEnvCfg):
    """E0509 Direct 환경 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 12.0
    action_scale = 0.1
    action_space = 6      # 6 DOF 팔
    observation_space = 27  # joint_pos(6) + joint_vel(6) + grasp_pos(3) + cap_pos(3) + rel_pos(3) + gripper_z(3) + pen_z(3)
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
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
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
                stiffness=400.0,
                damping=40.0,
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
        "y": (-0.15, 0.15),
        "z": (0.25, 0.35),
    }

    # 펜 방향 랜덤화 범위 (라디안)
    # roll: x축 회전, pitch: y축 회전, yaw: z축 회전
    # === 1단계: 수직 고정 (기본 동작 학습) ===
    pen_rot_range = {
        "roll": (0.0, 0.0),     # 고정
        "pitch": (0.0, 0.0),    # 고정
        "yaw": (0.0, 0.0),      # 고정
    }
    # === 2단계: 랜덤화 (나중에 활성화) ===
    # pen_rot_range = {
    #     "roll": (-0.5, 0.5),    # ±30도 정도
    #     "pitch": (-0.5, 0.5),   # ±30도 정도
    #     "yaw": (-3.14, 3.14),   # 전체 회전
    # }

    # 데이터 수집 설정
    # === 1단계: 비활성화 ===
    collect_data = False  # 2단계에서 True로 변경
    data_save_path = "./logs/feasibility_data"  # 저장 경로

    # 보상 스케일
    rew_scale_approach_dist = -2.0      # 접근 단계: 거리 페널티
    rew_scale_approach_progress = 5.0   # 접근 단계: 거리 감소 보상
    rew_scale_align_dist = -1.0         # 정렬 단계: 거리 유지 페널티
    rew_scale_align_dot = 2.0           # 정렬 단계: 정렬 보상
    rew_scale_grasp_dist = -3.0         # 잡기 단계: 거리 페널티
    rew_scale_grasp_dot = 1.0           # 잡기 단계: 정렬 유지 보상
    rew_scale_success = 100.0           # 성공 보상
    rew_scale_phase_transition = 10.0   # 단계 전환 보상
    rew_scale_action = -0.001           # 액션 페널티
    rew_scale_wrong_direction = -1.0    # dot 양수(반대 방향) 페널티


class E0509DirectEnv(DirectRLEnv):
    """E0509 Direct 환경 (단계별 상태 머신)"""

    cfg: E0509DirectEnvCfg

    def __init__(self, cfg: E0509DirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스
        self._arm_joint_ids, _ = self.robot.find_joints(["joint_[1-6]"])
        self._gripper_joint_ids, _ = self.robot.find_joints(["gripper_rh_.*"])

        # 액션 스케일
        self.action_scale = self.cfg.action_scale

        # 관절 한계
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :6, 0]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :6, 1]

        # 상태 머신: 각 환경의 현재 단계
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 이전 거리 (progress 보상용)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

        # 성공 카운터
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 단계별 통계
        self.phase_counts = torch.zeros(3, device=self.device, dtype=torch.long)

        # === 데이터 수집용 버퍼 ===
        # 각 환경의 에피소드 시작 시 펜 상태 저장
        self.episode_pen_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.episode_pen_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 수집된 데이터 리스트
        self.collected_data = []
        self.total_episodes = 0
        self.successful_episodes = 0

    def _setup_scene(self):
        """씬 구성"""
        # 로봇 생성
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 펜 생성
        self.pen = RigidObject(self.cfg.pen_cfg)
        self.scene.rigid_objects["pen"] = self.pen

        # 지형 (바닥)
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
        )
        terrain_cfg.func("/World/ground", terrain_cfg)

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 환경 복제
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """액션 전처리"""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """액션 적용 (delta position control)"""
        # 현재 관절 위치
        current_pos = self.robot.data.joint_pos[:, :6]

        # 델타 적용
        target_pos = current_pos + self.actions * self.action_scale

        # 관절 한계 클램핑
        target_pos = torch.clamp(
            target_pos,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # 전체 관절 타겟 (그리퍼는 열린 상태 고정)
        full_target = torch.zeros(self.num_envs, 10, device=self.device)
        full_target[:, :6] = target_pos
        full_target[:, 6:] = 0.0  # 그리퍼 열림

        self.robot.set_joint_position_target(full_target)

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

        # 관찰 결합
        obs = torch.cat([
            joint_pos,       # 6
            joint_vel,       # 6
            grasp_pos_local, # 3
            cap_pos_local,   # 3
            rel_pos,         # 3
            gripper_z,       # 3
            pen_z,           # 3
        ], dim=-1)           # 총 27

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """단계별 보상 계산"""
        # 현재 상태 계산
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        distance = torch.norm(grasp_pos - cap_pos, dim=-1)
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)

        # 보상 초기화
        rewards = torch.zeros(self.num_envs, device=self.device)

        # === 단계별 보상 ===

        # APPROACH 단계 (거리 줄이기에 집중)
        approach_mask = (self.phase == PHASE_APPROACH)
        if approach_mask.any():
            # 거리 페널티
            rewards[approach_mask] += self.cfg.rew_scale_approach_dist * distance[approach_mask]
            # 거리 감소 보상 (progress)
            progress = self.prev_distance[approach_mask] - distance[approach_mask]
            rewards[approach_mask] += self.cfg.rew_scale_approach_progress * torch.clamp(progress, min=0)

        # ALIGN 단계 (정렬에 집중, 거리 유지)
        align_mask = (self.phase == PHASE_ALIGN)
        if align_mask.any():
            # 거리 유지 페널티
            rewards[align_mask] += self.cfg.rew_scale_align_dist * distance[align_mask]
            # 정렬 보상 (dot이 -1에 가까울수록 좋음)
            align_reward = (-dot_product[align_mask] - 0.5) / 0.5  # -0.5 ~ -1 → 0 ~ 1
            rewards[align_mask] += self.cfg.rew_scale_align_dot * torch.clamp(align_reward, min=0)

        # GRASP 단계 (최종 접근 + 정렬 유지)
        grasp_mask = (self.phase == PHASE_GRASP)
        if grasp_mask.any():
            # 거리 페널티 (더 강하게)
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_dist * distance[grasp_mask]
            # 정렬 유지 보상
            rewards[grasp_mask] += self.cfg.rew_scale_grasp_dot * (-dot_product[grasp_mask])

        # === 단계 전환 체크 및 보상 ===

        # APPROACH → ALIGN
        transition_to_align = approach_mask & (distance < APPROACH_TO_ALIGN_DIST)
        if transition_to_align.any():
            self.phase[transition_to_align] = PHASE_ALIGN
            rewards[transition_to_align] += self.cfg.rew_scale_phase_transition

        # ALIGN → GRASP
        transition_to_grasp = align_mask & (dot_product < ALIGN_TO_GRASP_DOT)
        if transition_to_grasp.any():
            self.phase[transition_to_grasp] = PHASE_GRASP
            rewards[transition_to_grasp] += self.cfg.rew_scale_phase_transition

        # === 성공 보상 ===
        success = (distance < SUCCESS_DIST) & (dot_product < SUCCESS_DOT)
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # === 액션 페널티 ===
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        # === 반대 방향 페널티 (dot > 0이면 페널티) ===
        wrong_direction_mask = dot_product > 0
        if wrong_direction_mask.any():
            # dot이 양수일 때: dot 값에 비례한 페널티
            rewards[wrong_direction_mask] += self.cfg.rew_scale_wrong_direction * dot_product[wrong_direction_mask]

        # 이전 거리 업데이트
        self.prev_distance = distance.clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        # 성공 종료
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        distance = torch.norm(grasp_pos - cap_pos, dim=-1)
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)

        success = (distance < SUCCESS_DIST) & (dot_product < SUCCESS_DOT)

        # 타임아웃
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """환경 리셋"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        # === 데이터 수집 (리셋 전) ===
        if self.cfg.collect_data and len(env_ids) > 0:
            self._collect_episode_data(env_ids_tensor)

        super()._reset_idx(env_ids)

        # 단계 통계 업데이트 (리셋 전)
        for phase in range(3):
            self.phase_counts[phase] += (self.phase[env_ids_tensor] == phase).sum()

        # 로봇 리셋 (랜덤 오프셋)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, :6] += sample_uniform(
            -0.1, 0.1,
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
        pen_state[:, 0] += self.scene.env_origins[env_ids, 0]
        pen_state[:, 1] += self.scene.env_origins[env_ids, 1]
        pen_state[:, 2] += self.scene.env_origins[env_ids, 2]

        # 랜덤 위치 오프셋
        pen_pos_x = sample_uniform(
            self.cfg.pen_pos_range["x"][0],
            self.cfg.pen_pos_range["x"][1],
            (len(env_ids),),
            device=self.device,
        )
        pen_pos_y = sample_uniform(
            self.cfg.pen_pos_range["y"][0],
            self.cfg.pen_pos_range["y"][1],
            (len(env_ids),),
            device=self.device,
        )
        pen_pos_z = sample_uniform(
            self.cfg.pen_pos_range["z"][0],
            self.cfg.pen_pos_range["z"][1],
            (len(env_ids),),
            device=self.device,
        )
        pen_state[:, 0] = self.scene.env_origins[env_ids, 0] + pen_pos_x
        pen_state[:, 1] = self.scene.env_origins[env_ids, 1] + pen_pos_y
        pen_state[:, 2] = self.scene.env_origins[env_ids, 2] + pen_pos_z

        # === 펜 방향 랜덤화 (오일러 → 쿼터니언) ===
        roll = sample_uniform(
            self.cfg.pen_rot_range["roll"][0],
            self.cfg.pen_rot_range["roll"][1],
            (len(env_ids),),
            device=self.device,
        )
        pitch = sample_uniform(
            self.cfg.pen_rot_range["pitch"][0],
            self.cfg.pen_rot_range["pitch"][1],
            (len(env_ids),),
            device=self.device,
        )
        yaw = sample_uniform(
            self.cfg.pen_rot_range["yaw"][0],
            self.cfg.pen_rot_range["yaw"][1],
            (len(env_ids),),
            device=self.device,
        )
        # 오일러(roll, pitch, yaw) → 쿼터니언(w, x, y, z)
        pen_quat = self._euler_to_quat(roll, pitch, yaw)
        pen_state[:, 3:7] = pen_quat

        self.pen.write_root_pose_to_sim(pen_state[:, :7], env_ids)
        self.pen.write_root_velocity_to_sim(pen_state[:, 7:], env_ids)

        # === 새 에피소드 펜 상태 저장 ===
        self.episode_pen_pos[env_ids_tensor] = pen_state[:, :3] - self.scene.env_origins[env_ids]
        self.episode_pen_rot[env_ids_tensor] = pen_quat
        self.episode_success[env_ids_tensor] = False

        # 단계 리셋 (APPROACH부터 시작)
        self.phase[env_ids] = PHASE_APPROACH

        # 이전 거리 초기화
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        self.prev_distance[env_ids] = torch.norm(
            grasp_pos[env_ids] - cap_pos[env_ids], dim=-1
        )

    # =============================================================================
    # 헬퍼 함수들
    # =============================================================================
    def _get_grasp_point(self) -> torch.Tensor:
        """그리퍼 잡기 포인트 계산 (손가락 중심점)"""
        # 그리퍼 손가락 4개 위치
        l1 = self.robot.data.body_pos_w[:, 7, :]   # gripper_rh_l1
        r1 = self.robot.data.body_pos_w[:, 8, :]   # gripper_rh_r1
        l2 = self.robot.data.body_pos_w[:, 9, :]   # gripper_rh_l2
        r2 = self.robot.data.body_pos_w[:, 10, :]  # gripper_rh_r2

        # 손가락 기저부 중심
        base_center = (l1 + r1) / 2.0
        # 손가락 끝 중심
        tip_center = (l2 + r2) / 2.0

        # 손가락 방향
        finger_dir = tip_center - base_center
        finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

        # 잡기 포인트: 기저부에서 손가락 방향으로 2cm
        return base_center + finger_dir * 0.02

    def _get_pen_cap_pos(self) -> torch.Tensor:
        """펜 캡 위치 (펜 +Z 방향)"""
        pen_pos = self.pen.data.root_pos_w
        pen_quat = self.pen.data.root_quat_w

        # 쿼터니언에서 Z축 방향 추출
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

    def get_phase_stats(self) -> dict:
        """단계별 통계 반환"""
        current_phases = torch.bincount(self.phase, minlength=3)
        return {
            "approach": current_phases[0].item(),
            "align": current_phases[1].item(),
            "grasp": current_phases[2].item(),
            "total_success": self.success_count.sum().item(),
        }

    # =============================================================================
    # 데이터 수집 및 유틸리티 함수
    # =============================================================================
    def _euler_to_quat(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """오일러 각도(roll, pitch, yaw)를 쿼터니언(w, x, y, z)으로 변환"""
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

    def _collect_episode_data(self, env_ids: torch.Tensor):
        """에피소드 종료 시 데이터 수집"""
        # 성공 여부 확인 (현재 성공 상태)
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()
        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()

        distance = torch.norm(grasp_pos - cap_pos, dim=-1)
        dot_product = torch.sum(gripper_z * pen_z, dim=-1)
        current_success = (distance < SUCCESS_DIST) & (dot_product < SUCCESS_DOT)

        for env_id in env_ids:
            env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id

            # 데이터 저장
            data_point = {
                "pen_pos": self.episode_pen_pos[env_id_int].cpu().numpy().tolist(),
                "pen_rot": self.episode_pen_rot[env_id_int].cpu().numpy().tolist(),
                "success": bool(current_success[env_id_int].item()),
            }
            self.collected_data.append(data_point)

            # 통계 업데이트
            self.total_episodes += 1
            if data_point["success"]:
                self.successful_episodes += 1

    def save_collected_data(self, filepath: str = None):
        """수집된 데이터를 JSON 파일로 저장"""
        import json

        if filepath is None:
            os.makedirs(self.cfg.data_save_path, exist_ok=True)
            filepath = os.path.join(self.cfg.data_save_path, "feasibility_data.json")

        data = {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.successful_episodes / max(1, self.total_episodes),
            "pen_pos_range": self.cfg.pen_pos_range,
            "pen_rot_range": self.cfg.pen_rot_range,
            "episodes": self.collected_data,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Data] Saved {len(self.collected_data)} episodes to {filepath}")
        print(f"[Data] Success rate: {data['success_rate']*100:.2f}%")

        return filepath

    def get_data_stats(self) -> dict:
        """수집된 데이터 통계 반환"""
        success_rate = self.successful_episodes / max(1, self.total_episodes)
        return {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": success_rate,
            "collected_count": len(self.collected_data),
        }

    def clear_collected_data(self):
        """수집된 데이터 초기화"""
        self.collected_data = []
        self.total_episodes = 0
        self.successful_episodes = 0


@configclass
class E0509DirectEnvCfg_PLAY(E0509DirectEnvCfg):
    """테스트용 설정"""

    def __post_init__(self):
        # 작은 환경 수
        self.scene.num_envs = 50
