"""
E0509 Domain Randomization 환경 (Sim2Real Ready)

=== Domain Randomization 항목 ===
1. 관측 노이즈 (joint pos/vel, EE pose)
2. 액션 노이즈 및 지연
3. 로봇 동역학 랜덤화 (stiffness, damping)
4. 펜 위치/기울기 랜덤화 (확장)
5. 초기 로봇 자세 랜덤화

=== Sim2Real 전략 ===
- 3DoF 위치 제어 (IK가 자세 자동 계산)
- Domain Randomization으로 실제 로봇 분포 커버
- 관측/액션 노이즈로 센서/액추에이터 불확실성 모델링
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

# 성공 조건
SUCCESS_DIST_TO_CAP = 0.03       # 캡까지 거리 < 3cm
SUCCESS_PERP_DIST = 0.01         # 펜 축에서 벗어난 거리 < 1cm
SUCCESS_HOLD_STEPS = 30          # 30 스텝 유지하면 성공


# =============================================================================
# Domain Randomization 설정
# =============================================================================
@configclass
class DomainRandomizationCfg:
    """Domain Randomization 파라미터"""

    # === 관측 노이즈 ===
    obs_noise_joint_pos: float = 0.01      # 관절 위치 노이즈 (rad)
    obs_noise_joint_vel: float = 0.05      # 관절 속도 노이즈 (rad/s)
    obs_noise_ee_pos: float = 0.005        # EE 위치 노이즈 (m)
    obs_noise_pen_pos: float = 0.01        # 펜 위치 노이즈 (m)

    # === 액션 노이즈 ===
    action_noise: float = 0.02             # 액션 노이즈 스케일
    action_delay_prob: float = 0.1         # 액션 지연 확률
    action_delay_steps: int = 1            # 지연 스텝 수

    # === 로봇 동역학 랜덤화 ===
    stiffness_range: tuple = (0.8, 1.2)    # stiffness 스케일 범위
    damping_range: tuple = (0.8, 1.2)      # damping 스케일 범위

    # === 펜 위치 랜덤화 (확장된 범위) ===
    pen_pos_x_range: tuple = (0.25, 0.55)  # X 범위 (더 넓음)
    pen_pos_y_range: tuple = (-0.20, 0.20) # Y 범위 (더 넓음)
    pen_pos_z_range: tuple = (0.15, 0.30)  # Z 범위 (캡 높이)
    pen_tilt_range: tuple = (0.0, 0.52)    # 기울기 0~30도

    # === 초기 로봇 자세 랜덤화 ===
    init_joint_noise: float = 0.1          # 초기 관절 노이즈 (rad)

    # === 활성화 플래그 ===
    enable_obs_noise: bool = True
    enable_action_noise: bool = True
    enable_dynamics_randomization: bool = True


# =============================================================================
# 환경 설정
# =============================================================================
@configclass
class E0509DREnvCfg(DirectRLEnvCfg):
    """E0509 Domain Randomization 환경 설정"""

    # 환경 기본 설정
    decimation = 2
    episode_length_s = 15.0
    action_scale = 0.03
    action_space = 3          # [Δx, Δy, Δz]
    observation_space = 27    # 관찰 차원
    state_space = 0

    # Domain Randomization 설정
    dr: DomainRandomizationCfg = DomainRandomizationCfg()

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

    # IK 컨트롤러 설정
    ik_method = "dls"
    ik_lambda = 0.05
    ee_body_name = "link_6"
    ee_offset_pos = [0.0, 0.0, 0.15]

    # ==========================================================================
    # 보상 스케일
    # ==========================================================================
    rew_scale_dist_to_cap = -15.0
    rew_scale_dist_exp = 10.0
    rew_scale_perp_dist = -8.0
    rew_scale_perp_exp = 5.0
    rew_scale_approach_progress = 3.0
    rew_scale_alignment = 5.0
    rew_scale_ready_bonus = 10.0
    rew_scale_success = 100.0
    rew_scale_action = -0.01
    rew_scale_collision = -10.0


class E0509DREnv(DirectRLEnv):
    """E0509 Domain Randomization 환경"""

    cfg: E0509DREnvCfg

    def __init__(self, cfg: E0509DREnvCfg, render_mode: str | None = None, **kwargs):
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

        print(f"[E0509DREnv] EE body: {body_names[0]} (idx={self._ee_body_idx})")
        print(f"[E0509DREnv] Arm joints: {self._arm_joint_names}")
        print(f"[E0509DREnv] Domain Randomization 활성화:")
        print(f"  - Obs noise: {self.cfg.dr.enable_obs_noise}")
        print(f"  - Action noise: {self.cfg.dr.enable_action_noise}")
        print(f"  - Dynamics rand: {self.cfg.dr.enable_dynamics_randomization}")

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

        # 상태 변수
        self.prev_distance_to_cap = torch.zeros(self.num_envs, device=self.device)
        self.success_hold_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Domain Randomization 버퍼
        self._prev_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._action_delay_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 환경별 동역학 스케일 (stiffness, damping)
        self._stiffness_scale = torch.ones(self.num_envs, device=self.device)
        self._damping_scale = torch.ones(self.num_envs, device=self.device)

        # 로깅
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
        """액션 전처리 + Domain Randomization"""
        self.actions = actions.clone()

        # === 액션 노이즈 ===
        if self.cfg.dr.enable_action_noise:
            noise = torch.randn_like(self.actions) * self.cfg.dr.action_noise
            self.actions = self.actions + noise

            # 액션 지연 (일부 환경에서 이전 액션 사용)
            if self.cfg.dr.action_delay_prob > 0:
                delay_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.dr.action_delay_prob
                self.actions[delay_mask] = self._prev_actions[delay_mask]

        self._prev_actions = self.actions.clone()

    def _compute_auto_orientation(self) -> torch.Tensor:
        """펜 축 기반 자동 자세 계산"""
        pen_z = self._get_pen_z_axis()

        gripper_z = -pen_z

        world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        gripper_x = torch.cross(world_z, gripper_z, dim=-1)
        gripper_x_norm = torch.norm(gripper_x, dim=-1, keepdim=True)

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
        target_quat = math_utils.quat_from_matrix(rot_matrix)

        return target_quat

    def _apply_action(self) -> None:
        """액션 적용 (3DoF 위치 + 자동 자세)"""
        ee_pos_curr, ee_quat_curr = self._compute_ee_pose()

        pos_delta = self.actions * self.action_scale
        target_quat = self._compute_auto_orientation()

        quat_curr_inv = math_utils.quat_inv(ee_quat_curr)
        quat_delta = math_utils.quat_mul(target_quat, quat_curr_inv)
        rot_delta_axis_angle = math_utils.axis_angle_from_quat(quat_delta)

        rot_scale = 0.3
        rot_delta_scaled = rot_delta_axis_angle * rot_scale

        ik_command = torch.cat([pos_delta, rot_delta_scaled], dim=-1)
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

    def _add_observation_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """관측값에 노이즈 추가"""
        if not self.cfg.dr.enable_obs_noise:
            return obs

        noisy_obs = obs.clone()

        # joint_pos (0-5): 관절 위치 노이즈
        noisy_obs[:, 0:6] += torch.randn(self.num_envs, 6, device=self.device) * self.cfg.dr.obs_noise_joint_pos

        # joint_vel (6-11): 관절 속도 노이즈
        noisy_obs[:, 6:12] += torch.randn(self.num_envs, 6, device=self.device) * self.cfg.dr.obs_noise_joint_vel

        # grasp_pos_local (12-14): EE 위치 노이즈
        noisy_obs[:, 12:15] += torch.randn(self.num_envs, 3, device=self.device) * self.cfg.dr.obs_noise_ee_pos

        # cap_pos_local (15-17): 펜 캡 위치 노이즈
        noisy_obs[:, 15:18] += torch.randn(self.num_envs, 3, device=self.device) * self.cfg.dr.obs_noise_pen_pos

        return noisy_obs

    def _get_observations(self) -> dict:
        """관측값 계산 + Domain Randomization 노이즈"""
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

        obs = torch.cat([
            joint_pos,                           # 6
            joint_vel,                           # 6
            grasp_pos_local,                     # 3
            cap_pos_local,                       # 3
            rel_pos,                             # 3
            pen_z,                               # 3
            perpendicular_dist.unsqueeze(-1),   # 1
            distance_to_cap.unsqueeze(-1),       # 1
            torch.zeros(self.num_envs, 1, device=self.device),  # phase (always 0)
        ], dim=-1)

        # === Domain Randomization: 관측 노이즈 ===
        obs = self._add_observation_noise(obs)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """보상 계산"""
        grasp_pos = self._get_grasp_point()
        cap_pos = self._get_pen_cap_pos()

        perpendicular_dist, axis_distance, on_correct_side = self._compute_axis_metrics()
        distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        gripper_z = self._get_gripper_z_axis()
        pen_z = self._get_pen_z_axis()
        dot = torch.sum(gripper_z * pen_z, dim=-1)

        # 거리 보상
        rewards += self.cfg.rew_scale_dist_to_cap * distance_to_cap
        rewards += self.cfg.rew_scale_dist_exp * torch.exp(-distance_to_cap * 15.0)

        # 펜 축 정렬 보상
        rewards += self.cfg.rew_scale_perp_dist * perpendicular_dist
        rewards += self.cfg.rew_scale_perp_exp * torch.exp(-perpendicular_dist * 50.0)

        # 접근 진행 보상
        progress = self.prev_distance_to_cap - distance_to_cap
        rewards += self.cfg.rew_scale_approach_progress * torch.clamp(progress * 50, min=0, max=1)

        # 자세 정렬 보상
        alignment_reward = (-dot - 0.5) * 0.5
        alignment_reward = torch.clamp(alignment_reward, min=0)
        rewards += self.cfg.rew_scale_alignment * alignment_reward

        # 캡 위 보너스
        above_cap_bonus = on_correct_side.float() * 0.5
        rewards += above_cap_bonus

        # 성공 조건 근접 보너스
        near_success = (
            (distance_to_cap < SUCCESS_DIST_TO_CAP * 2) &
            (perpendicular_dist < SUCCESS_PERP_DIST * 2) &
            on_correct_side
        )
        rewards[near_success] += self.cfg.rew_scale_ready_bonus * 0.5

        # 성공 조건
        success_condition = (
            (distance_to_cap < SUCCESS_DIST_TO_CAP) &
            (perpendicular_dist < SUCCESS_PERP_DIST) &
            on_correct_side
        )

        self.success_hold_count[success_condition] += 1
        self.success_hold_count[~success_condition] = 0

        success = self.success_hold_count >= SUCCESS_HOLD_STEPS
        rewards[success] += self.cfg.rew_scale_success
        self.success_count[success] += 1

        # 충돌 페널티
        collision = self._check_pen_collision()
        if collision.any():
            rewards[collision] += self.cfg.rew_scale_collision

        # 캡 지나침 페널티
        passed_cap = ~on_correct_side
        rewards[passed_cap] -= 1.0

        # 액션 페널티
        rewards += self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=-1)

        self.prev_distance_to_cap = distance_to_cap.clone()

        # 로깅
        self._global_step += 1
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/dist_to_cap_mean"] = distance_to_cap.mean().item()
        self.extras["log"]["Metrics/perp_dist_mean"] = perpendicular_dist.mean().item()
        self.extras["log"]["Metrics/success_rate"] = (self.success_hold_count >= SUCCESS_HOLD_STEPS).float().mean().item()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        success = self.success_hold_count >= SUCCESS_HOLD_STEPS
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """환경 리셋 + Domain Randomization"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        super()._reset_idx(env_ids)

        # === 로봇 리셋 + 초기 자세 랜덤화 ===
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, :6] += sample_uniform(
            -self.cfg.dr.init_joint_noise, self.cfg.dr.init_joint_noise,
            (len(env_ids), 6),
            device=self.device,
        )
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # === 펜 리셋 + 위치/기울기 랜덤화 ===
        pen_state = self.pen.data.default_root_state[env_ids].clone()

        # 캡 위치 랜덤화 (확장된 범위)
        cap_pos_x = sample_uniform(
            self.cfg.dr.pen_pos_x_range[0], self.cfg.dr.pen_pos_x_range[1],
            (len(env_ids),), device=self.device
        )
        cap_pos_y = sample_uniform(
            self.cfg.dr.pen_pos_y_range[0], self.cfg.dr.pen_pos_y_range[1],
            (len(env_ids),), device=self.device
        )
        cap_pos_z = sample_uniform(
            self.cfg.dr.pen_pos_z_range[0], self.cfg.dr.pen_pos_z_range[1],
            (len(env_ids),), device=self.device
        )

        pen_center_x = cap_pos_x
        pen_center_y = cap_pos_y
        pen_center_z = cap_pos_z - PEN_LENGTH / 2

        pen_state[:, 0] = self.scene.env_origins[env_ids, 0] + pen_center_x
        pen_state[:, 1] = self.scene.env_origins[env_ids, 1] + pen_center_y
        pen_state[:, 2] = self.scene.env_origins[env_ids, 2] + pen_center_z

        # 펜 기울기 랜덤화 (전체 범위)
        tilt = sample_uniform(
            self.cfg.dr.pen_tilt_range[0], self.cfg.dr.pen_tilt_range[1],
            (len(env_ids),), device=self.device
        )
        azimuth = sample_uniform(0, 2 * 3.14159, (len(env_ids),), device=self.device)
        yaw = sample_uniform(-3.14, 3.14, (len(env_ids),), device=self.device)
        pen_quat = self._cone_angle_to_quat(tilt, azimuth, yaw)
        pen_state[:, 3:7] = pen_quat

        self.pen.write_root_pose_to_sim(pen_state[:, :7], env_ids)
        self.pen.write_root_velocity_to_sim(pen_state[:, 7:], env_ids)

        # === 동역학 랜덤화 ===
        if self.cfg.dr.enable_dynamics_randomization:
            self._stiffness_scale[env_ids] = sample_uniform(
                self.cfg.dr.stiffness_range[0], self.cfg.dr.stiffness_range[1],
                (len(env_ids),), device=self.device
            )
            self._damping_scale[env_ids] = sample_uniform(
                self.cfg.dr.damping_range[0], self.cfg.dr.damping_range[1],
                (len(env_ids),), device=self.device
            )

        # 상태 리셋
        self.success_hold_count[env_ids] = 0
        self._prev_actions[env_ids] = 0

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

    def _cone_angle_to_quat(self, tilt: torch.Tensor, azimuth: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """원뿔 각도 → 쿼터니언"""
        q1_w = torch.cos(tilt * 0.5)
        q1_x = torch.zeros_like(tilt)
        q1_y = torch.sin(tilt * 0.5)
        q1_z = torch.zeros_like(tilt)

        q2_w = torch.cos(azimuth * 0.5)
        q2_x = torch.zeros_like(azimuth)
        q2_y = torch.zeros_like(azimuth)
        q2_z = torch.sin(azimuth * 0.5)

        q3_w = torch.cos(yaw * 0.5)
        q3_x = torch.zeros_like(yaw)
        q3_y = torch.zeros_like(yaw)
        q3_z = torch.sin(yaw * 0.5)

        r1_w = q2_w * q1_w - q2_z * q1_y
        r1_x = q2_w * q1_x + q2_z * q1_z
        r1_y = q2_w * q1_y + q2_z * q1_x
        r1_z = q2_z * q1_w + q2_w * q1_z

        w = q3_w * r1_w - q3_z * r1_z
        x = q3_w * r1_x + q3_z * r1_y
        y = q3_w * r1_y - q3_z * r1_x
        z = q3_w * r1_z + q3_z * r1_w

        return torch.stack([w, x, y, z], dim=-1)


# =============================================================================
# 설정 변형
# =============================================================================
@configclass
class E0509DREnvCfg_PLAY(E0509DREnvCfg):
    """시각화용 설정 (DR 비활성화)"""
    def __post_init__(self):
        self.scene.num_envs = 50
        self.dr.enable_obs_noise = False
        self.dr.enable_action_noise = False
        self.dr.enable_dynamics_randomization = False


@configclass
class E0509DREnvCfg_TRAIN(E0509DREnvCfg):
    """학습용 설정 (DR 전체 활성화)"""
    def __post_init__(self):
        self.scene.num_envs = 4096
        self.dr.enable_obs_noise = True
        self.dr.enable_action_noise = True
        self.dr.enable_dynamics_randomization = True
