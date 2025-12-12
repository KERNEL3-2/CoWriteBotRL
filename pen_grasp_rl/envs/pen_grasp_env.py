"""
Pen Grasp Environment for Isaac Lab
Robot: Doosan E0509 + RH-P12-RN-A Gripper
Task: Grasp pen cap and hold in writing pose
"""
from __future__ import annotations

import os
import torch

# Get path to USD files (relative to this file)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")
PEN_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "pen.usd")

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

##
# Pen specifications
##
PEN_DIAMETER = 0.0198  # 19.8mm
PEN_LENGTH = 0.117     # 117mm (without cap)
PEN_MASS = 0.0163      # 16.3g

##
# Scene Configuration
##
@configclass
class PenGraspSceneCfg(InteractiveSceneCfg):
    """Configuration for the pen grasp scene."""

    # Ground plane
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Robot: Doosan E0509 + Gripper
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=True,
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
                "joint_3": 1.57,  # slightly bent
                "joint_4": 0.0,
                "joint_5": -1.57,
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

    # Pen object (floating in air with collision - can be knocked down by gripper)
    pen: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PEN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # Floating in air (held by human)
                kinematic_enabled=False,  # Enable collision physics
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PEN_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.3),  # Center of workspace (x:0.3~0.7, y:±0.3, z:0.1~0.5)
        ),
    )


##
# Action Term - Joint Position Control with Gripper Mimic
##
class ArmGripperActionTerm(ActionTerm):
    """Action term for arm (6 joints) + gripper (1 command -> 4 joints mimic)."""

    _asset: Articulation

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Action: 6 arm + 1 gripper = 7
        self._raw_actions = torch.zeros(env.num_envs, 7, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 7, device=self.device)
        # Full joint targets: 6 arm + 4 gripper = 10
        self._joint_pos_target = torch.zeros(env.num_envs, 10, device=self.device)
        # Scale for actions
        self.arm_scale = 0.1  # Scale down arm actions
        self.gripper_scale = 1.0

    @property
    def action_dim(self) -> int:
        return 7  # 6 arm + 1 gripper

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

    def apply_actions(self):
        # Get current joint positions
        current_pos = self._asset.data.joint_pos

        # Arm actions (delta position)
        arm_delta = self._processed_actions[:, :6] * self.arm_scale
        self._joint_pos_target[:, :6] = current_pos[:, :6] + arm_delta

        # Gripper: single command -> 4 joints (mimic)
        gripper_cmd = self._processed_actions[:, 6:7] * self.gripper_scale
        self._joint_pos_target[:, 6:10] = gripper_cmd.repeat(1, 4)

        # Apply to robot
        self._asset.set_joint_position_target(self._joint_pos_target)


@configclass
class ArmGripperActionTermCfg(ActionTermCfg):
    """Configuration for arm + gripper action term."""
    class_type: type = ArmGripperActionTerm


##
# Observation Terms
##
def joint_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current joint positions (normalized)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def joint_vel_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def get_grasp_point(robot: Articulation) -> torch.Tensor:
    """Get ideal grasp point: (l1+r1)/2 center + 2cm along finger direction.

    This point is stable regardless of gripper open/close state.
    """
    # [7] gripper_rh_p12_rn_l1 = left finger base
    # [8] gripper_rh_p12_rn_r1 = right finger base
    # [9] gripper_rh_p12_rn_l2 = left finger tip
    # [10] gripper_rh_p12_rn_r2 = right finger tip
    l1 = robot.data.body_pos_w[:, 7, :]
    r1 = robot.data.body_pos_w[:, 8, :]
    l2 = robot.data.body_pos_w[:, 9, :]
    r2 = robot.data.body_pos_w[:, 10, :]

    base_center = (l1 + r1) / 2.0
    tip_center = (l2 + r2) / 2.0
    finger_dir = tip_center - base_center
    finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

    return base_center + finger_dir * 0.02  # 2cm along finger direction


def ee_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp point position relative to env origin."""
    asset: Articulation = env.scene[asset_cfg.name]
    grasp_pos_w = get_grasp_point(asset)
    return grasp_pos_w - env.scene.env_origins


def pen_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pen position relative to env origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def relative_ee_pen_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative position from grasp point to pen center."""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]
    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    return pen_pos - grasp_pos


def pen_orientation_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pen orientation as quaternion (to distinguish cap vs tip)."""
    pen: RigidObject = env.scene["pen"]
    return pen.data.root_quat_w  # (num_envs, 4)


def gripper_state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gripper open/close state (mean of 4 gripper joints).

    Returns value between 0 (open) and ~1 (closed).
    """
    robot: Articulation = env.scene["robot"]
    gripper_pos = robot.data.joint_pos[:, 6:10]  # joints 6-9 are gripper
    return gripper_pos.mean(dim=-1, keepdim=True)  # (num_envs, 1)


def pen_cap_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Position of pen cap (point b) relative to env origin.

    Pen is a cylinder with local z-axis along its length.
    Point a (tip) is at -z, point b (cap) is at +z.
    """
    pen: RigidObject = env.scene["pen"]
    pen_pos = pen.data.root_pos_w  # center position
    pen_quat = pen.data.root_quat_w  # orientation

    # Get pen's local z-axis in world frame (cap direction)
    # Quaternion rotation of [0, 0, 1] vector
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]

    # Rotate [0, 0, 1] by quaternion
    cap_dir_x = 2.0 * (qx * qz + qw * qy)
    cap_dir_y = 2.0 * (qy * qz - qw * qx)
    cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)

    # Cap position is center + half_length * cap_direction
    cap_pos = pen_pos + (PEN_LENGTH / 2) * cap_dir

    return cap_pos - env.scene.env_origins


def relative_ee_cap_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative position from grasp point to pen cap (point b)."""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w

    # Get cap direction
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    cap_dir_x = 2.0 * (qx * qz + qw * qy)
    cap_dir_y = 2.0 * (qy * qz - qw * qx)
    cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)

    # Cap position
    cap_pos = pen_pos + (PEN_LENGTH / 2) * cap_dir

    return cap_pos - grasp_pos


##
# Reward Terms
##
def distance_ee_cap_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for moving grasp point close to pen cap (point b)."""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w

    # Get cap direction
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    cap_dir_x = 2.0 * (qx * qz + qw * qy)
    cap_dir_y = 2.0 * (qy * qz - qw * qx)
    cap_dir_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cap_dir = torch.stack([cap_dir_x, cap_dir_y, cap_dir_z], dim=-1)

    # Cap position
    cap_pos = pen_pos + (PEN_LENGTH / 2) * cap_dir

    distance = torch.norm(grasp_pos - cap_pos, dim=-1)
    return 1.0 / (1.0 + distance * 10.0)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for large action changes."""
    return -torch.sum(torch.square(env.action_manager.action), dim=-1) * 0.001


def floor_collision_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for robot links getting too close to the floor.

    Checks link_2~6 and gripper (indices 2-10) for z < 0.05m.
    """
    robot: Articulation = env.scene["robot"]

    # Get z positions of movable links (indices 2-10)
    # [2-6] link_2~6, [7-10] gripper fingers
    link_z = robot.data.body_pos_w[:, 2:11, 2]  # (num_envs, 9)

    # Check if any link is below threshold (5cm)
    floor_threshold = 0.05
    below_floor = (link_z < floor_threshold).any(dim=-1).float()

    # Return negative penalty when any link is too low
    return -below_floor


def z_axis_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning gripper z-axis with pen z-axis.

    Distance-weighted alignment reward:
    - Alignment is rewarded at any distance
    - Closer distance gives higher weight (1/distance scaling)
    - This encourages both approaching AND aligning simultaneously
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    # Get pen z-axis in world frame
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_z_x = 2.0 * (qx * qz + qw * qy)
    pen_z_y = 2.0 * (qy * qz - qw * qx)
    pen_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pen_z_axis = torch.stack([pen_z_x, pen_z_y, pen_z_z], dim=-1)

    # Cap position (pen center + half_length * z_axis)
    cap_pos = pen_pos + (PEN_LENGTH / 2) * pen_z_axis

    # Get grasp point and distance to cap
    grasp_pos = get_grasp_point(robot)
    distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

    # Get gripper z-axis from link_6 orientation
    link6_quat = robot.data.body_quat_w[:, 6, :]  # [6] link_6
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)

    # Dot product: 1.0 = parallel (same direction), -1.0 = opposite (facing each other)
    # For grasping: gripper should face OPPOSITE to pen z-axis (gripper approaches cap from outside)
    # So we reward when dot_product = -1.0 (opposite directions)
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # Distance-weighted alignment reward
    # - alignment_score: 0 (perpendicular or same direction) ~ 1 (opposite, correct grasp direction)
    # - negate dot_product: -1.0 becomes +1.0 (reward), +1.0 becomes -1.0 (no reward)
    alignment_score = torch.clamp(-dot_product, min=0.0)  # 0 ~ 1

    # Distance weight: 1/(distance + 0.05) normalized
    # At 5cm: weight = 1/(0.05+0.05) = 10
    # At 50cm: weight = 1/(0.5+0.05) ≈ 1.8
    distance_weight = 1.0 / (distance_to_cap + 0.05)

    # Normalize to reasonable range (divide by 10 to keep similar scale)
    return alignment_score * distance_weight * 0.1


def pen_displacement_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for displacing the pen from its initial position.

    Since kinematic_enabled=False, the pen can be knocked around.
    This penalty discourages hitting the pen without proper grasp.
    """
    pen: RigidObject = env.scene["pen"]

    # Get pen velocity - if pen is moving, it's being knocked
    pen_vel = pen.data.root_lin_vel_w  # (num_envs, 3)
    vel_magnitude = torch.norm(pen_vel, dim=-1)

    # Penalty proportional to velocity (pen should stay still unless properly grasped)
    return -vel_magnitude * 0.5


def grasp_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for successfully grasping the pen cap.

    Grasp is successful when:
    1. Gripper is close to pen cap (< 3cm)
    2. Gripper is aligned with pen axis (dot < -0.8, opposite direction)
    3. Gripper is closed (gripper joints > 0.5)

    This gives a large bonus for achieving proper grasp pose.
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    # Get pen z-axis and cap position
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_z_x = 2.0 * (qx * qz + qw * qy)
    pen_z_y = 2.0 * (qy * qz - qw * qx)
    pen_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pen_z_axis = torch.stack([pen_z_x, pen_z_y, pen_z_z], dim=-1)
    cap_pos = pen_pos + (PEN_LENGTH / 2) * pen_z_axis

    # Get grasp point and distance to cap
    grasp_pos = get_grasp_point(robot)
    distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

    # Get gripper z-axis
    link6_quat = robot.data.body_quat_w[:, 6, :]
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)

    # Check alignment (opposite direction = correct grasp)
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # Check gripper closure (joints 6-9 are gripper)
    gripper_pos = robot.data.joint_pos[:, 6:10]
    gripper_closed = (gripper_pos > 0.5).all(dim=-1).float()

    # Success conditions
    close_enough = (distance_to_cap < 0.03).float()  # < 3cm
    aligned = (dot_product < -0.8).float()  # Opposite direction (correct grasp)

    # Large reward when all conditions met
    return close_enough * aligned * gripper_closed * 5.0


##
# Termination Terms
##
def pen_dropped_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate if pen is displaced too far from initial position.

    Since kinematic_enabled=False, the pen can be knocked in any direction.
    If pen moves more than 15cm from its initial position, episode ends.

    Initial pen position: (0.5, 0.0, 0.3) with randomization ±(0.2, 0.3, 0.2)
    """
    pen: RigidObject = env.scene["pen"]

    # Get current pen position relative to env origin
    pen_pos = pen.data.root_pos_w - env.scene.env_origins  # (num_envs, 3)

    # Initial position center (before randomization)
    init_pos = torch.tensor([0.5, 0.0, 0.3], device=pen_pos.device)

    # Calculate displacement from initial center
    displacement = torch.norm(pen_pos - init_pos, dim=-1)

    # Terminate if displaced more than 15cm
    # (considering randomization range, this gives some margin)
    return displacement > 0.15


##
# Configuration Classes
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_gripper = ArmGripperActionTermCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=joint_pos_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=joint_vel_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        ee_pos = ObsTerm(func=ee_pos_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        pen_pos = ObsTerm(func=pen_pos_obs, params={"asset_cfg": SceneEntityCfg("pen")})
        pen_orientation = ObsTerm(func=pen_orientation_obs)  # (4,) quaternion
        relative_ee_pen = ObsTerm(func=relative_ee_pen_obs)
        relative_ee_cap = ObsTerm(func=relative_ee_cap_obs)  # distance to cap (point b)
        gripper_state = ObsTerm(func=gripper_state_obs)  # (1,) gripper open/close

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward specifications for the MDP."""
    distance_to_cap = RewTerm(func=distance_ee_cap_reward, weight=1.0)  # Reward for approaching cap (point b)
    z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=0.5)  # Reward for aligning gripper z-axis with pen (opposite direction)
    floor_collision = RewTerm(func=floor_collision_penalty, weight=1.0)  # Penalty for links touching floor
    pen_displacement = RewTerm(func=pen_displacement_penalty, weight=1.0)  # Penalty for knocking pen
    grasp_success = RewTerm(func=grasp_success_reward, weight=2.0)  # Large reward for successful grasp
    action_rate = RewTerm(func=action_rate_penalty, weight=0.1)


@configclass
class TerminationsCfg:
    """Termination specifications for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pen_dropped = DoneTerm(func=pen_dropped_termination)


@configclass
class EventsCfg:
    """Event specifications for the MDP."""
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_pen = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),      # 0.3~0.7m from base (base pos + 0.5 ± 0.2)
                "y": (-0.3, 0.3),      # ±30cm left/right
                "z": (-0.2, 0.2),      # 0.1~0.5m height (base z=0.3 ± 0.2)
                "roll": (-3.14, 3.14),   # Full rotation - pen can be upside down
                "pitch": (-3.14, 3.14),  # Full rotation - cap can point any direction
                "yaw": (-3.14, 3.14),    # Full rotation around z
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )


##
# Environment Configuration
##
@configclass
class PenGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pen grasp environment."""

    # Scene settings
    scene: PenGraspSceneCfg = PenGraspSceneCfg(num_envs=64, env_spacing=2.0)

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0  # 60Hz simulation
        self.episode_length_s = 10.0  # 10 seconds per episode


##
# Environment Class
##
class PenGraspEnv(ManagerBasedRLEnv):
    """Pen grasping environment."""
    cfg: PenGraspEnvCfg
