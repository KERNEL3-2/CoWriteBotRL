"""
Pen Grasp Environment for Isaac Lab
Robot: Doosan E0509 + RH-P12-RN-A Gripper
Task: Grasp pen cap and hold in writing pose
"""
from __future__ import annotations

import os
import torch

# Get path to robot USD file (relative to this file)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "first_control.usd")

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
PEN_LENGTH = 0.1207    # 120.7mm
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

    # Pen object (floating in air, like held by human)
    pen: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pen",
        spawn=sim_utils.CylinderCfg(
            radius=PEN_DIAMETER / 2,
            height=PEN_LENGTH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # Floating in air (held by human)
                kinematic_enabled=True,  # Fixed in place
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PEN_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.3)),
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


def ee_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position relative to env origin."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, -1, :]  # Last body (gripper)
    return ee_pos_w - env.scene.env_origins


def pen_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pen position relative to env origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def relative_ee_pen_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative position from end-effector to pen center."""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]
    ee_pos = robot.data.body_pos_w[:, -1, :]
    pen_pos = pen.data.root_pos_w
    return pen_pos - ee_pos


def pen_orientation_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pen orientation as quaternion (to distinguish cap vs tip)."""
    pen: RigidObject = env.scene["pen"]
    return pen.data.root_quat_w  # (num_envs, 4)


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
    """Relative position from end-effector to pen cap (point b)."""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    ee_pos = robot.data.body_pos_w[:, -1, :]
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

    return cap_pos - ee_pos


##
# Reward Terms
##
def distance_ee_cap_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for moving end-effector close to pen cap (point b)."""
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    ee_pos = robot.data.body_pos_w[:, -1, :]
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

    distance = torch.norm(ee_pos - cap_pos, dim=-1)
    return 1.0 / (1.0 + distance * 10.0)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for large action changes."""
    return -torch.sum(torch.square(env.action_manager.action), dim=-1) * 0.001


##
# Termination Terms
##
def pen_dropped_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate if pen falls below ground."""
    pen: RigidObject = env.scene["pen"]
    pen_z = pen.data.root_pos_w[:, 2]
    return pen_z < 0.01


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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward specifications for the MDP."""
    distance_to_cap = RewTerm(func=distance_ee_cap_reward, weight=1.0)  # Reward for approaching cap (point b)
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
                "roll": (-0.5, 0.5),   # Random tilt ~±30 degrees
                "pitch": (-0.5, 0.5),  # Random tilt ~±30 degrees
                "yaw": (-3.14, 3.14),  # Full rotation around z
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
