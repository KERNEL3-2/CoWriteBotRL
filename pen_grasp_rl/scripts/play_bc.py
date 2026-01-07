#!/usr/bin/env python3
"""
BC (Behavioral Cloning) 모델 테스트 스크립트

학습된 BC 모델을 Isaac Lab 환경에서 테스트합니다.
BC 모델은 joint position을 직접 출력하므로, position control로 테스트합니다.

사용법:
    python play_bc.py --checkpoint /path/to/best_model.pth

    # 환경 개수 조절
    python play_bc.py --checkpoint /path/to/best_model.pth --num_envs 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# 명령행 인자
parser = argparse.ArgumentParser(description="BC 모델 테스트")
parser.add_argument("--num_envs", type=int, default=50, help="병렬 환경 개수")
parser.add_argument("--checkpoint", type=str, required=True, help="BC 모델 체크포인트")
parser.add_argument("--max_steps", type=int, default=5000, help="최대 스텝 수")
parser.add_argument("--episode_length", type=int, default=100, help="에피소드 최대 길이")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import sample_uniform

# 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(SCRIPT_DIR, "..", "models", "first_control.usd")
PEN_USD_PATH = os.path.join(SCRIPT_DIR, "..", "models", "pen.usd")

PEN_LENGTH = 0.1207
SUCCESS_DIST = 0.03
SUCCESS_PERP = 0.01
SUCCESS_HOLD_STEPS = 30


# =============================================================================
# BC Policy 네트워크 (train_bc.py와 동일)
# =============================================================================
class BCPolicy(nn.Module):
    def __init__(self, obs_dim=25, action_dim=6, hidden_dims=[512, 512, 256]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


# =============================================================================
# 메인
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 로드
    print(f"\n체크포인트 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = BCPolicy().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"모델 로드 완료 (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('valid_loss', '?'):.6f})")

    # ==========================================================================
    # 시뮬레이션 설정
    # ==========================================================================
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

    # Ground
    spawn_ground_plane("/World/ground", GroundPlaneCfg())

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 환경 원점 설정
    num_envs = args.num_envs
    env_spacing = 2.5
    envs_per_row = int(np.ceil(np.sqrt(num_envs)))

    env_origins = torch.zeros(num_envs, 3, device=device)
    for i in range(num_envs):
        row = i // envs_per_row
        col = i % envs_per_row
        env_origins[i, 0] = col * env_spacing
        env_origins[i, 1] = row * env_spacing

    # ==========================================================================
    # 로봇 생성
    # ==========================================================================
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
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

    # 환경별 로봇 생성
    robots = []
    for i in range(num_envs):
        cfg = robot_cfg.copy()
        cfg.prim_path = f"/World/envs/env_{i}/Robot"
        cfg.init_state.pos = tuple(env_origins[i].cpu().numpy())
        robot = Articulation(cfg)
        robots.append(robot)

    # ==========================================================================
    # 펜 생성
    # ==========================================================================
    pen_cfg = RigidObjectCfg(
        prim_path="/World/Pen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PEN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
    )

    pens = []
    for i in range(num_envs):
        cfg = pen_cfg.copy()
        cfg.prim_path = f"/World/envs/env_{i}/Pen"
        origin = env_origins[i].cpu().numpy()
        cfg.init_state.pos = (origin[0] + 0.4, origin[1], origin[2] + 0.2)
        pen = RigidObject(cfg)
        pens.append(pen)

    # 시뮬레이션 리셋
    sim.reset()
    print(f"\n환경 생성 완료: {num_envs}개")

    # 관절 한계
    joint_lower = robots[0].data.soft_joint_pos_limits[0, :6, 0].clone()
    joint_upper = robots[0].data.soft_joint_pos_limits[0, :6, 1].clone()

    # ==========================================================================
    # 상태 변수
    # ==========================================================================
    episode_step = torch.zeros(num_envs, device=device, dtype=torch.long)
    success_hold_count = torch.zeros(num_envs, device=device, dtype=torch.long)
    total_success = 0
    total_episodes = 0

    # ==========================================================================
    # 테스트 루프
    # ==========================================================================
    print("\n" + "=" * 60)
    print("BC 모델 테스트 시작")
    print("=" * 60)

    step = 0

    while simulation_app.is_running() and step < args.max_steps:
        # 각 환경에서 데이터 수집 및 action 실행
        for i in range(num_envs):
            robot = robots[i]
            pen = pens[i]

            # Observation 구성 (25차원)
            joint_pos = robot.data.joint_pos[0, :6]  # (6,)
            joint_vel = robot.data.joint_vel[0, :6]  # (6,)

            ee_pos_w = robot.data.body_pos_w[0, 6]  # link_6
            ee_quat_w = robot.data.body_quat_w[0, 6]  # wxyz

            ee_pos_local = ee_pos_w - env_origins[i]
            ee_quat_xyzw = torch.cat([ee_quat_w[1:4], ee_quat_w[0:1]])

            pen_pos_w = pen.data.root_pos_w[0]
            pen_pos_local = pen_pos_w - env_origins[i]

            pen_quat = pen.data.root_quat_w[0]
            qw, qx, qy, qz = pen_quat[0], pen_quat[1], pen_quat[2], pen_quat[3]
            pen_axis = torch.stack([
                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy)
            ])

            # BC 입력 (25차원)
            bc_obs = torch.cat([
                joint_pos, joint_vel, ee_pos_local, ee_quat_xyzw, pen_pos_local, pen_axis
            ]).unsqueeze(0)

            # BC 모델로 action 예측
            with torch.no_grad():
                target_joint_pos = model(bc_obs).squeeze(0)

            # 관절 한계 클리핑
            target_joint_pos = torch.clamp(target_joint_pos, joint_lower, joint_upper)

            # 그리퍼는 열린 상태 유지
            full_target = torch.zeros(10, device=device)
            full_target[:6] = target_joint_pos
            full_target[6:] = 0.0  # gripper open

            robot.set_joint_position_target(full_target.unsqueeze(0))

            # 성공 조건 체크
            grasp_pos = robot.data.body_pos_w[0, 6]  # 간단히 EE 위치 사용
            cap_pos = pen_pos_w + (PEN_LENGTH / 2) * pen_axis

            dist_to_cap = torch.norm(grasp_pos - cap_pos)

            # 펜 축에서 수직 거리
            grasp_to_cap = cap_pos - grasp_pos
            proj = torch.dot(grasp_to_cap, pen_axis)
            perp_dist = torch.norm(grasp_to_cap - proj * pen_axis)

            # 캡 위에 있는지
            on_correct_side = proj < 0

            if dist_to_cap < SUCCESS_DIST and perp_dist < SUCCESS_PERP and on_correct_side:
                success_hold_count[i] += 1
            else:
                success_hold_count[i] = 0

        # 시뮬레이션 스텝
        sim.step()

        # 로봇/펜 상태 업데이트
        for robot in robots:
            robot.update(sim_cfg.dt)
        for pen in pens:
            pen.update(sim_cfg.dt)

        episode_step += 1
        step += 1

        # 에피소드 종료 체크
        success = success_hold_count >= SUCCESS_HOLD_STEPS
        timeout = episode_step >= args.episode_length

        done = success | timeout

        if done.any():
            done_indices = torch.where(done)[0]

            for idx in done_indices:
                total_episodes += 1
                if success[idx]:
                    total_success += 1

                # 환경 리셋
                robot = robots[idx.item()]
                pen = pens[idx.item()]
                origin = env_origins[idx]

                # 로봇 리셋
                default_joint_pos = torch.tensor(
                    [0.0, -0.3, 0.8, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0],
                    device=device
                ).unsqueeze(0)
                robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))

                # 펜 랜덤 위치
                pen_x = origin[0] + 0.3 + torch.rand(1, device=device) * 0.2
                pen_y = origin[1] - 0.15 + torch.rand(1, device=device) * 0.3
                pen_z = origin[2] + 0.15 + torch.rand(1, device=device) * 0.1
                pen_pos = torch.stack([pen_x, pen_y, pen_z], dim=-1).squeeze()
                pen_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # 수직

                pen_state = torch.cat([pen_pos, pen_quat, torch.zeros(6, device=device)]).unsqueeze(0)
                pen.write_root_pose_to_sim(pen_state[:, :7])

                episode_step[idx] = 0
                success_hold_count[idx] = 0

        # 로깅
        if step % 500 == 0:
            rate = total_success / total_episodes if total_episodes > 0 else 0
            print(f"Step {step:5d} | Episodes: {total_episodes} | Success: {total_success} | Rate: {rate:.1%}")

    # 최종 결과
    success_rate = total_success / total_episodes if total_episodes > 0 else 0

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"총 에피소드: {total_episodes}")
    print(f"성공: {total_success}")
    print(f"성공률: {success_rate:.1%}")
    print("=" * 60)

    simulation_app.close()


if __name__ == "__main__":
    main()
