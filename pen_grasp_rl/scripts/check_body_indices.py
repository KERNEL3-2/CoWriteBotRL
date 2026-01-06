"""
로봇 바디 인덱스 확인 스크립트

사용법:
    python check_body_indices.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from isaaclab.app import AppLauncher

# headless 모드로 실행
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from isaaclab.assets import Articulation, ArticulationCfg
import isaaclab.sim as sim_utils


def main():
    # USD 경로
    ROBOT_USD_PATH = os.path.join(SCRIPT_DIR, "..", "models", "first_control.usd")

    # 시뮬레이션 컨텍스트
    sim_cfg = sim_utils.SimulationCfg(dt=1/60)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 2.0], [0.0, 0.0, 0.5])

    # 로봇 설정
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # 로봇 생성
    robot = Articulation(robot_cfg)

    # 시뮬레이션 리셋
    sim.reset()

    # 바디 정보 출력
    print("\n" + "="*60)
    print("로봇 바디 정보")
    print("="*60)

    body_names = robot.body_names
    print(f"\n총 바디 수: {len(body_names)}")
    print("\n바디 목록:")
    print("-"*40)
    for idx, name in enumerate(body_names):
        print(f"  [{idx:2d}] {name}")

    # 관절 정보
    print("\n" + "="*60)
    print("로봇 관절 정보")
    print("="*60)

    joint_names = robot.joint_names
    print(f"\n총 관절 수: {len(joint_names)}")
    print("\n관절 목록:")
    print("-"*40)
    for idx, name in enumerate(joint_names):
        print(f"  [{idx:2d}] {name}")

    # grasp_point 바디 찾기
    print("\n" + "="*60)
    print("grasp_point 바디 확인")
    print("="*60)

    try:
        grasp_body_ids = robot.find_bodies("grasp_point")
        print(f"\ngrasp_point 바디 ID: {grasp_body_ids}")
    except Exception as e:
        print(f"\ngrasp_point 바디를 찾을 수 없음: {e}")

    # 손가락 관련 바디 찾기
    print("\n" + "="*60)
    print("손가락 바디 확인")
    print("="*60)

    finger_keywords = ["finger", "gripper", "L_", "R_", "link"]
    for keyword in finger_keywords:
        matching = [(idx, name) for idx, name in enumerate(body_names) if keyword.lower() in name.lower()]
        if matching:
            print(f"\n'{keyword}' 포함 바디:")
            for idx, name in matching:
                print(f"  [{idx:2d}] {name}")

    # 시뮬레이션 스텝
    sim.step()
    robot.update(sim.cfg.dt)

    # 바디 위치 출력
    print("\n" + "="*60)
    print("바디 위치 (월드 좌표)")
    print("="*60)

    body_pos = robot.data.body_pos_w[0]  # 첫 번째 환경
    for idx, name in enumerate(body_names):
        pos = body_pos[idx].cpu().numpy()
        print(f"  [{idx:2d}] {name:20s}: ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f})")

    print("\n" + "="*60)
    print("완료")
    print("="*60)

    # 종료
    simulation_app.close()


if __name__ == "__main__":
    main()
