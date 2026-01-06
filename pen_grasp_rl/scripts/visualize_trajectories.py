"""
전문가 궤적 시각화 스크립트

expert_policy.py로 수집한 데이터를 시각화합니다.

사용법:
    python visualize_trajectories.py --data expert_data.pt --output trajectory_plot.png
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_dataset(path: str) -> dict:
    """데이터셋 로드"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    dataset = torch.load(path)
    print(f"데이터셋 로드 완료: {path}")
    print(f"  에피소드 수: {dataset['num_episodes']}")
    print(f"  성공률: {dataset['success_rate']*100:.1f}%")

    return dataset


def extract_positions(observations: list) -> list:
    """관측값에서 위치 정보 추출"""
    # 관측값 인덱스
    GRASP_POS = slice(12, 15)
    CAP_POS = slice(15, 18)

    trajectories = []

    for ep_obs in observations:
        grasp_positions = ep_obs[:, GRASP_POS].numpy()
        cap_position = ep_obs[0, CAP_POS].numpy()  # 캡 위치 (첫 프레임)

        trajectories.append({
            "grasp_path": grasp_positions,
            "cap_pos": cap_position,
        })

    return trajectories


def plot_3d_trajectories(trajectories: list, output_path: str, max_trajectories: int = 20):
    """3D 궤적 시각화"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 색상 맵
    colors = plt.cm.viridis(np.linspace(0, 1, min(len(trajectories), max_trajectories)))

    for i, traj in enumerate(trajectories[:max_trajectories]):
        path = traj["grasp_path"]
        cap = traj["cap_pos"]

        # 궤적
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color=colors[i], alpha=0.7, linewidth=1)

        # 시작점
        ax.scatter(path[0, 0], path[0, 1], path[0, 2],
                   color='green', s=30, marker='o')

        # 끝점
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2],
                   color='red', s=30, marker='x')

        # 캡 위치
        ax.scatter(cap[0], cap[1], cap[2],
                   color='orange', s=50, marker='*')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Expert Trajectories (n={min(len(trajectories), max_trajectories)})')

    # 범례
    ax.scatter([], [], [], color='green', s=30, marker='o', label='Start')
    ax.scatter([], [], [], color='red', s=30, marker='x', label='End')
    ax.scatter([], [], [], color='orange', s=50, marker='*', label='Pen Cap')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D 궤적 저장: {output_path}")
    plt.close()


def plot_2d_top_view(trajectories: list, output_path: str, max_trajectories: int = 50):
    """위에서 본 2D 궤적 (XY 평면)"""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, min(len(trajectories), max_trajectories)))

    for i, traj in enumerate(trajectories[:max_trajectories]):
        path = traj["grasp_path"]
        cap = traj["cap_pos"]

        # 궤적 (XY)
        ax.plot(path[:, 0], path[:, 1],
                color=colors[i], alpha=0.5, linewidth=1)

        # 시작점
        ax.scatter(path[0, 0], path[0, 1],
                   color='green', s=20, marker='o', zorder=5)

        # 끝점
        ax.scatter(path[-1, 0], path[-1, 1],
                   color='red', s=20, marker='x', zorder=5)

        # 캡 위치
        ax.scatter(cap[0], cap[1],
                   color='orange', s=50, marker='*', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Expert Trajectories - Top View (n={min(len(trajectories), max_trajectories)})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 범례
    ax.scatter([], [], color='green', s=20, marker='o', label='Start')
    ax.scatter([], [], color='red', s=20, marker='x', label='End')
    ax.scatter([], [], color='orange', s=50, marker='*', label='Pen Cap')
    ax.legend()

    plt.tight_layout()

    # 파일명 수정
    base, ext = os.path.splitext(output_path)
    top_view_path = f"{base}_top{ext}"
    plt.savefig(top_view_path, dpi=150, bbox_inches='tight')
    print(f"2D 상단뷰 저장: {top_view_path}")
    plt.close()


def plot_statistics(trajectories: list, dataset: dict, output_path: str):
    """통계 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 에피소드 길이 분포
    episode_lengths = [len(traj["grasp_path"]) for traj in trajectories]
    axes[0, 0].hist(episode_lengths, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Episode Length (steps)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Episode Length Distribution (mean={np.mean(episode_lengths):.1f})')
    axes[0, 0].axvline(np.mean(episode_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[0, 0].legend()

    # 2. 최종 거리 분포
    final_distances = []
    for traj in trajectories:
        final_pos = traj["grasp_path"][-1]
        cap_pos = traj["cap_pos"]
        dist = np.linalg.norm(final_pos - cap_pos)
        final_distances.append(dist)

    axes[0, 1].hist(final_distances, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Final Distance to Cap (m)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Final Distance Distribution (mean={np.mean(final_distances):.4f}m)')
    axes[0, 1].axvline(0.03, color='red', linestyle='--', label='Success threshold (3cm)')
    axes[0, 1].legend()

    # 3. 경로 효율성 (직선거리 / 실제 이동거리)
    efficiencies = []
    for traj in trajectories:
        path = traj["grasp_path"]

        # 직선 거리
        straight_dist = np.linalg.norm(path[-1] - path[0])

        # 실제 이동 거리
        actual_dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

        if actual_dist > 0:
            efficiency = straight_dist / actual_dist
            efficiencies.append(efficiency)

    axes[1, 0].hist(efficiencies, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Path Efficiency (straight/actual)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Path Efficiency Distribution (mean={np.mean(efficiencies):.3f})')
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Perfect (1.0)')
    axes[1, 0].legend()

    # 4. 보상 분포 (있는 경우)
    if 'rewards' in dataset and len(dataset['rewards']) > 0:
        total_rewards = [sum(ep_rewards) for ep_rewards in dataset['rewards']]
        axes[1, 1].hist(total_rewards, bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Total Episode Reward')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Reward Distribution (mean={np.mean(total_rewards):.2f})')
    else:
        axes[1, 1].text(0.5, 0.5, 'No reward data', ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Reward Distribution')

    plt.tight_layout()

    # 파일명 수정
    base, ext = os.path.splitext(output_path)
    stats_path = f"{base}_stats{ext}"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"통계 그래프 저장: {stats_path}")
    plt.close()


def print_summary(trajectories: list, dataset: dict):
    """요약 통계 출력"""
    print("\n" + "="*60)
    print("데이터셋 요약")
    print("="*60)

    # 기본 정보
    print(f"총 에피소드 수: {len(trajectories)}")
    print(f"성공률: {dataset['success_rate']*100:.1f}%")

    # 에피소드 길이
    lengths = [len(traj["grasp_path"]) for traj in trajectories]
    print(f"\n에피소드 길이:")
    print(f"  평균: {np.mean(lengths):.1f}")
    print(f"  최소: {np.min(lengths)}")
    print(f"  최대: {np.max(lengths)}")
    print(f"  표준편차: {np.std(lengths):.1f}")

    # 최종 거리
    final_distances = []
    for traj in trajectories:
        final_pos = traj["grasp_path"][-1]
        cap_pos = traj["cap_pos"]
        dist = np.linalg.norm(final_pos - cap_pos)
        final_distances.append(dist)

    print(f"\n최종 거리 (캡까지):")
    print(f"  평균: {np.mean(final_distances)*100:.2f} cm")
    print(f"  최소: {np.min(final_distances)*100:.2f} cm")
    print(f"  최대: {np.max(final_distances)*100:.2f} cm")

    # 경로 효율성
    efficiencies = []
    for traj in trajectories:
        path = traj["grasp_path"]
        straight_dist = np.linalg.norm(path[-1] - path[0])
        actual_dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        if actual_dist > 0:
            efficiencies.append(straight_dist / actual_dist)

    print(f"\n경로 효율성 (직선/실제):")
    print(f"  평균: {np.mean(efficiencies):.3f}")
    print(f"  최소: {np.min(efficiencies):.3f}")
    print(f"  최대: {np.max(efficiencies):.3f}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize Expert Trajectories")
    parser.add_argument("--data", type=str, required=True,
                        help="전문가 데이터 파일 경로 (.pt)")
    parser.add_argument("--output", type=str, default="trajectory_plot.png",
                        help="출력 이미지 경로")
    parser.add_argument("--max_traj", type=int, default=50,
                        help="시각화할 최대 궤적 수")

    args = parser.parse_args()

    # 데이터 로드
    dataset = load_dataset(args.data)

    # 위치 추출
    trajectories = extract_positions(dataset["observations"])

    # 요약 출력
    print_summary(trajectories, dataset)

    # 시각화
    plot_3d_trajectories(trajectories, args.output, max_trajectories=args.max_traj)
    plot_2d_top_view(trajectories, args.output, max_trajectories=args.max_traj)
    plot_statistics(trajectories, dataset, args.output)

    print("\n시각화 완료!")


if __name__ == "__main__":
    main()
