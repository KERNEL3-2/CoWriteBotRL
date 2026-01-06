"""
전문가 정책 (Expert Policy) for 펜 잡기 태스크

규칙 기반 전문가 정책으로 최적 궤적을 생성합니다.
- 펜 캡 방향으로 직선 이동
- 모방 학습용 데이터 수집
- 전문가 성능 검증

사용법:
    # 전문가 정책 실행 (시각화)
    python expert_policy.py --mode play --num_envs 1

    # 전문가 성능 평가
    python expert_policy.py --mode eval --num_envs 64 --episodes 100

    # 궤적 수집 (모방 학습용)
    python expert_policy.py --mode collect --num_envs 256 --episodes 1000 --output expert_data.pt
"""

from __future__ import annotations

import os
import sys
import argparse
import torch
import numpy as np
from collections import defaultdict

# IsaacLab 환경 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from isaaclab.app import AppLauncher

# argparse 먼저 처리
parser = argparse.ArgumentParser(description="Expert Policy for Pen Grasping")
parser.add_argument("--mode", type=str, default="play",
                    choices=["play", "eval", "collect"],
                    help="실행 모드: play(시각화), eval(평가), collect(데이터수집)")
parser.add_argument("--num_envs", type=int, default=1,
                    help="환경 수")
parser.add_argument("--episodes", type=int, default=10,
                    help="에피소드 수")
parser.add_argument("--output", type=str, default="expert_data.pt",
                    help="수집된 데이터 저장 경로")
parser.add_argument("--approach_height", type=float, default=0.05,
                    help="접근 높이 (m)")
parser.add_argument("--move_speed", type=float, default=0.8,
                    help="이동 속도 스케일 (0-1)")

# AppLauncher 인자 추가
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# headless 모드 설정
if args.mode in ["eval", "collect"]:
    args.headless = True

# Isaac Sim 시작
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Lab 임포트 (시뮬레이션 시작 후)
from envs.e0509_expert_env import E0509ExpertEnv, E0509ExpertEnvCfg, E0509ExpertEnvCfg_PLAY


def extract_obs(obs):
    """관측값에서 텐서 추출 (딕셔너리 또는 텐서 지원)"""
    if isinstance(obs, dict):
        return obs.get("policy", obs.get("obs", list(obs.values())[0]))
    return obs


class ExpertPolicy:
    """
    규칙 기반 전문가 정책 (펜 축 기준 접근)

    IK 환경처럼 펜 축 방향으로 접근합니다.
    - Phase 0: 펜 축 방향 접근 위치로 이동
    - Phase 1: 펜 축 따라 캡으로 이동
    - Phase 2: 캡 위치 유지
    """

    # 관측값 인덱스 (e0509_expert_env 기준)
    OBS_JOINT_POS = slice(0, 6)
    OBS_JOINT_VEL = slice(6, 12)
    OBS_GRASP_POS = slice(12, 15)
    OBS_CAP_POS = slice(15, 18)
    OBS_REL_POS = slice(18, 21)
    OBS_PEN_Z = slice(21, 24)
    OBS_DIST_TO_CAP = 24
    OBS_APPROACH_DIR = slice(25, 28)  # 접근 방향 (-pen_z)
    OBS_APPROACH_DIST = 28            # 축 방향 거리

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        approach_height: float = 0.05,
        move_speed: float = 0.8,
    ):
        """
        Args:
            num_envs: 환경 수
            device: 디바이스
            approach_height: 접근 높이 (m) - 캡 위에서 먼저 접근
            move_speed: 이동 속도 스케일 (0-1)
        """
        self.num_envs = num_envs
        self.device = device
        self.approach_height = approach_height
        self.move_speed = move_speed

        # 각 환경의 페이즈 (0: approach, 1: descend, 2: reach)
        self.phase = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # 상태 전환 임계값
        self.approach_threshold = 0.03  # 3cm 이내면 다음 페이즈
        self.descend_threshold = 0.02   # 2cm 이내면 다음 페이즈

    def reset(self, env_ids: torch.Tensor = None):
        """페이즈 리셋"""
        if env_ids is None:
            self.phase.zero_()
        else:
            self.phase[env_ids] = 0

    def get_action(self, obs) -> torch.Tensor:
        """
        관측값에서 행동 계산 (펜 축 기준 접근)

        Args:
            obs: 관측값 (딕셔너리 또는 텐서)

        Returns:
            action: (num_envs, 3) 위치 변화량 [Δx, Δy, Δz]
        """
        # 관측값 추출
        obs_tensor = extract_obs(obs)

        # 관측값 파싱
        grasp_pos = obs_tensor[:, self.OBS_GRASP_POS]    # 현재 그립 위치
        cap_pos = obs_tensor[:, self.OBS_CAP_POS]        # 캡 위치
        rel_pos = obs_tensor[:, self.OBS_REL_POS]        # 캡까지 상대 위치
        pen_z = obs_tensor[:, self.OBS_PEN_Z]            # 펜 축 방향
        approach_dir = obs_tensor[:, self.OBS_APPROACH_DIR]  # 접근 방향 (-pen_z)

        # 목표 위치 계산 (페이즈별)
        actions = torch.zeros(self.num_envs, 3, device=self.device)

        # 페이즈 0: 펜 축 방향 접근 위치로 이동
        # 캡에서 approach_height 만큼 떨어진 위치 (펜 축 방향)
        phase0_mask = self.phase == 0
        if phase0_mask.any():
            # 접근 위치: 캡 + approach_dir * approach_height
            target_pos = cap_pos + approach_dir * self.approach_height
            direction = target_pos - grasp_pos

            dist = torch.norm(direction, dim=-1, keepdim=True)
            direction_norm = direction / (dist + 1e-6)

            actions[phase0_mask] = (direction_norm * self.move_speed)[phase0_mask]

            # 페이즈 전환: 접근 위치에 도착
            close_enough = dist.squeeze(-1) < self.approach_threshold
            self.phase[phase0_mask & close_enough] = 1

        # 페이즈 1: 펜 축 따라 캡으로 이동
        phase1_mask = self.phase == 1
        if phase1_mask.any():
            # 캡 방향으로 이동 (펜 축 따라)
            # approach_dir의 반대 방향이 캡 방향
            move_dir = -approach_dir  # pen_z 방향

            # 캡까지 거리 확인
            dist_to_cap = torch.norm(rel_pos, dim=-1, keepdim=True)

            # 속도 조절: 가까우면 천천히
            speed_scale = torch.clamp(dist_to_cap / 0.05, min=0.3, max=1.0)

            actions[phase1_mask] = (move_dir * self.move_speed * 0.5 * speed_scale)[phase1_mask]

            # 페이즈 전환: 캡 근처 도착
            close_enough = dist_to_cap.squeeze(-1) < self.descend_threshold
            self.phase[phase1_mask & close_enough] = 2

        # 페이즈 2: 캡 위치 유지 (성공 조건 만족)
        phase2_mask = self.phase == 2
        if phase2_mask.any():
            # 캡으로 미세 조정
            dist = torch.norm(rel_pos, dim=-1, keepdim=True)
            direction_norm = rel_pos / (dist + 1e-6)

            actions[phase2_mask] = (direction_norm * self.move_speed * 0.2)[phase2_mask]

        return actions


class SimpleExpertPolicy:
    """
    단순 전문가 정책 (페이즈 없음)

    캡 방향으로 직선 이동만 수행합니다.
    """

    OBS_REL_POS = slice(18, 21)
    OBS_DIST_TO_CAP = 25

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        move_speed: float = 0.8,
    ):
        self.num_envs = num_envs
        self.device = device
        self.move_speed = move_speed

    def reset(self, env_ids: torch.Tensor = None):
        pass  # 상태 없음

    def get_action(self, obs) -> torch.Tensor:
        """캡 방향으로 직선 이동"""
        obs_tensor = extract_obs(obs)
        rel_pos = obs_tensor[:, self.OBS_REL_POS]  # 캡까지 상대 위치
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        # 정규화된 방향
        direction = rel_pos / (dist + 1e-6)

        # 거리에 비례하여 속도 조절 (가까우면 천천히)
        speed_scale = torch.clamp(dist / 0.1, min=0.2, max=1.0)

        return direction * self.move_speed * speed_scale


def play_expert(env, expert, max_steps: int = 500):
    """전문가 정책 시각화 실행"""
    print("\n" + "="*60)
    print("전문가 정책 시각화 모드")
    print("="*60)
    print("Ctrl+C로 종료")
    print("="*60 + "\n")

    obs, _ = env.reset()
    expert.reset()

    step = 0
    episode = 0

    try:
        while simulation_app.is_running():
            # 전문가 행동
            action = expert.get_action(obs)

            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            step += 1

            # 상태 출력 (10 스텝마다)
            if step % 10 == 0:
                obs_tensor = extract_obs(obs)
                dist = obs_tensor[0, 25].item()  # distance_to_cap
                print(f"\r에피소드 {episode+1} | 스텝 {step} | 거리: {dist:.3f}m", end="", flush=True)

            # 에피소드 종료 처리
            if done.any():
                episode += 1
                if info.get("success", torch.zeros(1))[0]:
                    print(f"\n[성공] 에피소드 {episode} 완료!")
                else:
                    print(f"\n[실패] 에피소드 {episode} 종료")

                obs, _ = env.reset()
                expert.reset()
                step = 0

    except KeyboardInterrupt:
        print("\n\n종료됨")


def evaluate_expert(env, expert, num_episodes: int = 100):
    """전문가 정책 성능 평가"""
    print("\n" + "="*60)
    print("전문가 정책 평가 모드")
    print(f"평가 에피소드: {num_episodes}")
    print("="*60 + "\n")

    successes = 0
    total_steps = 0
    total_rewards = 0
    episode_count = 0

    obs, _ = env.reset()
    expert.reset()

    episode_steps = torch.zeros(env.num_envs, device=env.device)
    episode_rewards = torch.zeros(env.num_envs, device=env.device)

    while episode_count < num_episodes:
        action = expert.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        episode_steps += 1
        episode_rewards += reward

        # 완료된 에피소드 처리
        done_envs = done.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_envs:
            if episode_count >= num_episodes:
                break

            idx = idx.item()
            success = info.get("success", torch.zeros(env.num_envs, device=env.device))[idx].item()

            if success:
                successes += 1
            total_steps += episode_steps[idx].item()
            total_rewards += episode_rewards[idx].item()
            episode_count += 1

            # 리셋
            episode_steps[idx] = 0
            episode_rewards[idx] = 0
            expert.reset(torch.tensor([idx], device=env.device))

            # 진행 상황 출력
            if episode_count % 10 == 0:
                print(f"진행: {episode_count}/{num_episodes} | 성공률: {successes/episode_count*100:.1f}%")

    # 최종 결과
    print("\n" + "="*60)
    print("평가 결과")
    print("="*60)
    print(f"총 에피소드: {num_episodes}")
    print(f"성공 횟수: {successes}")
    print(f"성공률: {successes/num_episodes*100:.1f}%")
    print(f"평균 스텝: {total_steps/num_episodes:.1f}")
    print(f"평균 보상: {total_rewards/num_episodes:.2f}")
    print("="*60)

    return {
        "success_rate": successes / num_episodes,
        "avg_steps": total_steps / num_episodes,
        "avg_reward": total_rewards / num_episodes,
    }


def collect_trajectories(env, expert, num_episodes: int = 1000, output_path: str = "expert_data.pt"):
    """전문가 궤적 수집"""
    print("\n" + "="*60)
    print("전문가 궤적 수집 모드")
    print(f"수집 에피소드: {num_episodes}")
    print(f"저장 경로: {output_path}")
    print("="*60 + "\n")

    # 데이터 저장용
    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []

    # 현재 에피소드 버퍼 (환경별)
    episode_obs = [[] for _ in range(env.num_envs)]
    episode_actions = [[] for _ in range(env.num_envs)]
    episode_rewards = [[] for _ in range(env.num_envs)]

    obs, _ = env.reset()
    expert.reset()

    episode_count = 0
    success_count = 0

    while episode_count < num_episodes:
        action = expert.get_action(obs)

        # 현재 상태 저장
        obs_tensor = extract_obs(obs)
        for i in range(env.num_envs):
            episode_obs[i].append(obs_tensor[i].cpu().clone())
            episode_actions[i].append(action[i].cpu().clone())

        # 환경 스텝
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # 보상 저장
        for i in range(env.num_envs):
            episode_rewards[i].append(reward[i].cpu().item())

        # 완료된 에피소드 처리
        done_envs = done.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_envs:
            if episode_count >= num_episodes:
                break

            idx = idx.item()
            success = info.get("success", torch.zeros(env.num_envs, device=env.device))[idx].item()

            # 성공한 에피소드만 저장 (옵션)
            if success:
                success_count += 1

                # 에피소드 데이터 저장
                ep_obs = torch.stack(episode_obs[idx])
                ep_actions = torch.stack(episode_actions[idx])
                ep_rewards = torch.tensor(episode_rewards[idx])
                ep_dones = torch.zeros(len(episode_rewards[idx]))
                ep_dones[-1] = 1.0

                all_observations.append(ep_obs)
                all_actions.append(ep_actions)
                all_rewards.append(ep_rewards)
                all_dones.append(ep_dones)

            episode_count += 1

            # 에피소드 버퍼 리셋
            episode_obs[idx] = []
            episode_actions[idx] = []
            episode_rewards[idx] = []
            expert.reset(torch.tensor([idx], device=env.device))

            # 진행 상황 출력
            if episode_count % 50 == 0:
                print(f"진행: {episode_count}/{num_episodes} | 성공: {success_count} ({success_count/episode_count*100:.1f}%)")

        obs = next_obs

    # 데이터 저장
    if len(all_observations) > 0:
        dataset = {
            "observations": all_observations,
            "actions": all_actions,
            "rewards": all_rewards,
            "dones": all_dones,
            "num_episodes": len(all_observations),
            "success_rate": success_count / num_episodes,
        }

        # 폴더 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(dataset, output_path)

        print("\n" + "="*60)
        print("수집 완료")
        print("="*60)
        print(f"총 에피소드: {num_episodes}")
        print(f"저장된 에피소드 (성공): {len(all_observations)}")
        print(f"성공률: {success_count/num_episodes*100:.1f}%")
        print(f"저장 경로: {output_path}")

        # 통계
        total_steps = sum(len(ep) for ep in all_observations)
        avg_steps = total_steps / len(all_observations) if all_observations else 0
        print(f"총 스텝 수: {total_steps}")
        print(f"평균 에피소드 길이: {avg_steps:.1f}")
        print("="*60)
    else:
        print("\n[경고] 저장할 성공 에피소드가 없습니다!")

    return dataset if len(all_observations) > 0 else None


def main():
    """메인 함수"""
    # 환경 설정 (전문가 궤적용 IK 환경)
    if args.mode == "play":
        cfg = E0509ExpertEnvCfg_PLAY()
    else:
        cfg = E0509ExpertEnvCfg()

    cfg.scene.num_envs = args.num_envs
    print(f"E0509ExpertEnv 사용 (IK 기반, 펜 축 접근)")

    # 환경 생성
    env = E0509ExpertEnv(cfg)

    # 펜 축 기준 전문가 정책
    expert = ExpertPolicy(
        num_envs=args.num_envs,
        device=env.device,
        approach_height=args.approach_height,
        move_speed=args.move_speed,
    )

    print(f"\n전문가 정책: {expert.__class__.__name__}")
    print(f"환경 수: {args.num_envs}")
    print(f"이동 속도: {args.move_speed}")

    try:
        if args.mode == "play":
            play_expert(env, expert)
        elif args.mode == "eval":
            evaluate_expert(env, expert, num_episodes=args.episodes)
        elif args.mode == "collect":
            collect_trajectories(env, expert, num_episodes=args.episodes, output_path=args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
