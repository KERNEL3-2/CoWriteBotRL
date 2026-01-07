"""
환경(Environment) 서브패키지

이 패키지는 Isaac Lab 기반의 강화학습 환경들을 포함합니다.

=== Manager-Based 환경 ===
- PenGraspEnv: 펜 잡기 강화학습 환경 클래스 (v1, 기존)
- PenGraspEnvCfg: 환경 설정 클래스 (v1)
- PenGraspEnvV2: 펜 캡 접근 환경 (v2, reach 예제 기반)
- PenGraspEnvCfgV2: 환경 설정 클래스 (v2)
- E0509ReachEnvCfg: E0509 Reach 환경 설정 (Isaac Lab reach 예제 기반) ★추천
- E0509ReachEnvCfg_PLAY: 테스트/시연용 설정

=== Direct 환경 ===
- E0509DirectEnv: 단계별 상태 머신 환경 (approach → align → grasp)
- E0509DirectEnvCfg: Direct 환경 설정
- E0509DirectEnvCfg_PLAY: 테스트/시연용 설정

=== Sim2Real 테스트용 ===
- SimpleMoveEnv: 간단한 이동 환경 (TCP 5cm 이동 + Home 복귀)
- SimpleMoveEnvCfg: 환경 설정
- TargetTrackingEnv: 랜덤 target 추적 환경 (Visual Servoing)
- TargetTrackingEnvCfg: 환경 설정
"""

from .pen_grasp_env import PenGraspEnv, PenGraspEnvCfg
from .pen_grasp_env_v2 import PenGraspEnv as PenGraspEnvV2
from .pen_grasp_env_v2 import PenGraspEnvCfg as PenGraspEnvCfgV2
from .e0509_reach_env import E0509ReachEnvCfg, E0509ReachEnvCfg_PLAY
from .e0509_direct_env import E0509DirectEnv, E0509DirectEnvCfg, E0509DirectEnvCfg_PLAY
from .simple_move_env import SimpleMoveEnv, SimpleMoveEnvCfg, SimpleMoveEnvCfg_PLAY
from .target_tracking_env import TargetTrackingEnv, TargetTrackingEnvCfg, TargetTrackingEnvCfg_PLAY

# IK 기반 환경
from .e0509_ik_env import E0509IKEnv, E0509IKEnvCfg, E0509IKEnvCfg_PLAY
from .e0509_ik_env_v3 import E0509IKEnvV3, E0509IKEnvV3Cfg, E0509IKEnvV3Cfg_PLAY
from .e0509_ik_env_v4 import E0509IKEnvV4, E0509IKEnvV4Cfg, E0509IKEnvV4Cfg_PLAY
from .e0509_ik_env_v5 import (
    E0509IKEnvV5, E0509IKEnvV5Cfg, E0509IKEnvV5Cfg_PLAY,
    E0509IKEnvV5Cfg_L0, E0509IKEnvV5Cfg_L1, E0509IKEnvV5Cfg_L2, E0509IKEnvV5Cfg_L3,
)
from .e0509_ik_env_v7 import (
    E0509IKEnvV7, E0509IKEnvV7Cfg, E0509IKEnvV7Cfg_PLAY,
    E0509IKEnvV7Cfg_L0, E0509IKEnvV7Cfg_L1, E0509IKEnvV7Cfg_L2, E0509IKEnvV7Cfg_L3,
)

# OSC 기반 환경
from .e0509_osc_env import E0509OSCEnv, E0509OSCEnvCfg, E0509OSCEnvCfg_PLAY

# Domain Randomization 환경 (Sim2Real Ready)
from .e0509_dr_env import E0509DREnv, E0509DREnvCfg, E0509DREnvCfg_PLAY, E0509DREnvCfg_TRAIN

__all__ = [
    # Manager-Based
    "PenGraspEnv", "PenGraspEnvCfg",
    "PenGraspEnvV2", "PenGraspEnvCfgV2",
    "E0509ReachEnvCfg", "E0509ReachEnvCfg_PLAY",
    # Direct
    "E0509DirectEnv", "E0509DirectEnvCfg", "E0509DirectEnvCfg_PLAY",
    # Sim2Real
    "SimpleMoveEnv", "SimpleMoveEnvCfg", "SimpleMoveEnvCfg_PLAY",
    "TargetTrackingEnv", "TargetTrackingEnvCfg", "TargetTrackingEnvCfg_PLAY",
    # IK 기반
    "E0509IKEnv", "E0509IKEnvCfg", "E0509IKEnvCfg_PLAY",
    "E0509IKEnvV3", "E0509IKEnvV3Cfg", "E0509IKEnvV3Cfg_PLAY",
    "E0509IKEnvV4", "E0509IKEnvV4Cfg", "E0509IKEnvV4Cfg_PLAY",
    "E0509IKEnvV5", "E0509IKEnvV5Cfg", "E0509IKEnvV5Cfg_PLAY",
    "E0509IKEnvV5Cfg_L0", "E0509IKEnvV5Cfg_L1", "E0509IKEnvV5Cfg_L2", "E0509IKEnvV5Cfg_L3",
    # V7 (APPROACH only - Sim2Real Ready)
    "E0509IKEnvV7", "E0509IKEnvV7Cfg", "E0509IKEnvV7Cfg_PLAY",
    "E0509IKEnvV7Cfg_L0", "E0509IKEnvV7Cfg_L1", "E0509IKEnvV7Cfg_L2", "E0509IKEnvV7Cfg_L3",
    # OSC (Operational Space Control)
    "E0509OSCEnv", "E0509OSCEnvCfg", "E0509OSCEnvCfg_PLAY",
    # Domain Randomization (Sim2Real Ready)
    "E0509DREnv", "E0509DREnvCfg", "E0509DREnvCfg_PLAY", "E0509DREnvCfg_TRAIN",
]
