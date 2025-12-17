# 설치 가이드

## 요구 사항

- Ubuntu 22.04
- NVIDIA GPU (RTX 3060 이상 권장)
- CUDA 12.x
- Isaac Sim 4.2+
- Isaac Lab

## 설치 순서

### 1. Isaac Sim 설치

Omniverse Launcher를 통해 Isaac Sim 설치:
```bash
# Omniverse Launcher 다운로드 후 Isaac Sim 설치
# 버전: 4.2 이상
```

### 2. Isaac Lab 설치

```bash
# Isaac Lab 클론
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 가상환경 생성 및 설치
./isaaclab.sh --install
```

### 3. 가상환경 활성화

```bash
source ~/isaacsim_env/bin/activate
```

### 4. pen_grasp_rl 설치

```bash
# CoWriteBotRL 클론
git clone https://github.com/KERNEL3-2/CoWriteBotRL.git

# pen_grasp_rl 폴더를 IsaacLab으로 복사
cp -r CoWriteBotRL/pen_grasp_rl ~/IsaacLab/
```

### 5. 의존성 확인

```bash
cd ~/IsaacLab
pip list | grep rsl-rl
pip list | grep tensorboard
```

## 실행 테스트

```bash
cd ~/IsaacLab

# 환경 테스트 (GUI)
python pen_grasp_rl/scripts/train_reach.py --num_envs 64

# 학습 실행 (headless)
python pen_grasp_rl/scripts/train_reach.py --headless --num_envs 4096
```

## 폴더 구조

```
IsaacLab/
└── pen_grasp_rl/
    ├── envs/
    │   ├── e0509_reach_env.py    # E0509 Reach 환경
    │   └── __init__.py
    ├── scripts/
    │   ├── train_reach.py        # 학습 스크립트
    │   └── play_reach.py         # 테스트 스크립트
    └── models/
        └── first_control.usd     # E0509 로봇 모델
```

## 문제 해결

### Isaac Sim이 실행되지 않는 경우
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# 환경 변수 설정
export ISAACSIM_PATH=~/.local/share/ov/pkg/isaac-sim-4.2.0
```

### 모듈 import 오류
```bash
# IsaacLab 폴더에서 실행 확인
cd ~/IsaacLab
python pen_grasp_rl/scripts/train_reach.py --help
```

## 참고 링크

- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
