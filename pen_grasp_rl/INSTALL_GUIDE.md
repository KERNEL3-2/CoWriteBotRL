# Pen Grasp RL - 설치 가이드

## 시스템 요구사항

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| NVIDIA Driver | 580.x 이상 |
| Python | 3.11 |
| GPU | CUDA 지원 (RTX 3070 이상 권장) |

## 설치 순서

### 1. NVIDIA 드라이버 설치
```bash
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 2. Python 3.11 가상환경 생성
```bash
sudo apt install python3.11 python3.11-venv
python3.11 -m venv ~/isaacsim_env
source ~/isaacsim_env/bin/activate
```

### 3. Isaac Sim 설치
```bash
pip install --upgrade pip
pip install isaacsim==4.5.0.0 --extra-index-url https://pypi.nvidia.com
```

### 4. Isaac Lab 설치
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

### 5. 추가 패키지 설치
```bash
pip install tensordict==0.10.0
pip install git+https://github.com/leggedrobotics/rsl_rl.git
pip install tensorboard
```

### 6. 프로젝트 파일 복사
- `pen_grasp_rl/` 폴더를 `IsaacLab/` 디렉토리에 복사
- 로봇 USD 파일 경로 확인: `/home/fhekwn549/isaac_pen_trajectory/models/doosan_e0509/first_control.usd`

## 학습 실행

```bash
cd ~/IsaacLab
source ~/isaacsim_env/bin/activate

# Headless 학습 (GUI 없이)
python pen_grasp_rl/scripts/train.py --headless --num_envs 4096

# GUI로 학습 확인
python pen_grasp_rl/scripts/train.py --num_envs 64
```

## TensorBoard로 학습 모니터링

```bash
tensorboard --logdir=./logs/pen_grasp
# 브라우저에서 http://localhost:6006 접속
```

## 문제 해결

### 패키지 충돌 발생 시
```bash
pip install -r pen_grasp_rl/requirements.txt --no-deps
```

### CPU 절전 모드 해제 (학습 속도 향상)
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## 환경 정보

- Isaac Sim: 5.1.0
- Isaac Lab: main branch (2024.12)
- PyTorch: 2.7.0
- CUDA: 12.6

---

## Docker로 실행하기 (권장)

Docker를 사용하면 복잡한 의존성 설치 없이 바로 학습 환경을 구축할 수 있습니다.

### 사전 요구사항

1. **Docker Engine** 26.0.0 이상
2. **Docker Compose** 2.25.0 이상
3. **NVIDIA Container Toolkit**

```bash
# NVIDIA Container Toolkit 설치
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### NGC 계정 설정

Isaac Sim Docker 이미지는 NVIDIA NGC에서 제공됩니다. 먼저 로그인이 필요합니다:

1. [NVIDIA NGC](https://ngc.nvidia.com/) 계정 생성
2. API Key 발급 (Account > Setup > Generate API Key)
3. Docker 로그인:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC_API_KEY>
```

### Docker 이미지 빌드

```bash
cd ~/IsaacLab/docker

# bash history 파일 생성 (없으면 에러 발생)
touch .isaac-lab-docker-history

# Docker 이미지 빌드 (약 30분 소요)
docker compose --profile base build
```

### Docker 컨테이너 실행

```bash
cd ~/IsaacLab/docker

# 컨테이너 시작
docker compose --profile base up -d

# 컨테이너 진입
docker compose --profile base exec isaac-lab-base bash
```

또는 Isaac Lab 제공 스크립트 사용:
```bash
cd ~/IsaacLab

# 컨테이너 시작
./docker/container.py start

# 컨테이너 진입
./docker/container.py enter base
```

### 컨테이너 내부에서 학습 실행

```bash
# 의존성 설치 (최초 1회)
cd /workspace/isaaclab
./pen_grasp_rl/docker_setup.sh

# 학습 실행
python pen_grasp_rl/scripts/train.py --headless --num_envs 4096 --max_iterations 3000
```

### 주요 경로

| 호스트 | 컨테이너 |
|--------|----------|
| `~/IsaacLab/pen_grasp_rl` | `/workspace/isaaclab/pen_grasp_rl` |
| `~/IsaacLab/source` | `/workspace/isaaclab/source` |
| `~/IsaacLab/logs` | `/workspace/isaaclab/logs` (볼륨) |

### 컨테이너 관리

```bash
# 컨테이너 중지
docker compose --profile base down

# 컨테이너 로그 확인
docker compose --profile base logs -f

# 볼륨 정리 (캐시 삭제)
docker volume prune
```

### 장점

- **재현성**: 동일한 환경을 어디서든 재현 가능
- **볼륨 마운트**: 코드 수정이 바로 반영됨
- **격리**: 호스트 시스템 오염 없음
- **배포 용이**: 다른 PC로 쉽게 이동 가능
