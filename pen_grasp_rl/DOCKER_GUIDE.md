# Pen Grasp RL - Docker 사용 가이드

## 개요

이 프로젝트는 Docker를 사용해 Isaac Sim/Lab 환경을 실행합니다.
**노트북에 Isaac Sim을 직접 설치할 필요 없이**, Docker만 있으면 학습 환경을 바로 구축할 수 있습니다.

### Docker의 장점
- 호스트 PC의 Isaac Sim/Lab 버전과 무관하게 동작
- 동일한 환경을 모든 팀원이 공유 가능
- 노트북 GPU 성능만 활용 (환경은 컨테이너 내부)
- 코드 수정 시 바로 반영 (볼륨 마운트)

---

## 1. 사전 요구사항

| 항목 | 요구사항 |
|------|----------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA (RTX 3070 이상 권장) |
| NVIDIA Driver | 535 이상 |
| Docker | 26.0.0 이상 |
| Docker Compose | 2.25.0 이상 |

---

## 2. 최초 설정 (1회만)

### 2.1 NVIDIA 드라이버 설치
```bash
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

드라이버 확인:
```bash
nvidia-smi
```

### 2.2 Docker 설치
```bash
# Docker 설치
sudo apt install docker.io docker-compose-v2

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 사용)
sudo usermod -aG docker $USER
newgrp docker
```

### 2.3 NVIDIA Container Toolkit 설치
```bash
# 저장소 추가
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2.4 NGC 로그인

Isaac Sim Docker 이미지는 NVIDIA NGC에서 제공됩니다.

1. [NGC 웹사이트](https://ngc.nvidia.com/) 가입 (무료)
2. 로그인 후 우측 상단 프로필 → **Setup**
3. **Generate API Key** 클릭 (기본 권한으로 생성)
4. 생성된 키 복사 (한 번만 표시되므로 꼭 저장!)

```bash
docker login nvcr.io
# Username: $oauthtoken  (그대로 입력)
# Password: <발급받은 API Key>
```

---

## 3. 프로젝트 설정

### 3.1 프로젝트 폴더 받기
```bash
# 1. Isaac Lab 공식 레포 클론
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 2. 팀 프로젝트 클론 (pen_grasp_rl 포함)
git clone https://github.com/KERNEL3-2/COBOT_WRITER.git temp_cobot
mv temp_cobot/pen_grasp_rl ./
rm -rf temp_cobot

# 3. 로봇 USD 파일 받기 (팀장/관리자에게 요청)
# first_control.usd 파일을 적절한 위치에 복사
# 경로: pen_grasp_rl/envs/pen_grasp_env.py 에서 확인
```

### 3.2 Docker 설정 파일 수정
```bash
cd ~/IsaacLab/docker
```

`docker-compose.yaml` 파일을 열고 `x-default-isaac-lab-volumes` 섹션에 아래 내용 추가:

```yaml
    # Pen Grasp RL project
  - type: bind
    source: ../pen_grasp_rl
    target: ${DOCKER_ISAACLAB_PATH}/pen_grasp_rl
    # Logs - 호스트에서 바로 접근 가능하도록
  - type: bind
    source: ../logs
    target: ${DOCKER_ISAACLAB_PATH}/logs
```

> **참고**: logs를 bind mount하면 학습 결과를 호스트에서 바로 확인/복사할 수 있습니다.

### 3.3 Docker 이미지 빌드
```bash
cd ~/IsaacLab/docker

# 필수 파일 생성
touch .isaac-lab-docker-history

# 이미지 빌드 (최초 1회, 약 30분 소요)
docker compose --profile base build
```

---

## 4. 학습 실행

### 4.1 컨테이너 시작
```bash
cd ~/IsaacLab/docker

# 컨테이너 시작 (백그라운드)
docker compose --profile base up -d

# 컨테이너 진입
docker compose --profile base exec isaac-lab-base bash
```

### 4.2 의존성 설치 (컨테이너 내부, 최초 1회)
```bash
cd /workspace/isaaclab
./pen_grasp_rl/docker_setup.sh
```

### 4.3 학습 실행 (컨테이너 내부)
```bash
# Headless 학습 (GUI 없이)
python pen_grasp_rl/scripts/train.py --headless --num_envs 4096 --max_iterations 3000

# 학습 결과는 /workspace/isaaclab/logs/pen_grasp 에 저장됨
```

### 4.4 컨테이너 종료
```bash
# 컨테이너 내부에서 나가기
exit

# 컨테이너 중지
cd ~/IsaacLab/docker
docker compose --profile base down
```

---

## 5. 일상적인 사용

### 매일 작업 시작할 때
```bash
cd ~/IsaacLab/docker
docker compose --profile base up -d
docker compose --profile base exec isaac-lab-base bash

# 컨테이너 내부
cd /workspace/isaaclab
python pen_grasp_rl/scripts/train.py --headless --num_envs 4096
```

### 작업 끝날 때
```bash
exit  # 컨테이너에서 나가기
docker compose --profile base down  # 컨테이너 중지
```

---

## 6. 코드 수정

코드는 **호스트(노트북)에서 수정**하면 됩니다.
볼륨 마운트 되어 있어서 컨테이너 안에 바로 반영됩니다.

| 호스트 경로 | 컨테이너 경로 |
|-------------|---------------|
| `~/IsaacLab/pen_grasp_rl/` | `/workspace/isaaclab/pen_grasp_rl/` |
| `~/IsaacLab/source/` | `/workspace/isaaclab/source/` |

```bash
# 예: VS Code로 수정
code ~/IsaacLab/pen_grasp_rl/

# 수정 후 컨테이너에서 바로 실행 가능
```

---

## 7. TensorBoard로 학습 모니터링

### 방법 1: 호스트에서 실행
```bash
# 호스트 터미널에서
pip install tensorboard
tensorboard --logdir=~/IsaacLab/logs/pen_grasp
# 브라우저: http://localhost:6006
```

### 방법 2: 컨테이너 안에서 실행
```bash
# 컨테이너 내부에서
tensorboard --logdir=/workspace/isaaclab/logs/pen_grasp --bind_all
# 브라우저: http://localhost:6006
```

---

## 8. 문제 해결

### GPU 인식 안 될 때
```bash
# Docker에서 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 권한 문제
```bash
# docker 그룹에 추가 안 된 경우
sudo usermod -aG docker $USER
newgrp docker
```

### 이미지 빌드 실패
```bash
# 캐시 삭제 후 재빌드
docker system prune -a
docker compose --profile base build --no-cache
```

### 컨테이너 상태 확인
```bash
docker ps -a
docker compose --profile base logs
```

---

## 9. NGC 계정 관리

### 다른 PC에서 사용
- 동일한 API Key를 여러 PC에서 사용 가능
- 각 PC에서 `docker login nvcr.io` 실행

### PC 사용 종료 시
```bash
# 해당 PC에서 로그아웃
docker logout nvcr.io
```

### API Key 분실/유출 시
1. [NGC 웹사이트](https://ngc.nvidia.com/) 접속
2. Setup → API Key → **Revoke** (폐기)
3. 새 키 발급
4. 모든 PC에서 다시 로그인

---

## 10. 요약: 새 PC 설정 체크리스트

- [ ] NVIDIA 드라이버 설치
- [ ] Docker 설치
- [ ] NVIDIA Container Toolkit 설치
- [ ] NGC 로그인 (`docker login nvcr.io`)
- [ ] Isaac Lab 클론 (`git clone https://github.com/isaac-sim/IsaacLab.git`)
- [ ] 팀 프로젝트 클론 (`git clone https://github.com/KERNEL3-2/COBOT_WRITER.git`)
- [ ] pen_grasp_rl 폴더를 IsaacLab 안으로 이동
- [ ] 로봇 USD 파일 복사 (팀장에게 요청)
- [ ] docker-compose.yaml에 pen_grasp_rl 볼륨 마운트 추가
- [ ] `docker compose --profile base build`
- [ ] `./pen_grasp_rl/docker_setup.sh` (컨테이너 내부)

설정 완료 후에는 4번(학습 실행) 섹션부터 따라하면 됩니다.
