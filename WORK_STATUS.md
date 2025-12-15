# Pen Grasp RL 작업 현황

> 마지막 업데이트: 2025-12-15

---

## 1. 현재 진행 상황

### 학습 상태
- **진행된 iteration**: 3500
- **체크포인트 위치**:
  - Docker 컨테이너: `/workspace/isaaclab/logs/pen_grasp/model_3500.pt`
  - 로컬 백업: `/home/fhekwn549/pen_grasp_3500.pt`
- **TensorBoard 로그**: `/home/fhekwn549/pen_grasp_3500_events.tfevents`

### 학습 결과 요약 (3500 iter)
| 항목 | 값 | 평가 |
|------|-----|------|
| 평균 리워드 | -8.5 → +7.0 | 안정적 |
| distance_to_cap | +0.61 | 펜에 접근 중 |
| z_axis_alignment | 0.00 | 10cm 이내 미진입 |
| Value Loss | 0.02 | 안정적 |

### 이전 학습 대비 개선
- 이전(5900 iter): 리워드 +1.7 → -3.9 급락 (학습 붕괴)
- 현재(3500 iter): 리워드 -8.5 → +7.0 유지 (안정적)

---

## 2. 변경된 설정 (2025-12-15)

### train.py
| 항목 | 이전 | 변경 |
|------|------|------|
| init_noise_std | 1.0 | **0.3** |

### pen_grasp_env.py
| 항목 | 이전 | 변경 |
|------|------|------|
| joint_5 초기값 | -90도 | **+90도** (그리퍼 아래 향함) |
| 펜 모델 | CylinderCfg | **pen.usd** |
| distance_to_cap 보상 | `1/(1+d*10)` | **`exp(-d*10)`** |
| z_axis_alignment 조건 | 5cm | **10cm** |
| action_rate weight | 0.1 | **0.01** |
| 펜 방향 | 360도 랜덤 | **수직 고정** |

---

## 3. 다음 할 일

### 즉시 (학습 재개)
```bash
# Docker 컨테이너 시작
cd ~/IsaacLab
./docker/container.py start
./docker/container.py enter base

# 학습 이어서 하기
cd /workspace/isaaclab
python pen_grasp_rl/scripts/train.py --headless --num_envs 8192 --max_iterations 6500 \
    --resume --checkpoint /workspace/isaaclab/logs/pen_grasp/model_3500.pt
```

### 학습 완료 후 (10000 iter 이상)
1. 학습 곡선 분석 (TensorBoard)
2. play.py로 정책 시각화
3. Domain Randomization 적용 검토

---

## 4. Sim-to-Real 적용 계획

### Phase 1: 기본 학습 완료 (현재)
- [x] 환경 설정 최적화
- [x] 안정적인 학습 확인
- [ ] 10000+ iteration 학습
- [ ] z_axis_alignment 보상 획득 확인

### Phase 2: Domain Randomization 추가
```python
# 추가할 randomization
physics:
  pen_mass: [0.008, 0.015]        # kg
  pen_friction: [0.4, 1.0]
  gripper_friction: [0.6, 1.2]
  joint_friction: [0.9, 1.1]      # 비율

dynamics:
  action_delay: [0, 2]            # timesteps
  observation_noise:
    position: 0.002               # meters
    orientation: 0.01             # radians
```

### Phase 3: Observation 확장
- Contact force 추가 (그리퍼-펜 접촉력)
- Previous action 추가 (smoothness)

### Phase 4: Curriculum Learning
1. 펜 위치 고정
2. 펜 위치 랜덤 (현재)
3. 펜 각도 랜덤 (±30도부터)
4. 전체 랜덤 + 사람 손 모델

---

## 5. 주요 파일 경로

### 로컬 (이 노트북)
```
~/CoWriteBotRL/
├── pen_grasp_rl/
│   ├── envs/pen_grasp_env.py      # 환경 설정
│   ├── scripts/train.py           # 학습 스크립트 (resume 기능 추가됨)
│   ├── scripts/play.py            # 정책 테스트
│   └── models/pen.usd             # 펜 모델
└── WORK_STATUS.md                 # 이 파일

~/IsaacLab/
├── sim2real_guide.md              # Sim-to-Real 가이드 (참고)
└── scripts/sim2sim_transfer/      # 정책 전이 스크립트
```

### Docker 컨테이너 (다른 노트북)
```
/workspace/isaaclab/
├── pen_grasp_rl/                  # git pull로 동기화
└── logs/pen_grasp/
    ├── model_3500.pt              # 최신 체크포인트
    └── events.out.tfevents.*      # TensorBoard 로그
```

---

## 6. 유용한 명령어

### 학습 모니터링
```bash
# GPU 온도 실시간 확인 (다른 터미널에서)
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir=~/pen_grasp_logs
```

### 체크포인트 복사 (Docker → 로컬)
```bash
docker cp isaac-lab-base:/workspace/isaaclab/logs/pen_grasp/model_XXXX.pt ~/
docker cp isaac-lab-base:/workspace/isaaclab/logs/pen_grasp/events.out.tfevents.* ~/
```

### Git 동기화
```bash
# 로컬에서 push 후 Docker에서 pull
cd /workspace/isaaclab/pen_grasp_rl
git pull origin main
```

---

## 7. 참고 자료

- `~/IsaacLab/sim2real_guide.md` - Domain Randomization, Reward 설계, Contact 설정 등
- `~/IsaacLab/pen_grasp_rl/TRAINING_LOG.md` - 학습 히스토리

---

## 8. 알려진 이슈

### Docker GPU 인식 안 됨
컨테이너 재시작으로 해결:
```bash
./docker/container.py stop
./docker/container.py start
./docker/container.py enter base
nvidia-smi  # GPU 확인
```

### 팬 소음
- RTX 5080 노트북은 75~85도에서 팬 소음 정상
- 90도 이상이면 스로틀링 (성능 저하되지만 안전)
- 걱정 안 해도 됨
