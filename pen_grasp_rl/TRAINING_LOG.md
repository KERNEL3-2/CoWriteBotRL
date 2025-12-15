# Pen Grasp RL Training Log

## 2025-12-15: 학습 설정 최적화

### 이전 학습 분석 (2025-12-15_02-55-00)
- 5,900 iterations 진행
- **문제점**: 리워드가 +1.7에서 -3.9로 급락, Value Loss 1000배 증가
- **원인 분석**:
  - init_noise_std=1.0이 너무 높음 (초기 행동이 거의 랜덤)
  - 펜 방향 360도 랜덤화로 학습 난이도 과다
  - alignment 보상 조건이 너무 엄격 (5cm 이내)
  - action_rate 페널티가 너무 작음

### 변경 사항

#### 1. PPO 하이퍼파라미터 (train.py)
| 항목 | 이전 | 변경 |
|------|------|------|
| init_noise_std | 1.0 | **0.3** |

#### 2. 환경 설정 (pen_grasp_env.py)

**펜 방향**
- 이전: roll/pitch/yaw 모두 ±180도 랜덤
- 변경: **수직 고정** (캡이 위를 향함)
- 향후: 학습 성공 후 점진적으로 각도 추가 예정

**로봇 초기 자세**
- 이전: joint_5 = -90도
- 변경: joint_5 = **+90도** (그리퍼가 아래를 향함)

**리워드 함수**
| 항목 | 이전 | 변경 |
|------|------|------|
| distance_to_cap | `1/(1+d*10)` | **`exp(-d*10)`** (exponential) |
| z_axis_alignment 거리조건 | 5cm | **10cm** |
| action_rate 함수 | `action² * 0.001` | **`action²`** |
| action_rate weight | 0.1 | **0.01** |

#### 3. 펜 모델
- 이전: CylinderCfg (단순 실린더)
- 변경: **pen.usd** (BackCap, Body, TipCone, TipSphere 포함)

### 학습 실행 명령어
```bash
cd ~/IsaacLab
source ~/isaacsim_env/bin/activate
python pen_grasp_rl/scripts/train.py --headless --num_envs 4096 --max_iterations 5000
```

### 다음 단계
1. 현재 설정으로 학습 진행
2. 학습 성공 시 펜 각도 랜덤화 추가 (±30도부터 시작)
3. 그리퍼 잡기 동작 추가
