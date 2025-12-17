# E0509 Direct 환경 학습 로그

## 환경 개요

| 항목 | 값 |
|------|-----|
| 환경 타입 | Direct (DirectRLEnv) |
| 로봇 | Doosan E0509 + RH-P12-RN-A 그리퍼 |
| 작업 | 펜 캡 접근 + Z축 정렬 |
| 학습 방식 | PPO (RSL-RL) |

## 상태 머신 (3단계)

```
APPROACH (접근)
    ↓ 거리 < 10cm
ALIGN (정렬)
    ↓ dot < -0.8
GRASP (잡기)
    ↓ 거리 < 2cm & dot < -0.9
SUCCESS!
```

## 관찰 공간 (27차원)

| 관찰 | 차원 | 설명 |
|------|------|------|
| joint_pos | 6 | 관절 위치 |
| joint_vel | 6 | 관절 속도 |
| grasp_pos | 3 | 그리퍼 잡기 포인트 (로컬) |
| cap_pos | 3 | 펜 캡 위치 (로컬) |
| rel_pos | 3 | 상대 위치 (cap - grasp) |
| gripper_z | 3 | 그리퍼 Z축 방향 |
| pen_z | 3 | 펜 Z축 방향 |

## 액션 공간 (6차원)

- 6 DOF 팔 관절 delta position control
- action_scale: 0.1
- 그리퍼: 열린 상태 고정

## 보상 구조

### APPROACH 단계
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 거리 페널티 | -2.0 | distance * scale |
| 진행 보상 | +5.0 | (prev_dist - dist) * scale |
| 단계 전환 보너스 | +10.0 | ALIGN 진입 시 |

### ALIGN 단계
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 거리 유지 페널티 | -1.0 | distance * scale |
| 정렬 보상 | +2.0 | (-dot - 0.5) / 0.5 * scale |
| 단계 전환 보너스 | +10.0 | GRASP 진입 시 |

### GRASP 단계
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 거리 페널티 | -3.0 | distance * scale |
| 정렬 유지 보상 | +1.0 | -dot * scale |
| 성공 보너스 | +100.0 | 성공 시 |

### 공통
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 액션 페널티 | -0.001 | action^2 * scale |
| 반대 방향 페널티 | -1.0 | dot > 0일 때 |

## 환경 설정

### 펜 위치 랜덤화
```python
pen_pos_range = {
    "x": (0.3, 0.5),
    "y": (-0.15, 0.15),
    "z": (0.25, 0.35),
}
```

### 펜 방향 (1단계: 수직 고정)
```python
pen_rot_range = {
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
```

### 2단계 (나중에 활성화)
```python
pen_rot_range = {
    "roll": (-0.5, 0.5),    # ±30도
    "pitch": (-0.5, 0.5),   # ±30도
    "yaw": (-3.14, 3.14),   # 전체 회전
}
collect_data = True  # Feasibility 데이터 수집
```

## 학습 하이퍼파라미터

```python
train_cfg = {
    "num_steps_per_env": 24,
    "max_iterations": 5000,
    "save_interval": 100,
    "policy": {
        "init_noise_std": 0.3,
        "actor_hidden_dims": [256, 256, 128],
        "critic_hidden_dims": [256, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.01,
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
    },
}
```

## 학습 실행

```bash
source ~/isaacsim_env/bin/activate
cd ~/IsaacLab
python pen_grasp_rl/scripts/train_direct.py --headless --num_envs 4096 --max_iterations 5000
```

## 테스트 실행

```bash
python pen_grasp_rl/scripts/play_direct.py --checkpoint /path/to/model.pt --num_envs 50
```

---

## 학습 기록

### 실험 1: 1단계 기본 학습

**날짜**: 2024-12-17

**설정**:
- 펜 위치: 랜덤
- 펜 방향: 수직 고정
- num_envs: 4096
- max_iterations: (진행 중)

**결과**:
| Iteration | Mean Reward | APPROACH | ALIGN | GRASP | Success |
|-----------|-------------|----------|-------|-------|---------|
| 100 | | | | | |
| 500 | | | | | |
| 1000 | | | | | |

**관찰**:
- (학습 후 기록)

---

## 다음 단계

1. [ ] 1단계 학습 완료 및 성공률 확인
2. [ ] 성공률 > 30% 달성 시 2단계 진행
3. [ ] 2단계: 펜 방향 랜덤화 + Feasibility 데이터 수집
4. [ ] Feasibility Classifier 학습
