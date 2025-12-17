# E0509 Direct 환경 학습 로그

## 환경 개요

| 항목 | 값 |
|------|-----|
| 환경 타입 | Direct (DirectRLEnv) |
| 로봇 | Doosan E0509 + RH-P12-RN-A 그리퍼 |
| 작업 | 펜 캡 접근 + Z축 정렬 |
| 학습 방식 | PPO (RSL-RL) |

## 상태 머신 (3단계)

### V2 (2024-12-17 변경) - 현재
```
ALIGN (정렬) ← 먼저 정렬!
    ↓ dot < -0.8
APPROACH (접근) + 정렬 유지
    ↓ 거리 < 10cm
GRASP (잡기)
    ↓ 거리 < 2cm & dot < -0.9
SUCCESS!
```

**변경 이유**: 가까이서 정렬하기 어려움 → 멀리서 정렬 후 정렬 유지하며 접근

### V1 (기존) - Deprecated
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

### V2 (현재) - ALIGN → APPROACH → GRASP

#### ALIGN 단계 (1단계)
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 정렬 보상 | +2.0 | (-dot - 0.5) / 0.5 * scale |
| 약한 거리 페널티 | -1.0 | distance * scale (너무 멀어지지 않게) |
| 단계 전환 보너스 | +10.0 | APPROACH 진입 시 |

#### APPROACH 단계 (2단계)
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 거리 페널티 | -2.0 | distance * scale |
| 진행 보상 | +5.0 | (prev_dist - dist) * scale |
| **정렬 유지 보상** | +1.0 | (-dot - 0.7) * scale (V2 추가) |
| 단계 전환 보너스 | +10.0 | GRASP 진입 시 |

#### GRASP 단계 (3단계)
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

---

### 실험 1: V1 (APPROACH → ALIGN → GRASP)

**날짜**: 2024-12-17

**설정**:
- 펜 위치: 랜덤
- 펜 방향: 수직 고정
- num_envs: 4096
- max_iterations: 5000
- 로그 위치: `/home/fhekwn549/e0509_direct`

**결과**:
| Iteration | Mean Reward | Episode Length | Noise Std | 비고 |
|-----------|-------------|----------------|-----------|------|
| 0 | -38.6 | 359 | 0.30 | 초기값 |
| 1100 | 304.4 | 359 | 0.68 | 최종 |

**단계별 분포 (마지막)**:
| 단계 | 비율 | 설명 |
|------|------|------|
| APPROACH | 0% | 모두 통과 |
| ALIGN | ~100% | 여기서 막힘 |
| GRASP | 0% | 진입 실패 |

**문제점 분석**:
1. **Episode Length = 359 (max)**: 성공 종료 없음
2. **Noise Std 증가 (0.30 → 0.68)**: 불안정한 학습
3. **ALIGN 단계 정체**: 가까운 거리에서 정렬 어려움
4. **성공 조건 미달**: 거리 < 2cm & dot < -0.9 달성 못함

**원인**:
- 먼저 접근 후 정렬하면, 이미 가까운 거리에서 정렬해야 함
- 가까운 거리에서는 작은 움직임도 거리에 큰 영향
- 정렬하면서 거리가 멀어지는 악순환

**결론**: 단계 순서 변경 필요 (ALIGN 먼저)

---

### 실험 2: V2 (ALIGN → APPROACH → GRASP)

**날짜**: 2024-12-17

**변경 사항**:
1. 단계 순서: ALIGN → APPROACH → GRASP
2. APPROACH 단계에 정렬 유지 보상 추가
3. 먼저 정렬 → 정렬 유지하며 접근

**설정**:
- 펜 위치: 랜덤
- 펜 방향: 수직 고정
- num_envs: 4096
- max_iterations: 5000

**예상 효과**:
- 멀리서 정렬 (쉬움) → 정렬 유지하며 접근 (어렵지만 가능)
- ALIGN 단계 빠르게 통과
- APPROACH 단계에서 정렬 유지 보상으로 정렬 풀림 방지

**결과**:
| Iteration | Mean Reward | Episode Length | Noise Std | ALIGN | APPROACH | GRASP |
|-----------|-------------|----------------|-----------|-------|----------|-------|
| (학습 후 기록) | | | | | | |

**관찰**:
- (학습 후 기록)

---

## 다음 단계

1. [x] V1 학습 분석 완료
2. [x] V2 (단계 순서 변경) 코드 수정
3. [ ] V2 학습 실행 및 결과 확인
4. [ ] 성공률 > 30% 달성 시 2단계 진행
5. [ ] 2단계: 펜 방향 랜덤화 + Feasibility 데이터 수집
6. [ ] Feasibility Classifier 학습

---

## 변경 이력

| 날짜 | 변경 | 커밋 |
|------|------|------|
| 2024-12-17 | Direct 환경 초기 구현 (V1) | `3472b87` |
| 2024-12-17 | 단계 순서 변경 (V2): ALIGN → APPROACH → GRASP | `c3fe5c3` |
