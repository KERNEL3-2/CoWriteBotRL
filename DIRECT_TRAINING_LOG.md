# E0509 Direct 환경 학습 로그

## 환경 개요

| 항목 | 값 |
|------|-----|
| 환경 타입 | Direct (DirectRLEnv) |
| 로봇 | Doosan E0509 + RH-P12-RN-A 그리퍼 |
| 작업 | 펜 캡 접근 + Z축 정렬 |
| 학습 방식 | PPO (RSL-RL) |

## 상태 머신 버전 이력

### V3 (2024-12-18) - 현재 (Pre-grasp 방식)
```
PRE_GRASP (펜캡 위 7cm + 정렬)
    ↓ 거리 < 3cm & dot < -0.95
DESCEND (정렬 유지하며 수직 하강)
    ↓ 거리 < 2cm & dot < -0.95
SUCCESS!
```

**핵심 변경**:
- Pre-grasp 위치에서 충분히 정렬 후 수직 하강
- 지수적 정렬 보상 (dot < -0.9부터 급격히 증가)
- 특이점 회피 페널티 (joint 3, 5)
- 더 엄격한 정렬 조건 (dot < -0.95 = 약 18도 이내)

### V2 (2024-12-17) - Deprecated
```
ALIGN (정렬) ← 먼저 정렬!
    ↓ dot < -0.8
APPROACH (접근) + 정렬 유지
    ↓ 거리 < 10cm
GRASP (잡기)
    ↓ 거리 < 2cm & dot < -0.9
SUCCESS!
```

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

### V3 (현재) - PRE_GRASP → DESCEND + 지수적 정렬 보상

#### 지수적 정렬 보너스
```python
# dot < -0.9부터 증가 (밸런스 조정됨)
exponential_bonus = exp((−dot − 0.9) × 10)
```

| dot 값 | 오차 각도 | 지수 보너스 |
|--------|-----------|-------------|
| -0.9   | ~25°      | 1.0 (기준)  |
| -0.95  | ~18°      | 1.6         |
| -0.99  | ~8°       | 2.5         |
| -1.0   | 0°        | 2.7         |

#### PRE_GRASP 단계
| 보상 | 스케일 | 설명 |
|------|--------|------|
| pre-grasp 거리 페널티 | -8.0 | 펜캡 위 7cm 위치까지 거리 |
| 진행 보상 | +20.0 | 거리 감소량 |
| 기본 정렬 보상 | +1.5 | (-dot - 0.5) / 0.5 |
| 지수적 정렬 보너스 | +0.3 | exponential_bonus |
| 단계 전환 보너스 | +15.0 | DESCEND 진입 시 |

#### DESCEND 단계
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 펜캡 거리 페널티 | -10.0 | 펜캡까지 거리 |
| 기본 정렬 유지 보상 | +2.0 | -dot |
| 지수적 정렬 보너스 | +0.3 | exponential_bonus |
| 하강 진행 보상 | +15.0 | 거리 감소량 |
| 정렬 풀림 페널티 | -5.0 | dot > -0.9일 때 |
| 성공 보너스 | +100.0 | 성공 시 |

#### 특이점 회피 페널티
| 관절 | 조건 | 페널티 |
|------|------|--------|
| Joint 3 (팔꿈치) | \|pos\| < 0.15 rad | -1.0 |
| Joint 5 (손목) | \|pos\| < 0.15 rad | -1.0 |

#### 공통
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 액션 페널티 | -0.01 | action^2 * scale |
| 반대 방향 페널티 | -2.0 | dot > 0일 때 |

> **밸런스 조정 (2024-12-18)**: 정렬 보상이 거리/진행 보상보다 30~144배 컸던 문제 수정. 거리/진행 보상 강화, 지수적 정렬 보상 축소하여 약 1:1 비율로 조정.

---

### V2 (Deprecated) 보상 구조

<details>
<summary>V2 보상 구조 (클릭해서 펼치기)</summary>

#### ALIGN 단계 (1단계)
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 정렬 보상 | +2.0 | (-dot - 0.5) / 0.5 * scale |
| 약한 거리 페널티 | -1.0 | distance * scale |
| 단계 전환 보너스 | +10.0 | APPROACH 진입 시 |

#### APPROACH 단계 (2단계)
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 거리 페널티 | -2.0 | distance * scale |
| 진행 보상 | +5.0 | (prev_dist - dist) * scale |
| 정렬 유지 보상 | +1.0 | (-dot - 0.7) * scale |
| 단계 전환 보너스 | +10.0 | GRASP 진입 시 |

#### GRASP 단계 (3단계)
| 보상 | 스케일 | 설명 |
|------|--------|------|
| 거리 페널티 | -3.0 | distance * scale |
| 정렬 유지 보상 | +1.0 | -dot * scale |
| 성공 보너스 | +100.0 | 성공 시 |

</details>

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

**날짜**: 2024-12-17 ~ 2024-12-18

**변경 사항**:
1. 단계 순서: ALIGN → APPROACH → GRASP
2. APPROACH 단계에 정렬 유지 보상 추가
3. 먼저 정렬 → 정렬 유지하며 접근

**설정**:
- 펜 위치: 랜덤
- 펜 방향: 수직 고정
- num_envs: 4096
- max_iterations: 33000

**결과**:
| Iteration | Mean Reward | Episode Length | Noise Std | 비고 |
|-----------|-------------|----------------|-----------|------|
| 0 | -58.8 | 70 | - | 초기 |
| 5827 | **318.0** | 359 | - | 최고 리워드 (타임아웃) |
| 33000 | 129.0 | 24.1 | - | 수렴 (빠른 성공) |

**학습 그래프**:

![E0509 Direct V2 Training](images/e0509_direct_v2_training.png)

**핵심 분석**:

| 지표 | 최고점 (step 5827) | 수렴점 (step 33000) |
|------|-------------------|---------------------|
| 총 리워드 | 318 | 129 |
| 에피소드 길이 | 359 (타임아웃) | 24 |
| **스텝당 리워드** | **0.88** | **5.35** |

> **해석**: 리워드 감소는 "성능 하락"이 아님!
> - 최고점: 타임아웃까지 헤매면서 리워드 누적
> - 수렴점: 24스텝 만에 빠르게 성공 → 효율성 6배 증가

**문제점 발견**:
- Z축 정렬이 완벽하지 않음 (대략적으로만 맞춤)
- 정렬 조건이 느슨함 (dot < -0.8 = 36도 허용)
- 접근 보상(5.0)이 정렬 유지 보상(1.0)보다 5배 커서 정렬보다 접근 우선

**결론**: Pre-grasp 방식 + 지수적 정렬 보상 필요 → V3로 개선

---

### 실험 3: V3 (PRE_GRASP → DESCEND + 지수적 정렬 보상)

**날짜**: 2024-12-18

**핵심 변경 사항**:
1. **Pre-grasp 전략**: 펜캡 위 7cm에서 충분히 정렬 후 수직 하강
2. **지수적 정렬 보상**: dot < -0.9부터 보상 급격히 증가
3. **특이점 회피**: Joint 3, 5가 0 근처일 때 페널티
4. **엄격한 조건**: dot < -0.95 (18도 이내)

**설정**:
- 펜 위치: 랜덤
- 펜 방향: 수직 고정
- num_envs: 4096
- max_iterations: 10000 (예정)

**결과**:
| Iteration | Mean Reward | Episode Length | 비고 |
|-----------|-------------|----------------|------|
| 0 | ~0 | 359 | 초기 |
| 9650 | **771.77** | 359 | 최고 리워드 |
| 9999 | 753.44 | 359 | 최종 |

**학습 그래프**:

![E0509 Direct V3 Training (10000 iterations)](images/e0509_direct_10000_training.png)

**문제점 발견** (시뮬레이션 테스트 결과):
1. **로봇 링크 겹침**: self-collision 발생
2. **과격한 움직임**: action_scale 0.1이 너무 큼
3. **특이점 페널티 부작용**: 오히려 이상한 자세 유도
4. **Episode Length = 359**: 여전히 타임아웃 (성공 종료 없음)

**해결책 → V4로 개선**

---

### 실험 4: V4 (작업 공간 기반 관절 한계 + 액션 스케일 축소)

**날짜**: 2024-12-18

**핵심 변경 사항**:

1. **펜 소환 위치 확대**:
   ```python
   pen_pos_range = {
       "x": (0.3, 0.5),      # 유지
       "y": (-0.20, 0.20),   # -0.15~0.15 → -0.20~0.20
       "z": (0.20, 0.50),    # 0.25~0.35 → 0.20~0.50
   }
   ```

2. **작업 공간 기반 관절 한계** (IK 샘플링으로 계산):
   ```python
   WORKSPACE_JOINT_LIMITS_RAD = [
       (-0.96, 0.96),    # joint_1: ±55° (base rotation)
       (-1.31, 0.61),    # joint_2: -75° ~ 35° (shoulder)
       (0.87, 2.88),     # joint_3: 50° ~ 165° (elbow - 특이점 0° 회피)
       (-0.79, 0.79),    # joint_4: ±45° (wrist 1)
       (1.05, 2.09),     # joint_5: 60° ~ 120° (wrist 2 - 특이점 0° 회피)
       (-0.79, 0.79),    # joint_6: ±45° (wrist 3)
   ]
   ```

3. **액션 스케일 축소**: 0.1 → 0.05 (과격한 움직임 방지)

4. **특이점 페널티 비활성화**: -1.0 → 0.0 (관절 한계로 대체)

**관절 한계 계산 방법**:
- 작업 공간(펜 위치 범위 + pre-grasp 7cm)에서 5000개 포인트 IK 샘플링
- 유효한 IK 솔루션들의 관절 범위 + 15° 마진
- Joint 3, 5가 자연스럽게 특이점(0°) 회피

**설정**:
- 펜 위치: 확대된 범위
- 펜 방향: 수직 고정
- num_envs: 4096
- max_iterations: TBD

**결과**:
| Iteration | Mean Reward | Episode Length | 비고 |
|-----------|-------------|----------------|------|
| 0 | -51.26 | 22 | 초기 |
| 3221 | **269.67** | 359 | 최고 리워드 |
| 3410 | 264.64 | 359 | 최종 (정체) |

**문제점 발견** (시뮬레이션 테스트):
- 관절 한계가 너무 제한적 (특히 joint_5: 60~120도)
- 그리퍼가 위만 바라보고 아래를 향하지 못함
- 펜에 접근 불가

---

### 실험 5: V5 (관절 한계 확장)

**날짜**: 2024-12-18

**핵심 변경 사항**:

관절 한계 확장 (그리퍼가 아래를 향할 수 있도록):

| 관절 | V4 | V5 |
|------|-----|-----|
| joint_1 | ±55° | **±90°** |
| joint_2 | -75°~35° | **-90°~45°** |
| joint_3 | 50°~165° | **30°~150°** |
| joint_4 | ±45° | **±90°** |
| joint_5 | 60°~120° | **30°~150°** |
| joint_6 | ±45° | **±90°** |

**결과**:
| Iteration | Mean Reward | Episode Length | 비고 |
|-----------|-------------|----------------|------|
| 0 | -1366.51 | ~20 | 초기 |
| 2881 | **776.66** | 359 | 최고 리워드 |
| 4058 | 751.84 | 359 | 최종 (정체) |

**학습 그래프**:

![E0509 Direct V5 Training](images/e0509_direct_v5_training.png)

**문제점 발견** (시뮬레이션 테스트):
- **그리퍼가 아래에서 위로 올려다보는 자세**로 학습됨
- 관절 한계를 확장해도 "위에서 아래로 내려다보기" 자세 유도 어려움
- Joint space 제어의 한계: 원하는 자세를 직접 지정할 수 없음
- **Episode Length = 359**: 여전히 타임아웃 (성공 종료 없음)

**결론**: Joint space 제어의 한계 → **IK (Task Space) 제어**로 전환 필요

---

## IK 환경 (Task Space Control)

### 개요

Joint space 제어의 한계를 극복하기 위해 **IK (Inverse Kinematics) 기반 Task Space 제어** 환경 추가.

| 항목 | Direct 환경 (기존) | IK 환경 (신규) |
|------|-------------------|----------------|
| 제어 방식 | Joint space (관절 각도) | Task space (EE 위치/자세) |
| 액션 공간 | 6D (Δjoint1~6) | 6D (Δx, Δy, Δz, Δroll, Δpitch, Δyaw) |
| 관절 한계 | 작업 공간 기반 커스텀 | 로봇 기본 (soft limits) |
| 초기 자세 | 기본 | **위에서 내려다보기** (joint_5=90°) |
| 특이점 회피 | 관절 한계로 강제 | IK solver 자연 회피 |

### IK 환경 특징

**1. Task Space 액션**:
```python
action_space = 6  # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
action_scale = 0.02  # 미터/라디안 단위
```

**2. 초기 자세 (위에서 내려다보기)**:
```python
init_state = {
    "joint_1": 0.0,
    "joint_2": -0.5,      # 어깨 약간 앞으로
    "joint_3": 1.0,       # 팔꿈치 굽힘
    "joint_4": 0.0,
    "joint_5": 1.57,      # 손목 90도 - 그리퍼가 아래를 향함
    "joint_6": 0.0,
}
```

**3. DifferentialIKController 설정**:
```python
ik_controller = DifferentialIKController(
    cfg=DifferentialIKControllerCfg(
        command_type="pose",      # 위치 + 자세 제어
        use_relative_mode=True,   # Delta 제어
        ik_method="dls",          # Damped Least Squares
        ik_params={"lambda_val": 0.05},
    ),
    num_envs=num_envs,
    device=device,
)
```

**4. 관찰 공간 확장**:
| 관찰 | 차원 | 설명 |
|------|------|------|
| (기존 Direct와 동일) | 27 | joint, grasp, cap, rel, axes |
| ee_pos | 3 | End-effector 위치 (root frame) |
| ee_axis_angle | 3 | End-effector 자세 (axis-angle) |
| **총합** | **33** | |

**5. 보상 함수**: Direct 환경과 **동일** (특이점 페널티만 제거)

### IK 환경 장점

1. **직관적인 액션 공간**: "앞으로 이동", "아래로 내려가기" 등 직접 지정
2. **초기 자세 강제**: 학습 시작부터 원하는 자세로 시작
3. **관절 한계 문제 해소**: IK가 실행 가능한 자세 계산
4. **Sim2Real 용이**: 실제 로봇에서도 Task space 제어 가능

### 실행 방법

**학습**:
```bash
source ~/isaacsim_env/bin/activate
cd ~/IsaacLab
python pen_grasp_rl/scripts/train_ik.py --headless --num_envs 4096 --max_iterations 5000
```

**테스트**:
```bash
python pen_grasp_rl/scripts/play_ik.py --checkpoint /path/to/model.pt --num_envs 50
```

### 파일 구조

```
pen_grasp_rl/
├── envs/
│   ├── e0509_direct_env.py   # 기존 Joint space 환경
│   └── e0509_ik_env.py       # 신규 Task space 환경
└── scripts/
    ├── train_direct.py
    ├── play_direct.py
    ├── train_ik.py           # 신규
    └── play_ik.py            # 신규
```

---

## 다음 단계

1. [x] V1 학습 분석 완료
2. [x] V2 학습 및 분석 완료
3. [x] V3 (Pre-grasp + 지수적 정렬 보상) 코드 수정
4. [x] V4 학습 (작업 공간 기반 관절 한계) - 관절 한계 너무 제한적
5. [x] V5 학습 (관절 한계 확장) - 그리퍼가 아래에서 위로 올려다봄 (실패)
6. [x] **IK 환경 추가** (Task Space Control)
7. [ ] **IK 환경 학습 실행 및 결과 확인**
8. [ ] 성공률 > 50% & 정렬 정밀도 확인 후 2단계 진행
9. [ ] 2단계: 펜 방향 랜덤화 + Feasibility 데이터 수집
10. [ ] Feasibility Classifier 학습 (MLP)
11. [ ] Sim2Real 전이 테스트

---

## 변경 이력

| 날짜 | 변경 | 커밋 |
|------|------|------|
| 2024-12-17 | Direct 환경 초기 구현 (V1) | `3472b87` |
| 2024-12-17 | 단계 순서 변경 (V2): ALIGN → APPROACH → GRASP | `c3fe5c3` |
| 2024-12-18 | Pre-grasp 방식 (V3) + 지수적 정렬 보상 + 특이점 회피 | - |
| 2024-12-18 | V4: 작업 공간 기반 관절 한계 + action_scale 축소 + 특이점 페널티 제거 | - |
| 2024-12-18 | V5: 관절 한계 확장 (그리퍼 아래 향하도록) | `1d71d76` |
| 2024-12-18 | **IK 환경 추가**: Task Space Control (e0509_ik_env.py, train_ik.py, play_ik.py) | (현재) |
