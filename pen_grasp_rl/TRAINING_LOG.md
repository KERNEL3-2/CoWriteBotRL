# Pen Grasp RL 학습 기록

## TensorBoard 실행 방법
```bash
tensorboard --logdir=~/IsaacLab/logs/pen_grasp
# 브라우저에서 http://localhost:6006 접속
```

## 주요 지표 해석

| 지표 | 의미 | 좋은 신호 |
|------|------|-----------|
| Episode_Reward/iteration_wise | 전체 보상 | 📈 증가 |
| Episode_Reward/distance | 펜과의 거리 보상 | 📈 증가 |
| Episode_Reward/pen_lifted | 펜 들어올리기 보상 | 📈 증가 (0보다 커야 함) |
| Episode_Termination/time_out | 시간 초과 종료 비율 | 학습 초기엔 높음 |
| Episode_Termination/pen_dropped | 펜 떨어짐 종료 비율 | 낮을수록 좋음 |

---

## 학습 기록

### 2025-12-11 첫 번째 학습 (1000 iteration)
- **설정**: num_envs=4096, max_iterations=1000
- **소요 시간**: 약 15분
- **결과**:
  - distance 보상: 증가 추세 → 로봇이 펜 쪽으로 이동 학습 중
  - pen_lifted 보상: 거의 0 → 아직 펜 잡기 미성공
- **결론**: 더 많은 iteration 필요

### 2025-12-11 두 번째 학습 (3000 iteration)
- **설정**: num_envs=4096, max_iterations=3000
- **소요 시간**: 약 45분
- **결과**:
  - distance 보상: 지속 증가 → 로봇이 펜에 더 가까이 접근
  - pen_lifted 보상: 여전히 0 근처 → 펜 잡기 미성공
  - Episode_Termination: time_out이 대부분
- **분석**: play.py로 동작 확인 결과:
  - 펜이 z=0 평면(바닥)에서 소환됨
  - 그리퍼가 펜에 접근하나 잡는 동작 미완성
  - 펜을 들어올리는 것보다 잡는 것이 우선 필요

---

## 환경 수정 기록

### 2025-12-11 환경 개선 v2

#### 변경 목표
1. 펜을 공중에 띄워서 (사람이 손으로 들고 있는 상황 시뮬레이션)
2. 펜 자세를 랜덤하게 부여
3. 그리퍼가 펜의 cap 부분(point b)을 향해 접근하도록
4. pen_lifted 보상 제거 (잡기 먼저, 들기는 나중에)

#### 코드 수정 사항

**1. 펜 설정 변경 (`pen_grasp_env.py`)**
```python
# 이전: 바닥에서 소환, 중력 적용
pos=(0.4, 0.0, 0.0)

# 변경: 공중에서 소환, 중력 비활성화, kinematic
pen: RigidObjectCfg = RigidObjectCfg(
    spawn=sim_utils.CylinderCfg(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.3),  # z=0.3m 공중
    ),
)
```

**2. 펜 랜덤 자세 (`_reset_idx` 함수)**
```python
# 랜덤 orientation 생성
roll = torch.rand(num_resets, device=self.device) * 1.0 - 0.5   # ±0.5 rad (약 ±30°)
pitch = torch.rand(num_resets, device=self.device) * 1.0 - 0.5  # ±0.5 rad
yaw = torch.rand(num_resets, device=self.device) * 6.28 - 3.14  # 360° 랜덤
```

**3. 새로운 관측 함수**
- `pen_orientation_obs`: 펜의 quaternion 자세
- `pen_cap_pos_obs`: 펜 cap(point b) 위치 계산
- `relative_ee_cap_obs`: 그리퍼와 cap 간의 상대 위치

**4. 보상 함수 변경**
```python
# 제거: pen_lifted_reward (잡기 전에 들기 보상은 불필요)

# 추가: distance_ee_cap_reward
# - 펜 중심이 아닌 cap(point b) 위치로 접근 유도
# - cap 위치 = pen_pos + pen_orientation * (0, 0, -PEN_LENGTH/2)
```

**5. ObservationGroup 업데이트**
```python
"policy": ObservationGroup(
    terms=[
        ObservationTerm("joint_pos", ...),
        ObservationTerm("joint_vel", ...),
        ObservationTerm("ee_pos", ...),
        ObservationTerm("pen_pos", ...),
        ObservationTerm("pen_orientation", ...),    # 추가
        ObservationTerm("relative_ee_pen", ...),
        ObservationTerm("relative_ee_cap", ...),    # 추가
        ObservationTerm("gripper_state", ...),
    ]
)
```

#### 다음 단계
- [x] 수정된 환경 테스트 (`play.py`)
- [x] Docker 환경 구축
- [ ] 새 환경으로 학습 (10000 iteration) - 진행 중

---

### 2025-12-11 Docker 환경 구축 및 새 노트북 학습

#### Docker 환경 구축
- Isaac Lab 공식 Docker 사용 (`nvcr.io/nvidia/isaac-sim`)
- `container.py` 스크립트로 관리 (docker compose 직접 사용 시 환경변수 오류)
- 볼륨 마운트: pen_grasp_rl, logs, e0509_gripper_isaac

#### USD 파일 참조 문제 해결
- `first_control.usd`가 `/workspace/e0509_gripper_isaac/e0509_gripper_isaac.usd` 참조
- `e0509_gripper_isaac` 폴더를 레포에 추가하고 Docker 볼륨 마운트로 해결

#### 펜 스폰 범위 수정 (실제 작업 공간 기준)
```python
# 실제 로봇 작업 범위 측정값 기준
"pose_range": {
    "x": (-0.2, 0.2),      # 로봇 기준 0.3~0.7m
    "y": (-0.3, 0.3),      # 좌우 ±30cm
    "z": (-0.2, 0.2),      # 높이 0.1~0.5m
}
```

#### play.py 마커 추가
- Tip (파란색): 필기 끝 (pen_pos + axis * half_len)
- Cap (빨간색): 그리퍼가 잡아야 할 곳 (pen_pos - axis * half_len)

#### 새 노트북 학습 시작
- **하드웨어**: RTX 5080 (16GB VRAM)
- **설정**: num_envs=8192, max_iterations=10000
- **상태**: 학습 진행 중
- **TensorBoard**: 컨테이너 내부에서 실행 권장

#### 관련 문서
- `DOCKER_GUIDE.md`: Docker 환경 설정 가이드
- `docker_setup.sh`: 컨테이너 내 의존성 설치 스크립트

---

### 2025-12-11 Grasp Point 및 보상함수 개선

#### 문제 분석
- 기존 gripper center가 손가락 끝 중앙이라 그리퍼 open/close 상태에 따라 이동
- 보상함수가 펜에 접근만 유도하고, 정렬(orientation)은 고려하지 않음

#### 변경 사항

**1. Grasp Point 계산 방식 변경 (`pen_grasp_env.py`)**
```python
def get_grasp_point(robot: Articulation) -> torch.Tensor:
    """Get ideal grasp point: (l1+r1)/2 center + 2cm along finger direction.

    This point is stable regardless of gripper open/close state.
    """
    # [7] l1, [8] r1 = 손가락 베이스
    # [9] l2, [10] r2 = 손가락 끝
    l1 = robot.data.body_pos_w[:, 7, :]
    r1 = robot.data.body_pos_w[:, 8, :]
    l2 = robot.data.body_pos_w[:, 9, :]
    r2 = robot.data.body_pos_w[:, 10, :]

    base_center = (l1 + r1) / 2.0
    tip_center = (l2 + r2) / 2.0
    finger_dir = tip_center - base_center
    finger_dir = finger_dir / (torch.norm(finger_dir, dim=-1, keepdim=True) + 1e-6)

    return base_center + finger_dir * 0.02  # 2cm along finger direction
```

**2. z축 정렬 보상함수 추가**
```python
def z_axis_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning gripper z-axis with pen z-axis.

    Only gives reward when:
    1. Gripper is close to pen cap (within 5cm)
    2. Z-axes are nearly parallel (dot product > 0.9)
    """
    # ... pen z-axis, gripper z-axis 계산 ...

    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # Only reward when nearly parallel (dot > 0.9)
    alignment_reward = torch.clamp(dot_product - 0.9, min=0.0) * 10.0

    # Only apply when close to cap (within 5cm)
    distance_factor = torch.clamp(1.0 - distance_to_cap / 0.05, min=0.0)

    return alignment_reward * distance_factor
```

**3. 현재 보상함수 구성**
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 |
| `z_axis_alignment` | 0.5 | z축 정렬 (캡 5cm 이내 + dot>0.9 일때만) |
| `action_rate` | 0.1 | 액션 크기 페널티 |

**4. play.py 마커 개선**
- Cap (빨강): 펜 캡 위치 (목표)
- Grasp Point (초록): 그리퍼 잡기 위치
- Pen z-axis (파랑): 펜 중심에서 z축 방향 (5개 점, 15cm)
- Gripper z-axis (노랑): grasp point에서 link_6 z축 방향 (5개 점, 15cm)

#### 커리큘럼 러닝 전략
- **Phase 1 (현재)**: 펜 kinematic, 위치+정렬 학습
- **Phase 2 (추후)**: 펜 dynamic, 잡기 동작 학습
- 기존 학습된 "접근+정렬" 정책이 Phase 2에서 fine-tuning으로 활용됨

---

### 2025-12-11 추가 개선 사항

#### 1. 로봇 USD에서 불필요한 펜 제거
- `first_control.usd` 내부에 펜 오브젝트가 포함되어 있었음
- Isaac Sim에서 USD 열어서 Robot/Pen 삭제 후 저장
- 이전 학습에서 이 펜이 물리적 노이즈로 작용했을 가능성 있음

#### 2. 펜 자세 랜덤화 범위 확대
```python
# 이전: 거의 수직으로만 스폰
"roll": (-0.5, 0.5),   # ±30°
"pitch": (-0.5, 0.5),  # ±30°

# 변경: 완전 랜덤 (뒤집힘 포함)
"roll": (-3.14, 3.14),   # ±180°
"pitch": (-3.14, 3.14),  # ±180°
```

#### 3. 바닥 충돌 페널티 추가 (실제 접촉력 기반)
```python
def floor_collision_penalty(env) -> torch.Tensor:
    """로봇 링크가 바닥에 닿으면 페널티."""
    # 접촉력 z성분 확인 (바닥이 위로 밀어올림)
    contact_forces_z = robot.data.net_contact_forces_w[:, 2:11, 2]
    link_z = robot.data.body_pos_w[:, 2:11, 2]

    # 바닥 충돌: 위쪽 접촉력 > 1N AND 링크 z < 0.1m
    floor_contact = ((contact_forces_z > 1.0) & (link_z < 0.1)).any(dim=-1)
    return -floor_contact.float()
```

#### 4. 현재 보상함수 구성
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 |
| `z_axis_alignment` | 0.5 | z축 정렬 (5cm 이내 + dot>0.9) |
| `floor_collision` | 1.0 | 바닥 실제 충돌 시 -1 페널티 |
| `action_rate` | 0.1 | 액션 크기 페널티 |

---

### 2025-12-12 z_axis_alignment 보상함수 개선

#### 50,000 iteration 학습 결과 분석
- **distance_to_cap**: 0.96 (성공적으로 펜 캡 접근 학습)
- **z_axis_alignment**: ~0 (정렬 보상 거의 없음)
- **floor_collision**: -0.001 (바닥 충돌 거의 없음)

#### 문제점
기존 z_axis_alignment 조건이 너무 까다로움:
- 5cm 이내 접근 AND dot product > 0.9 일때만 보상
- 로봇이 접근은 하지만 정확한 각도로 정렬되는 순간이 거의 없어 보상을 못 받음

#### 해결책: 거리 기반 가중치 적용
```python
def z_axis_alignment_reward(env) -> torch.Tensor:
    # 기존: 5cm 이내 + dot > 0.9 일때만 보상
    # 변경: 거리와 무관하게 정렬 보상, 단 가까울수록 가중치 증가

    # dot product: 양수만 보상 (캡 방향만 허용, 팁 방향은 보상 0)
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)
    alignment_score = torch.clamp(dot_product, min=0.0)  # 0 ~ 1

    # 거리 가중치: 가까울수록 높음
    # 5cm: weight = 10, 50cm: weight ≈ 1.8
    distance_weight = 1.0 / (distance_to_cap + 0.05)

    return alignment_score * distance_weight * 0.1
```

#### 개선 효과
- 멀리서도 방향 맞추면 작은 보상 (방향 학습 힌트)
- 가까이 가면서 정렬하면 큰 보상
- 접근 + 정렬 동시 학습 유도

#### 다음 단계
- [x] 새로운 보상함수로 학습 실행
- [ ] TensorBoard에서 z_axis_alignment 보상 증가 확인

---

### 2025-12-12 Phase 2 구현: 펜 충돌 및 그립 동작

#### 50,000 iteration 학습 결과 추가 분석
- play.py 실행 결과, 로봇이 펜 **팁** 방향으로 접근하고 있었음
- **원인**: z_axis_alignment에서 `torch.clamp(dot_product, min=0.0)` 사용
  - dot=+1.0 (같은 방향)일 때 보상 → 잘못된 방향
  - 실제로는 dot=-1.0 (반대 방향)일 때 보상해야 함 (그리퍼가 캡을 마주보며 접근)

#### z_axis_alignment 방향 수정
```python
# 이전: 같은 방향일 때 보상 (틀림)
alignment_score = torch.clamp(dot_product, min=0.0)

# 수정: 반대 방향일 때 보상 (올바름)
alignment_score = torch.clamp(-dot_product, min=0.0)
```

#### Phase 2 변경 사항

**1. 펜 모델 변경**
- 팀원이 모델링한 pen.usd 적용 (뚜껑 없는 상태, 117mm)
- PEN_LENGTH: 0.1207 → 0.117

**2. 펜 충돌 활성화**
```python
# 이전: kinematic_enabled=True (고정)
# 변경: kinematic_enabled=False (충돌 가능)
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=True,      # 공중에 떠있음
    kinematic_enabled=False,   # 그리퍼에 맞으면 밀림
)
```

**3. 새로운 Observation 추가**
```python
gripper_state = ObsTerm(func=gripper_state_obs)  # 그리퍼 열림/닫힘 상태 (0~1)
```

**4. 새로운 보상함수 추가**
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `pen_displacement_penalty` | 1.0 | 펜을 치면 속도에 비례한 페널티 |
| `grasp_success_reward` | 2.0 | 3cm 이내 + 정렬 + 그리퍼 닫힘 시 큰 보상 |

```python
def pen_displacement_penalty(env) -> torch.Tensor:
    """펜 속도에 비례한 페널티 (펜을 함부로 치지 않도록)"""
    pen_vel = pen.data.root_lin_vel_w
    vel_magnitude = torch.norm(pen_vel, dim=-1)
    return -vel_magnitude * 0.5

def grasp_success_reward(env) -> torch.Tensor:
    """성공적인 그립 자세 달성 시 보상"""
    close_enough = (distance_to_cap < 0.03).float()  # 3cm 이내
    aligned = (dot_product < -0.8).float()           # 반대 방향 정렬
    gripper_closed = (gripper_pos > 0.5).all().float()  # 그리퍼 닫힘
    return close_enough * aligned * gripper_closed * 5.0
```

**5. Termination 조건 변경**
```python
# 이전: 펜 z < 0.01 (바닥에 떨어지면 종료)
# 변경: 펜이 초기 위치에서 15cm 이상 이탈하면 종료
def pen_dropped_termination(env) -> torch.Tensor:
    pen_pos = pen.data.root_pos_w - env.scene.env_origins
    init_pos = torch.tensor([0.5, 0.0, 0.3])
    displacement = torch.norm(pen_pos - init_pos, dim=-1)
    return displacement > 0.15  # 어느 방향이든 15cm 이상 밀리면 실패
```

**6. play.py cap 위치 수정**
```python
# 이전: cap_pos = pen_pos - pen_axis_world * half_len (틀림)
# 수정: cap_pos = pen_pos + pen_axis_world * half_len (올바름)
```

#### 현재 보상함수 구성
| 보상함수 | weight | 설명 |
|---------|--------|------|
| `distance_to_cap` | 1.0 | grasp point → 펜 캡 거리 |
| `z_axis_alignment` | 0.5 | z축 반대 방향 정렬 (거리 가중치) |
| `floor_collision` | 1.0 | 바닥 충돌 페널티 |
| `pen_displacement` | 1.0 | 펜 밀림 페널티 |
| `grasp_success` | 2.0 | 성공적 그립 보상 |
| `action_rate` | 0.1 | 액션 크기 페널티 |

#### 커리큘럼 러닝 진행 상황
- **Phase 1**: 펜 고정 (kinematic=True), 접근+정렬 학습 → 완료
- **Phase 2 (현재)**: 펜 충돌 활성화, 그립 동작 학습 → 준비 완료

#### 다음 단계
- [ ] Phase 2 학습 실행
- [ ] 펜을 밀지 않고 조심스럽게 접근하는지 확인
- [ ] grasp_success 보상이 발생하는지 확인
