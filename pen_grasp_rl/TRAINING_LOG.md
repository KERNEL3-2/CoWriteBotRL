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
