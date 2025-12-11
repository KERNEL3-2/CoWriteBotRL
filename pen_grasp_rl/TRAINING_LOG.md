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
- [ ] 수정된 환경 테스트 (`play.py`)
- [ ] 새 환경으로 학습 (10000 iteration, resume 기능 사용)
- [ ] Docker 환경 구축
