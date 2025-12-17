# 펜 잡기 RL 코드 수정 가이드
## 2024-12-16 학습 정체 문제 해결

---

## 문제 상황

```
- 기존 checkpoint에서 reward 구조 변경 후 학습 재개
- 50,000 iter부터 시작 → 2,000 iter 추가 학습
- task_success: 0.3~0.5% (매우 낮음)
- reward 정체 (올라가지 않음)
```

---

## 발견된 문제점

### 문제 1: Checkpoint + Reward 변경 충돌

```
기존 checkpoint에서 이어서 학습 + reward 구조 변경
→ 기존 policy가 새 reward에 혼란
→ 새로 시작하는 것이 더 효율적
```

**해결**: 새로 학습 시작 (checkpoint 없이)

---

### 문제 2: z_axis_alignment_reward "절벽 보상" ⚠️ 핵심

**현재 코드 (Line 583):**
```python
alignment_reward = torch.clamp(-dot_product - 0.9, min=0.0) * 10.0
```

**문제점:**

| dot product | -dot - 0.9 | clamp | 보상 |
|-------------|------------|-------|------|
| -1.0 (완벽 정렬) | 0.1 | 0.1 | 1.0 |
| -0.95 | 0.05 | 0.05 | 0.5 |
| -0.9 | 0.0 | 0.0 | **0** |
| -0.8 | -0.1 | 0.0 | **0** |
| -0.5 | -0.4 | 0.0 | **0** |
| 0.0 (수직) | -0.9 | 0.0 | **0** |
| +1.0 (반대) | -1.9 | 0.0 | **0** |

```
보상 그래프:

보상
↑
1.0 │        ╱
    │       ╱
0.5 │      ╱
    │     ╱
0.0 │─────────────────
    └──────────────────→ dot product
    +1    0   -0.9  -1
    
문제: dot > -0.9 이면 보상 = 0 (학습 신호 없음!)
```

**왜 문제인가?**
- 초기 policy는 랜덤한 자세
- 대부분 dot product가 -0.9보다 큼 (정렬 안 됨)
- 정렬 보상 = 0 → "어떻게 해야 정렬되는지" 학습 불가
- 학습 정체!

---

### 문제 3: 거리 조건이 너무 엄격

**현재 코드 (Line 586):**
```python
distance_factor = torch.clamp(1.0 - distance_to_cap / 0.10, min=0.0)
```

**문제점:**
- 10cm 이내에서만 정렬 보상
- 멀리 있으면 정렬 보상 = 0
- "일단 가까이 가야 정렬 보상 받음"
- 근데 가까이 가도 정렬 안 되어 있으면 보상 0
- 악순환!

---

### 문제 4: Success 조건이 너무 엄격

**현재 코드:**
```python
# task_success_termination (Line 704-705)
close_enough = distance_to_cap < 0.03  # 3cm
aligned = dot_product < -0.9            # 거의 완벽 정렬

# alignment_success_reward (Line 634-635)
close_enough = distance_to_cap < 0.03
aligned = dot_product < -0.9
```

**문제점:**
- 3cm 이내 + dot < -0.9 (약 25도 이내)
- 초기 학습에서는 거의 달성 불가능
- Success 보상을 못 받음 → 목표가 뭔지 학습 안 됨

---

## 수정 사항

### 수정 1: z_axis_alignment_reward 함수 (점진적 보상)

**파일**: `pen_grasp_env.py`
**위치**: Line 536-588

```python
def z_axis_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Z축 정렬 보상 (수정됨 - 점진적 보상)

    그리퍼의 Z축을 펜의 Z축과 평행하게 정렬하도록 유도합니다.

    === 수정 사항 (2025-12-16) ===
    - 기존: dot < -0.9 일 때만 보상 (절벽)
    - 수정: 전체 범위에서 점진적 보상
    - 거리 조건: 10cm → 30cm로 완화

    Returns:
        (num_envs,) - 정렬 보상
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    # --- 펜 Z축 계산 ---
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_z_x = 2.0 * (qx * qz + qw * qy)
    pen_z_y = 2.0 * (qy * qz - qw * qx)
    pen_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pen_z_axis = torch.stack([pen_z_x, pen_z_y, pen_z_z], dim=-1)

    # 캡 위치
    cap_pos = pen_pos + (PEN_LENGTH / 2) * pen_z_axis

    # --- 거리 계산 ---
    grasp_pos = get_grasp_point(robot)
    distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

    # --- 그리퍼 Z축 계산 (link_6의 방향에서) ---
    link6_quat = robot.data.body_quat_w[:, 6, :]  # link_6 방향
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)

    # --- 정렬 계산 (수정됨) ---
    # dot product: +1 = 같은 방향, -1 = 반대 방향, 0 = 수직
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # [수정] 전체 범위에서 점진적 보상
    # dot = -1 → 1.0, dot = 0 → 0.5, dot = +1 → 0.0
    alignment_reward = (-dot_product + 1.0) / 2.0

    # [수정] 거리 조건 완화: 10cm → 30cm
    distance_factor = torch.clamp(1.0 - distance_to_cap / 0.30, min=0.0)

    return alignment_reward * distance_factor
```

**수정 후 보상 테이블:**

| dot product | 보상 계산 | 보상 |
|-------------|-----------|------|
| -1.0 (완벽) | (1+1)/2 | **1.0** |
| -0.5 | (0.5+1)/2 | **0.75** |
| 0.0 (수직) | (0+1)/2 | **0.5** |
| +0.5 | (-0.5+1)/2 | **0.25** |
| +1.0 (반대) | (-1+1)/2 | **0.0** |

```
보상 그래프 (수정 후):

보상
↑
1.0 │            ╲
    │           ╲
0.5 │          ╲
    │         ╲
0.0 │        ╲
    └──────────────────→ dot product
    +1    0   -0.5  -1
    
모든 범위에서 학습 신호 있음!
```

---

### 수정 2: alignment_success_reward 함수 (조건 완화)

**파일**: `pen_grasp_env.py`
**위치**: Line 591-638

```python
def alignment_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위치 + 정렬 성공 보상 (수정됨 - 조건 완화)

    그리퍼가 펜 캡에 위치하고 정렬되면 큰 보상을 줍니다.

    === 수정 사항 (2025-12-16) ===
    - 거리: 3cm → 5cm
    - 정렬: dot < -0.9 → dot < -0.7 (약 45도)

    Returns:
        (num_envs,) - 성공 시 1.0, 아니면 0.0
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    # --- 거리 계산 ---
    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w

    # 캡 위치 계산
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_z_x = 2.0 * (qx * qz + qw * qy)
    pen_z_y = 2.0 * (qy * qz - qw * qx)
    pen_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pen_z_axis = torch.stack([pen_z_x, pen_z_y, pen_z_z], dim=-1)
    cap_pos = pen_pos + (PEN_LENGTH / 2) * pen_z_axis

    distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

    # --- 정렬 계산 ---
    link6_quat = robot.data.body_quat_w[:, 6, :]
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)

    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # [수정] 성공 조건 완화
    close_enough = distance_to_cap < 0.05  # 3cm → 5cm
    aligned = dot_product < -0.7           # -0.9 → -0.7 (약 45도)

    success = (close_enough & aligned).float()
    return success
```

---

### 수정 3: task_success_termination 함수 (조건 완화)

**파일**: `pen_grasp_env.py`
**위치**: Line 662-707

```python
def task_success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Task 성공 종료 조건 (수정됨 - 조건 완화)

    그리퍼가 펜 캡에 위치하고 정렬되면 에피소드를 성공으로 종료합니다.

    === 수정 사항 (2025-12-16) ===
    - 거리: 3cm → 5cm
    - 정렬: dot < -0.9 → dot < -0.7 (약 45도)

    Returns:
        (num_envs,) - True/False 종료 여부
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]

    # --- 거리 계산 ---
    grasp_pos = get_grasp_point(robot)
    pen_pos = pen.data.root_pos_w
    pen_quat = pen.data.root_quat_w

    # 캡 위치 계산
    qw, qx, qy, qz = pen_quat[:, 0], pen_quat[:, 1], pen_quat[:, 2], pen_quat[:, 3]
    pen_z_x = 2.0 * (qx * qz + qw * qy)
    pen_z_y = 2.0 * (qy * qz - qw * qx)
    pen_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pen_z_axis = torch.stack([pen_z_x, pen_z_y, pen_z_z], dim=-1)
    cap_pos = pen_pos + (PEN_LENGTH / 2) * pen_z_axis

    distance_to_cap = torch.norm(grasp_pos - cap_pos, dim=-1)

    # --- 정렬 계산 ---
    link6_quat = robot.data.body_quat_w[:, 6, :]
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)

    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)

    # [수정] 성공 조건 완화
    close_enough = distance_to_cap < 0.05  # 3cm → 5cm
    aligned = dot_product < -0.7           # -0.9 → -0.7

    return close_enough & aligned
```

---

## 수정 요약

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| **정렬 보상 방식** | 절벽 (dot < -0.9만) | 점진적 (전체 범위) |
| **정렬 보상 거리** | 10cm 이내 | 30cm 이내 |
| **Success 거리** | 3cm | 5cm |
| **Success 정렬** | dot < -0.9 (~25도) | dot < -0.7 (~45도) |
| **학습 시작** | checkpoint 이어서 | **새로 시작** |

---

## 학습 실행 방법

### 1) 코드 수정 후 새로 학습 시작

```bash
# checkpoint 없이 새로 시작!
python train.py --headless --num_envs 8192 --max_iterations 20000
```

### 2) 모니터링 포인트

```bash
# TensorBoard로 확인
tensorboard --logdir=./logs/pen_grasp
```

확인할 것:
- `distance_to_cap` reward: 올라가는지?
- `z_axis_alignment` reward: 올라가는지?
- `task_success` rate: 점진적 상승?

### 3) 예상 학습 곡선

```
Iteration:    0 -----> 5000 -----> 10000 ----> 15000
                
distance:     0.2 ---> 0.5 -----> 0.7 ------> 0.8
alignment:    0.3 ---> 0.5 -----> 0.7 ------> 0.8
success:      0% ----> 5% ------> 20% ------> 50%+
```

---

## 향후 개선 (학습 성공 후)

### 단계 1: 조건 점진적 강화

```python
# 학습 성공 후 조건 엄격하게
close_enough = distance_to_cap < 0.03  # 5cm → 3cm
aligned = dot_product < -0.85          # -0.7 → -0.85
```

### 단계 2: 펜 각도 DR 추가

```python
# EventsCfg의 reset_pen에 추가
"roll": (-0.26, 0.26),   # ±15도
"pitch": (-0.26, 0.26),
```

### 단계 3: 물리 DR 추가 (선택)

```python
pen_mass_range = [0.010, 0.020]      # 10~20g
pen_friction_range = [0.5, 1.0]
```

---

## 디버깅 팁

### 학습이 안 되면?

1. **Reward 스케일 확인**
   ```python
   # RewardsCfg에서 weight 조절
   distance_to_cap = RewTerm(..., weight=1.0)
   z_axis_alignment = RewTerm(..., weight=0.5)  # 줄여보기
   ```

2. **네트워크 크기 조절**
   ```python
   # train.py에서
   actor_hidden_dims=[128, 128, 64]  # 더 작게
   ```

3. **학습률 조절**
   ```python
   learning_rate=1e-4  # 3e-4 → 1e-4로 낮추기
   ```

---

*작성일: 2024-12-16*
