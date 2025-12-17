# Pen Grasp RL 코드 수정 지침서 V2

이 문서는 `pen_grasp_rl/envs/pen_grasp_env.py` 수정을 위한 Claude Code 지침입니다.

---

## 수정 개요

| 항목 | 현재 | 변경 | 이유 |
|------|------|------|------|
| z_axis_alignment weight | 0.5 | **1.5** | 정렬 학습 강화 |
| 주석 (line 544) | "dot > 0.9" | "dot < -0.9" | 실제 코드와 일치 |
| task_success 종료 조건 | 없음 | **추가** | 성공 시 조기 종료 |
| alignment_success 보상 | 없음 | **추가** | 성공 시 큰 보상 |

---

## 1. 주석 수정 (필수)

**파일**: `pen_grasp_env.py`
**위치**: `z_axis_alignment_reward` 함수의 docstring (약 line 542-545)

**현재**:
```python
    === 조건 (2025-12-15 수정) ===
    1. 그리퍼가 펜 캡에서 10cm 이내일 때만 보상 (기존 5cm에서 확장)
    2. dot product > 0.9 일 때만 보상 (거의 평행)
```

**수정**:
```python
    === 조건 (2025-12-16 수정) ===
    1. 그리퍼가 펜 캡에서 10cm 이내일 때만 보상
    2. dot product < -0.9 일 때만 보상 (반대 방향, 그리퍼가 캡을 향해 내려옴)
```

---

## 2. z_axis_alignment weight 증가 (필수)

**위치**: `RewardsCfg` 클래스 (약 line 680)

**현재**:
```python
z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=0.5)
```

**수정**:
```python
z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=1.5)
```

---

## 3. Task 성공 종료 조건 추가 (필수)

### 3-1. 종료 함수 추가

**위치**: `pen_dropped_termination` 함수 아래 (약 line 610 이후)

**추가할 코드**:
```python
def task_success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Task 성공 종료 조건
    
    그리퍼가 펜 캡에 정확히 위치하고 정렬되면 에피소드를 성공으로 종료합니다.
    
    성공 조건:
    1. 그리퍼-캡 거리 3cm 이내
    2. Z축 정렬 (dot product < -0.9)
    
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
    
    # 성공 조건: 3cm 이내 AND dot < -0.9 (반대 방향)
    close_enough = distance_to_cap < 0.03
    aligned = dot_product < -0.9
    
    return close_enough & aligned
```

### 3-2. TerminationsCfg에 등록

**위치**: `TerminationsCfg` 클래스 (약 line 686-694)

**현재**:
```python
@configclass
class TerminationsCfg:
    """
    종료 조건 설정

    - time_out: 에피소드 시간 초과 (10초)
    - pen_dropped: 펜이 바닥으로 떨어짐 (현재 kinematic이라 발동 안함)
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pen_dropped = DoneTerm(func=pen_dropped_termination)
```

**수정**:
```python
@configclass
class TerminationsCfg:
    """
    종료 조건 설정

    - time_out: 에피소드 시간 초과 (10초)
    - pen_dropped: 펜이 바닥으로 떨어짐 (현재 kinematic이라 발동 안함)
    - task_success: 위치+정렬 성공 시 조기 종료 (성공!)
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pen_dropped = DoneTerm(func=pen_dropped_termination)
    task_success = DoneTerm(func=task_success_termination, time_out=False)
```

---

## 4. Alignment Success 보상 추가 (권장)

### 4-1. 보상 함수 추가

**위치**: `z_axis_alignment_reward` 함수 아래 (약 line 589 이후)

**추가할 코드**:
```python
def alignment_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위치 + 정렬 성공 보상
    
    그리퍼가 펜 캡에 정확히 위치하고 정렬되면 큰 보상을 줍니다.
    task_success_termination과 동일한 조건이지만, 종료 전에 보상을 받습니다.
    
    성공 조건:
    1. 그리퍼-캡 거리 3cm 이내
    2. Z축 정렬 (dot product < -0.9)
    
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
    
    # 성공 조건
    close_enough = distance_to_cap < 0.03
    aligned = dot_product < -0.9
    
    success = (close_enough & aligned).float()
    return success
```

### 4-2. RewardsCfg에 등록

**위치**: `RewardsCfg` 클래스 (약 line 679-682)

**현재**:
```python
@configclass
class RewardsCfg:
    distance_to_cap = RewTerm(func=distance_ee_cap_reward, weight=1.0)
    z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=0.5)
    floor_collision = RewTerm(func=floor_collision_penalty, weight=1.0)
    action_rate = RewTerm(func=action_rate_penalty, weight=0.01)
```

**수정**:
```python
@configclass
class RewardsCfg:
    """
    보상 설정

    === 보상 구성 (2025-12-16 수정) ===
    양수 보상 (목표 행동 유도):
    - distance_to_cap (w=1.0): 캡에 가까워지기 (exponential)
    - z_axis_alignment (w=1.5): 축 정렬 - 반대 방향 (10cm 이내)
    - alignment_success (w=5.0): 위치+정렬 성공 시 큰 보상

    음수 보상 (나쁜 행동 억제):
    - floor_collision (w=1.0): 바닥 충돌
    - action_rate (w=0.01): 급격한 움직임
    """
    distance_to_cap = RewTerm(func=distance_ee_cap_reward, weight=1.0)
    z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=1.5)
    floor_collision = RewTerm(func=floor_collision_penalty, weight=1.0)
    action_rate = RewTerm(func=action_rate_penalty, weight=0.01)
    alignment_success = RewTerm(func=alignment_success_reward, weight=5.0)
```

---

## 수정 후 확인사항

### 1. Import 확인
`Articulation`, `RigidObject`, `torch`가 이미 import 되어 있으므로 추가 import 불필요.

### 2. 함수 순서
1. `z_axis_alignment_reward` (기존)
2. `alignment_success_reward` (새로 추가)
3. `pen_dropped_termination` (기존)
4. `task_success_termination` (새로 추가)

### 3. TensorBoard에서 확인할 지표
- `Episode_Reward/z_axis_alignment`: 증가해야 함
- `Episode_Reward/alignment_success`: 0보다 커지면 성공 발생
- `Episode_Termination/task_success`: 성공률 (높을수록 좋음)
- `Episode_Termination/time_out`: 낮아져야 함 (성공이 늘면 타임아웃 감소)

---

## 수정 완료 후 학습 명령어

```bash
# 이전 체크포인트에서 이어서 학습
source ~/isaacsim_env/bin/activate
cd ~/CoWriteBotRL
python pen_grasp_rl/scripts/train.py \
    --headless \
    --num_envs 4096 \
    --max_iterations 5000 \
    --resume \
    --checkpoint ./logs/pen_grasp/model_XXXX.pt
```

---

## 커밋 메시지 예시

```
feat: 정렬 보상 강화 및 성공 종료 조건 추가

수정 사항:
- z_axis_alignment weight 0.5 → 1.5 증가
- alignment_success 보상 함수 추가 (weight=5.0)
- task_success 종료 조건 추가 (3cm + dot<-0.9)
- docstring 수정 (dot > 0.9 → dot < -0.9)
```
