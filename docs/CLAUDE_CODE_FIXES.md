# Pen Grasp RL 코드 수정 지침서

이 문서는 Claude Code에서 `pen_grasp_env.py`와 `train.py`를 수정할 때 참고할 내용입니다.

---

## 🚨 긴급 수정 사항

### 1. Z축 정렬 보상 방향 수정 (필수)

**문제**: 현재 코드는 그리퍼와 펜이 **같은 방향**일 때 보상을 줌. 하지만 실제 목표는 **반대 방향** (그리퍼가 위에서 캡을 향해 내려옴)

**파일**: `pen_grasp_rl/envs/pen_grasp_env.py`

**위치**: `z_axis_alignment_reward` 함수 (약 536-586 라인)

**현재 코드**:
```python
# dot > 0.9 일 때만 보상 (0.9~1.0 → 0~1)
alignment_reward = torch.clamp(dot_product - 0.9, min=0.0) * 10.0
```

**수정 코드**:
```python
# dot < -0.9 일 때만 보상 (반대 방향일 때)
# -dot_product를 사용하여 반대 방향(dot=-1)일 때 최대 보상
alignment_reward = torch.clamp(-dot_product - 0.9, min=0.0) * 10.0
```

**이유**: 
- 펜: 캡이 위(+Z), 촉이 아래
- 그리퍼: 아래를 향함(-Z)
- 목표 dot product = -1 (반대 방향)

---

## ⚠️ 권장 수정 사항

### 2. Alignment Success 보상 추가 (권장)

**파일**: `pen_grasp_rl/envs/pen_grasp_env.py`

**위치**: 보상 함수 섹션 (4. 보상 정의 부분)

**추가할 함수**:
```python
def alignment_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    위치 + 정렬 성공 시 큰 보상
    
    조건:
    1. 그리퍼가 펜 캡에서 3cm 이내
    2. Z축이 거의 반대 방향 (dot < -0.9)
    
    Returns:
        (num_envs,) - 성공 시 1.0, 아니면 0.0
    """
    robot: Articulation = env.scene["robot"]
    pen: RigidObject = env.scene["pen"]
    
    # 거리 계산
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
    
    # 그리퍼 Z축 계산
    link6_quat = robot.data.body_quat_w[:, 6, :]
    qw, qx, qy, qz = link6_quat[:, 0], link6_quat[:, 1], link6_quat[:, 2], link6_quat[:, 3]
    gripper_z_x = 2.0 * (qx * qz + qw * qy)
    gripper_z_y = 2.0 * (qy * qz - qw * qx)
    gripper_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    gripper_z_axis = torch.stack([gripper_z_x, gripper_z_y, gripper_z_z], dim=-1)
    
    # 정렬 계산
    dot_product = torch.sum(pen_z_axis * gripper_z_axis, dim=-1)
    
    # 성공 조건: 3cm 이내 AND dot < -0.9
    close_enough = distance_to_cap < 0.03
    aligned = dot_product < -0.9
    
    success = (close_enough & aligned).float()
    return success
```

**RewardsCfg에 추가**:
```python
@configclass
class RewardsCfg:
    distance_to_cap = RewTerm(func=distance_ee_cap_reward, weight=1.0)
    z_axis_alignment = RewTerm(func=z_axis_alignment_reward, weight=0.5)
    floor_collision = RewTerm(func=floor_collision_penalty, weight=1.0)
    action_rate = RewTerm(func=action_rate_penalty, weight=0.01)
    # 추가
    alignment_success = RewTerm(func=alignment_success_reward, weight=5.0)
```

---

### 3. 함수 이름 명확화 (선택)

**현재**: `action_rate_penalty` - 이름과 동작 불일치

**실제 동작**: 액션 **크기**(magnitude) 페널티

**수정 옵션 A - 이름만 변경**:
```python
def action_magnitude_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    행동 크기 페널티
    
    큰 행동(움직임)에 페널티를 줘서 에너지 효율적인 동작을 유도합니다.
    
    Returns:
        (num_envs,) - 음수 페널티 값
    """
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)
```

**수정 옵션 B - 진짜 action rate 구현** (더 복잡):
```python
# ArmGripperActionTerm 클래스에 이전 액션 저장 추가 필요
def __init__(self, cfg, env):
    super().__init__(cfg, env)
    # ... 기존 코드 ...
    self._prev_actions = torch.zeros(env.num_envs, 7, device=self.device)

def process_actions(self, actions):
    self._prev_actions[:] = self._raw_actions.clone()
    self._raw_actions[:] = actions
    self._processed_actions[:] = actions

# 보상 함수
def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """액션 변화율 페널티 (부드러운 동작 유도)"""
    action_term = env.action_manager._terms["arm_gripper"]
    current = action_term._raw_actions
    previous = action_term._prev_actions
    action_diff = current - previous
    return -torch.sum(torch.square(action_diff), dim=-1)
```

---

## 📋 Resume 학습 시 참고사항

### 수정 후 Resume 명령어
```bash
source ~/isaacsim_env/bin/activate
cd ~/CoWriteBotRL
python pen_grasp_rl/scripts/train.py \
    --headless \
    --num_envs 4096 \
    --max_iterations 3000 \
    --resume \
    --checkpoint ./logs/pen_grasp/model_XXXX.pt
```

### 호환성 체크리스트
- [x] 관찰 차원: 36 (변경 없음)
- [x] 행동 차원: 7 (변경 없음)  
- [x] 네트워크 구조: 동일 (변경 없음)
- [x] 보상 함수: 수정해도 체크포인트와 호환

### 학습 모니터링 포인트
Resume 후 TensorBoard에서 확인할 사항:
1. `z_axis_alignment` 보상이 0보다 커지는지
2. `alignment_success` 보상이 발생하는지 (추가한 경우)
3. Value loss가 잠깐 튀었다가 안정되는지 (정상)

---

## 📁 수정 파일 요약

| 파일 | 수정 내용 | 우선순위 |
|------|----------|----------|
| `pen_grasp_env.py` | z_axis_alignment_reward 방향 수정 | 🔴 필수 |
| `pen_grasp_env.py` | alignment_success_reward 추가 | 🟡 권장 |
| `pen_grasp_env.py` | action_rate_penalty 이름/주석 수정 | 🟢 선택 |

---

## 🔍 수정 전 확인 사항

Claude Code 실행 전에 다음을 확인하세요:

1. 현재 학습이 완료되었는지
2. 체크포인트 파일 경로 확인 (`./logs/pen_grasp/model_XXXX.pt`)
3. 기존 코드 백업 (git commit 권장)

---

## 커밋 메시지 예시

```
fix: z_axis_alignment 보상 방향 수정 (dot → -dot)

- 목표: 그리퍼 Z축과 펜 Z축이 반대 방향일 때 보상
- 이전: dot > 0.9 (같은 방향)일 때 보상 (잘못됨)
- 수정: -dot > 0.9 (반대 방향)일 때 보상

추가:
- alignment_success_reward 함수 추가 (위치+정렬 성공 보상)
```
