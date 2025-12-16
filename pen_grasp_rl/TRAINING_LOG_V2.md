# Pen Grasp RL 학습 기록 (V2 환경)

## 개요

V2 환경은 Isaac Lab의 **reach 예제**를 기반으로 재설계된 단순화된 환경입니다.

### V1 vs V2 비교

| 항목 | V1 (기존) | V2 (신규) |
|------|-----------|-----------|
| 보상 개수 | 7개 | **4개** |
| 관찰 차원 | 36 | **27** |
| 구조 | 직접 구현 | **reach 예제 기반** |
| 목표 | 여러 조건 혼합 | **위치 + 방향만** |

### 학습 목표 (2가지만)

1. **위치**: `gripper_grasp_point` → `pen_cap_point` 거리 최소화
2. **방향**: `gripper_z` · `pen_z` → -1 (반대 방향 정렬)

---

## 환경 구조

### 관찰 공간 (27차원)

| 관찰 | 차원 | 설명 |
|------|------|------|
| `joint_pos` | 6 | 팔 관절 위치 |
| `joint_vel` | 6 | 팔 관절 속도 |
| `grasp_point` | 3 | 그리퍼 잡기 포인트 위치 |
| `pen_cap` | 3 | 펜 캡 위치 |
| `relative_pos` | 3 | 그리퍼→캡 상대 위치 (핵심!) |
| `pen_z_axis` | 3 | 펜 Z축 방향 |
| `gripper_z_axis` | 3 | 그리퍼 Z축 방향 |

### 보상 함수 (4개)

| 보상 | weight | 형태 | 설명 |
|------|--------|------|------|
| `position_error` | -0.5 | L2 거리 | 거리 페널티 |
| `position_fine` | +1.0 | 1 - tanh(d/0.1) | 정밀 위치 보상 |
| `orientation_error` | -0.3 | 1 + dot | 방향 오차 페널티 |
| `action_rate` | -0.001 | action² | 행동 페널티 |

### 보상 형태 (reach 예제 스타일)

**위치 보상:**
```python
# L2 거리 (페널티)
distance = ||grasp_pos - cap_pos||
position_error = distance  # weight: -0.5

# tanh 커널 (보상)
position_fine = 1 - tanh(distance / 0.1)  # weight: +1.0
```

**방향 보상:**
```python
# dot product 오차
dot = pen_z · gripper_z
orientation_error = 1 + dot  # dot=-1이면 0, dot=+1이면 2
# weight: -0.3
```

---

## 학습 실행 방법

```bash
source ~/isaacsim_env/bin/activate
cd ~/IsaacLab

# V2 환경으로 학습 (기본값)
python pen_grasp_rl/scripts/train.py --num_envs 64

# V1 환경으로 학습하려면
python pen_grasp_rl/scripts/train.py --num_envs 64 --env_version v1
```

---

## TensorBoard 확인 지표

```bash
tensorboard --logdir=~/IsaacLab/logs/pen_grasp
# 브라우저에서 http://localhost:6006 접속
```

| 지표 | 좋은 신호 |
|------|-----------|
| `Episode_Reward/position_error` | 📉 감소 (거리 줄어듦) |
| `Episode_Reward/position_fine` | 📈 증가 (가까워짐) |
| `Episode_Reward/orientation_error` | 📉 감소 (정렬됨) |
| `Train/mean_reward` | 📈 증가 |

---

## 학습 기록

### 2025-12-16 V2 환경 생성

#### 배경
- V1 환경에서 보상 함수를 여러 번 수정했으나 학습이 잘 안됨
- 측면에서 접근하여 펜과 충돌하는 문제 발생
- 기존 예제 없이 직접 구현한 것이 문제의 원인

#### 해결책
- Isaac Lab의 **reach 예제** 구조를 기반으로 재설계
- 보상 함수를 **4개로 단순화** (기존 7개)
- 검증된 보상 형태 사용 (L2, tanh)

#### 변경 사항

**1. 새 파일 생성**
- `pen_grasp_rl/envs/pen_grasp_env_v2.py`

**2. 보상 구조 단순화**
```python
# V1: 7개 보상 (복잡)
distance_to_cap, z_axis_alignment, base_orientation,
approach_from_above, alignment_success, floor_collision, action_rate

# V2: 4개 보상 (단순)
position_error, position_fine, orientation_error, action_rate
```

**3. 관찰 공간 정리**
- 36차원 → 27차원
- 불필요한 관찰 제거
- 핵심 관찰만 유지 (relative_pos, z_axis 등)

**4. train.py 수정**
- `--env_version` 인자 추가
- 기본값: v2

#### 다음 단계
- [ ] V2 환경으로 학습 실행
- [ ] position_fine 보상 증가 확인
- [ ] orientation_error 감소 확인
- [ ] play.py로 동작 확인

---
