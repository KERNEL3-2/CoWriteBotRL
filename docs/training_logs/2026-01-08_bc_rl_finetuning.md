# BC + RL Fine-tuning 학습 로그

**날짜**: 2026-01-08
**목적**: ACT(BC) 모델을 기본 정책으로 사용하고, RL로 residual을 학습하여 펜 잡기 성능 향상

---

## 1. 배경

### 문제 상황
- ACT 모델을 1,000개 MoveIt2 궤적으로 학습
- 시뮬레이션 테스트 결과: 1.8% 성공률 (2/113 에피소드)
- 문제점:
  - Open-loop(MoveIt2) vs Closed-loop(BC 정책) 불일치
  - Compounding error (작은 오차 누적)
  - 그리퍼가 펜 위에서 진동

### 해결 방향
- **Residual Policy Learning**: BC + RL 결합
- Action = BC_weight × ACT(obs) + residual_scale × RL(obs)
- RL이 BC의 오차를 보정

---

## 2. 구현

### 2.1 환경 설계 (Option A: Jacobian 변환)

IK 환경(E0509IKEnvV7)을 상속하여 BC+RL 환경 구현:

1. BC 모델이 6D joint position target 출력
2. 현재 joint position과의 차이(delta) 계산
3. **Jacobian으로 joint delta → EE delta (3D) 변환**
4. RL action (3D EE delta)과 결합
5. 부모 클래스의 _apply_action() 호출 → **자동 orientation 정렬 유지**

```python
# BC joint position → EE delta 변환
bc_joint_delta = bc_joint_target - current_joint
jacobian_pos = self._compute_bc_jacobian_pos()  # (num_envs, 3, 6)
bc_ee_delta = torch.bmm(jacobian_pos, bc_joint_delta.unsqueeze(-1)).squeeze(-1)

# BC + RL 결합
final_ee_delta = bc_weight * bc_ee_delta + residual_scale * rl_ee_delta
```

### 2.2 PPO 설정

```python
# 낮은 entropy (BC 정책 유지)
entropy_coef = 0.005

# 낮은 learning rate
learning_rate = 1e-4
schedule = "fixed"

# Actor/Critic 네트워크
hidden_dims = [256, 256, 128]
activation = "elu"
```

---

## 3. 학습 결과

### 3.1 1차 학습 (8,000 iterations)

| 메트릭 | 시작 (0 iter) | 종료 (8,000 iter) |
|--------|---------------|-------------------|
| Mean Reward | -165 | 4,961 |
| dist_to_cap | 42.5cm | 3.2cm |
| perp_dist | 27.6cm | 1.0cm |
| Total Success | 0 | 342,874 |

**발견된 문제**:
- 정책이 목표 도달 후 멈추지 않고 계속 움직임
- SUCCESS_HOLD_STEPS=30이 너무 길어서 정책이 "멈추기"를 학습 못함
- success_hold_count_mean = 2.9 (30에 한참 못 미침)

### 3.2 2차 학습 (8,000 → 11,000 iterations)

**환경 수정**: SUCCESS_HOLD_STEPS = 30 → 5

| 메트릭 | 8,000 iter | 11,000 iter | 변화 |
|--------|------------|-------------|------|
| Mean Reward | 359 | 4,660 | +1200% |
| dist_to_cap | 19.1cm | 2.9cm | 개선 |
| perp_dist | 5.9cm | 1.7cm | 개선 |
| Total Success | 4,931 | 177,847 | 36배 증가 |

---

## 4. 테스트 결과

### 4.1 시뮬레이션 테스트

**순정 모드 (후처리 없음)**:
- 정확도: 좋음 (dist < 3cm, perp < 2cm 도달)
- 문제: 펜 위에서 위아래 진동 발생

**--stop_on_success 모드**:
- 성공 조건 충족 시 action=0
- 진동 감소, 하지만 정책 자체가 "멈추기"를 학습하지 않음

### 4.2 진동 원인 분석

1. 정책이 "계속 움직여서 목표에 가까이 가라"고 학습됨
2. "도달하면 멈춰라"는 학습하지 않음
3. 목표 도달 후에도 action 계속 출력 → 진동

---

## 5. 생성/수정된 파일

| 파일 | 작업 | 설명 |
|------|------|------|
| `pen_grasp_rl/scripts/train_bc_rl.py` | 신규 | BC+RL 학습 스크립트 |
| `pen_grasp_rl/scripts/play_bc_rl.py` | 신규 | BC+RL 테스트 스크립트 |
| `pen_grasp_rl/envs/e0509_ik_env_v7.py` | 수정 | SUCCESS_HOLD_STEPS 30→5 |

### 체크포인트 위치
```
pen_grasp_rl/scripts/pen_grasp_rl/logs/bc_rl_finetune_w0.7_s0.3/
├── model_8000.pt   # 1차 학습 완료
├── model_10999.pt  # 2차 학습 완료 (최종)
└── events.out.tfevents.*  # TensorBoard 로그
```

---

## 6. 다음 단계

### 진동 문제 해결 방안
1. **학습 환경 수정**: 성공 조건 충족 시 action=0으로 클리핑하여 재학습
2. **Sim2Real 후처리**: 로봇 제어 단에서 성공 시 정지 로직 적용

### 추가 개선 방향
1. 펜 기울기 범위 확장 (현재 0-30도)
2. Curriculum learning 적용
3. Domain randomization 강화

---

## 7. 실행 명령어

### 학습
```bash
cd /home/fhekwn549/CoWriteBotRL/pen_grasp_rl/scripts && \
python train_bc_rl.py --headless --num_envs 4096 \
    --bc_checkpoint ../checkpoints/act/checkpoints/best_model_compat.pth \
    --max_iterations 3000
```

### 테스트
```bash
cd /home/fhekwn549/CoWriteBotRL/pen_grasp_rl/scripts && \
python play_bc_rl.py \
    --bc_checkpoint ../checkpoints/act/checkpoints/best_model_compat.pth \
    --rl_checkpoint ./pen_grasp_rl/logs/bc_rl_finetune_w0.7_s0.3/model_10999.pt \
    --num_envs 16 \
    --stop_on_success
```
