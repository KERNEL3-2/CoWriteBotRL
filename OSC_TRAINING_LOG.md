# OSC (Operational Space Control) 환경 구축 로그

**작성일**: 2025-12-30

## 개요

기존 IK(Inverse Kinematics) 기반 V7 환경을 OSC(Operational Space Control) 방식으로 변환하고, Sim2Real 연동을 위한 펜 범위 검증 시스템을 구축했습니다.

---

## 1. OSC 환경 생성

### 파일 생성
- `pen_grasp_rl/envs/e0509_osc_env.py` (NEW)
- `pen_grasp_rl/scripts/train_osc.py` (NEW)
- `pen_grasp_rl/scripts/play_osc.py` (NEW)

### IK vs OSC 차이점

| 항목 | IK (V7) | OSC |
|------|---------|-----|
| 제어 방식 | 관절 위치 타겟 | 관절 토크 타겟 |
| 함수 | `set_joint_position_target()` | `set_joint_effort_target()` |
| Actuator | stiffness=400, damping=40 | **stiffness=0, damping=0** |
| 컨트롤러 | `DifferentialIKController` | `OperationalSpaceController` |

### OSC 설정
```python
OperationalSpaceControllerCfg(
    target_types=["pose_rel"],
    impedance_mode="fixed",
    inertial_dynamics_decoupling=True,
    gravity_compensation=True,
    motion_stiffness_task=150.0,
    motion_damping_ratio_task=1.0,
)
```

---

## 2. USD 관절 제한 수정

### DART Platform 실제 로봇 값 적용
`pen_grasp_rl/models/first_control.usd` 수정

| 관절 | 이전 | 이후 |
|------|------|------|
| J1 | ±360° | ±360° (동일) |
| J2 | ±360° | **±95°** |
| J3 | ±155° | **±135°** |
| J4 | ±360° | ±360° (동일) |
| J5 | ±360° | **±135°** |
| J6 | ±360° | ±360° (동일) |

---

## 3. 펜 충돌 방지 보상함수 수정

### 문제점
- 기존 학습에서 그리퍼가 펜에 닿는 경우가 많았음
- 충돌 감지 범위가 너무 넓고 (3cm), 캡 영역도 포함되어 있었음

### 수정 내용

| 항목 | 이전 | 이후 |
|------|------|------|
| 충돌 감지 범위 | 3cm | **1.5cm** |
| 충돌 페널티 | -10 | **-50** |
| 충돌 시 | 페널티만 | **에피소드 즉시 종료** |
| 감지 영역 | 캡 포함 | **몸체만 (캡 제외)** |

### 충돌 감지 로직 수정
```python
# 펜 몸체만 충돌 감지 (캡 영역 제외)
# proj_length < 0: 그리퍼가 캡 위에 있음 (정상 접근, 충돌 감지 X)
# proj_length > 0: 그리퍼가 몸체 쪽에 있음 (충돌 감지 O)
in_pen_body = (proj_length > 0) & (proj_length < PEN_LENGTH / 2)
collision = in_pen_body & (perp_dist < 0.015)  # 1.5cm
```

---

## 4. Sim2Real 펜 범위 검증 시스템

### 공유 설정 파일 생성
`pen_grasp_rl/config/pen_workspace.py`

시뮬레이션과 실제 로봇에서 동일한 펜 위치/각도 범위를 사용하기 위한 공유 설정 파일입니다.

### 카메라 검증 기능 추가
`sim2real/sim2real/test_pen_detection_calibrated.py` 수정

- 펜 위치/각도가 학습 범위 내에 있는지 실시간 검증
- 화면에 "IN RANGE" (초록) / "OUT OF RANGE" (빨강) 표시
- 이동 평균 필터 추가 (10프레임) - 노이즈 제거

### 사용법
```bash
cd ~/sim2real/sim2real
python3 test_pen_detection_calibrated.py
```

---

## 5. 펜 스폰 범위 조정

### 최종 설정값

| 항목 | 이전 | 이후 |
|------|------|------|
| X 범위 | 30 ~ 45 cm | **25 ~ 55 cm** |
| Y 범위 | -12 ~ 12 cm | -12 ~ 12 cm (동일) |
| Z 범위 | 22 ~ 35 cm | 22 ~ 35 cm (동일) |
| 최대 기울기 | 20° | **45°** |

### 적용 파일
- `pen_grasp_rl/config/pen_workspace.py` (sim2real 검증용)
- `pen_grasp_rl/envs/e0509_osc_env.py` (학습용)

---

## 6. 파일 구조

```
pen_grasp_rl/
├── config/
│   ├── __init__.py
│   └── pen_workspace.py          # Sim2Real 공유 설정 (NEW)
├── envs/
│   ├── __init__.py
│   ├── e0509_ik_env_v7.py        # IK 환경 (기존)
│   └── e0509_osc_env.py          # OSC 환경 (NEW)
├── scripts/
│   ├── train_osc.py              # OSC 학습 (NEW)
│   └── play_osc.py               # OSC 테스트 (NEW)
└── models/
    └── first_control.usd         # 관절 제한 수정됨
```

---

---

# 학습 실험 로그

## 실험 1: Default OSC (2025-01-05)

### 설정
| 파라미터 | 값 |
|----------|-----|
| stiffness | 150 |
| damping_ratio | 1.0 |
| action_scale | 0.05 |
| Learning Rate | Fixed (1e-4) |
| num_envs | 4096 |
| iterations | 5500 |

### 결과

| 지표 | 시작 | 최종 | 비고 |
|------|------|------|------|
| Mean Reward | -228 | 6383 | 아주 좋음 |
| Episode Length | 22 | 427 | 거의 최대(450) |
| Dist to Cap | - | 3.7cm | 성공기준 3cm 근접 |
| Perp Dist | - | 0.8cm | 성공기준 1cm **달성** |
| Alignment (dot) | - | -0.74 | 좋음 (-1이 완벽) |
| Total Success | 0 | 314,277회 | 많이 성공 |
| Collision | - | ~0 | 충돌 거의 없음 |

### 학습 그래프

![OSC Training 5500 iter](images/osc_training_5500iter.png)

### 분석
1. **Mean Reward**: 초반 -86,000까지 폭락 후 빠르게 회복 → 6000대로 안정
2. **Value Function Loss**: 초반 3억까지 발산 → 현재 1600~2000대 안정화 (진동 있으나 reward에 영향 없음)
3. **거리 메트릭**: 캡까지 3.7cm, 축 정렬 0.8cm (성공 기준 달성)
4. **에피소드 길이**: 450에 가까움 (끝까지 생존)

### 문제점
- 동작이 빠르고 급함 (stiffness=150이 높아서)
- Sim2Real 시 실제 로봇 동작과 차이 예상

### 모델 위치
```
/home/fhekwn549/e0509_osc/model_5500.pt
```

### 테스트 명령어
```bash
cd ~/IsaacLab
python pen_grasp_rl/scripts/play_osc.py --checkpoint /home/fhekwn549/e0509_osc/model_5500.pt --num_envs 50
```

---

## 실험 2: Soft OSC (예정)

### 목표
- 부드러운 동작으로 Sim2Real gap 최소화
- 실제 로봇 임피던스 특성에 맞춤

### 설정 변경
| 파라미터 | Default | Soft | 이유 |
|----------|---------|------|------|
| stiffness | 150 | 60 | 더 부드러운 반응 |
| action_scale | 0.05 | 0.03 | 더 작은 이동량 |
| damping_ratio | 1.0 | 1.0 | 임계 감쇠 유지 |

### 학습 명령어
```bash
# Docker 컨테이너 내에서 실행
cd /workspace/isaaclab
python3 pen_grasp_rl/scripts/train_osc.py --headless --num_envs 4096 --soft --fixed_lr
```

### 로그 위치
```
/workspace/isaaclab/pen_grasp_rl/logs/e0509_osc_soft/
```

### 결과
(학습 후 기록 예정)

---

## stiffness 선택 가이드

| stiffness | 특징 | 용도 |
|-----------|------|------|
| 150+ | 빠른 반응, 정확한 위치 | 시뮬레이션 전용 |
| 60-100 | 부드러운 반응 | **Sim2Real 권장** |
| 30-60 | 매우 부드러움 | 민감한 조작 |

### 중요: 실제 로봇 속도 제한 vs 학습 stiffness

**두 방식은 결과가 다릅니다!**

| 방식 | 설명 | 결과 |
|------|------|------|
| 낮은 stiffness로 학습 | 정책이 부드러운 동작을 학습 | Sim2Real gap 적음 |
| 실제 로봇 속도 제한만 | 정책은 빠른 동작, 로봇이 강제로 느림 | 타이밍 불일치, 예상치 못한 동작 |

→ **Sim2Real 전이 시 학습 때 사용한 stiffness와 비슷한 값으로 실제 로봇 설정 권장**

---

## 변경 파일 목록

| 파일 | 상태 | 설명 |
|------|------|------|
| `envs/e0509_osc_env.py` | NEW | OSC 환경 |
| `scripts/train_osc.py` | NEW | OSC 학습 스크립트 |
| `scripts/play_osc.py` | NEW | OSC 테스트 스크립트 |
| `config/pen_workspace.py` | NEW | Sim2Real 공유 설정 |
| `config/__init__.py` | NEW | Config 모듈 |
| `models/first_control.usd` | MODIFIED | 관절 제한 수정 |
| `envs/__init__.py` | MODIFIED | OSC 환경 등록 |
