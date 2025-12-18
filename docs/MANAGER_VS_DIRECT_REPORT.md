# Manager-based vs Direct 환경 비교 리포트

## 개요

이 문서는 Isaac Lab에서 펜 잡기(Pen Grasp) 강화학습 환경을 개발하면서 **Manager-based** 방식에서 **Direct** 방식으로 전환한 과정과 그 이유를 정리합니다.

**학습 기간**: 2025-12-11 ~ 2025-12-18
**총 실험 횟수**: 약 20회 이상
**총 학습 iteration**: 약 500,000+

---

## 1. Isaac Lab 환경 구조 비교

### 1.1 Manager-based 환경 (`ManagerBasedRLEnv`)

```
┌─────────────────────────────────────────────────────────┐
│                    ManagerBasedRLEnv                     │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Observation  │  │   Reward     │  │   Action     │   │
│  │   Manager    │  │   Manager    │  │   Manager    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│         │                 │                 │           │
│  ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐   │
│  │ ObsTerm 1   │   │ RewTerm 1   │   │ ActionTerm  │   │
│  │ ObsTerm 2   │   │ RewTerm 2   │   └─────────────┘   │
│  │ ...         │   │ ...         │                     │
│  └─────────────┘   └─────────────┘                     │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ Termination  │  │   Event      │                     │
│  │   Manager    │  │   Manager    │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

**특징**:
- 모듈화된 구성 요소 (Manager들)
- 설정(Cfg) 클래스를 통한 선언적 정의
- 각 Term이 독립적으로 동작
- Isaac Lab 예제들이 주로 이 방식 사용

### 1.2 Direct 환경 (`DirectRLEnv`)

```
┌─────────────────────────────────────────────────────────┐
│                      DirectRLEnv                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              _get_observations()                 │   │
│  │  → 직접 텐서 연산으로 관찰값 구성                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              _get_rewards()                      │   │
│  │  → 직접 텐서 연산으로 보상 계산                    │   │
│  │  → 상태 머신 로직 포함 가능                       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              _apply_action()                     │   │
│  │  → 직접 액션 적용                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**특징**:
- 단일 클래스에서 모든 로직 직접 구현
- PyTorch 텐서 연산 직접 사용
- 상태 머신, 조건부 로직 구현 용이
- 디버깅 및 커스터마이징 유연

---

## 2. Manager-based 환경 실험 기록

### 2.1 V1 환경 (2025-12-11 ~ 12-15)

**파일**: `pen_grasp_env.py`

| 항목 | 값 |
|------|-----|
| 관찰 차원 | 36 |
| 보상 함수 | 7개 |
| 총 학습 | ~200,000 iterations |

**보상 함수 구성**:
```python
# 7개의 독립적인 보상 Term
rewards = {
    "distance_to_cap": RewTerm(func=..., weight=1.0),
    "z_axis_alignment": RewTerm(func=..., weight=0.5),
    "floor_collision": RewTerm(func=..., weight=1.0),
    "pen_displacement": RewTerm(func=..., weight=1.0),
    "alignment_success": RewTerm(func=..., weight=2.0),
    "action_rate": RewTerm(func=..., weight=0.1),
    "base_orientation": RewTerm(func=..., weight=0.5),
}
```

**주요 문제점**:

| 문제 | 원인 | 결과 |
|------|------|------|
| 학습 발산 | 보상 함수 간 충돌 | Value Loss 12.4T |
| 측면 접근 | 보상 구조가 방향 무관 | 펜과 충돌 |
| 정렬 실패 | "절벽 보상" 구조 | z_axis_alignment = 0 |
| 에피소드 즉시 종료 | 펜 밀림 → 종료 조건 | 학습 시간 부족 |

**학습 결과 (200K iteration)**:
```
Mean Reward: 0.05 → -1,600 (발산)
Episode Length: 5.1 → 2.0 (즉시 종료)
Value Loss: 0.01 → 12.4T (완전 발산)
pen_dropped: 99.9%
```

### 2.2 V2 환경 (2025-12-16)

**파일**: `pen_grasp_env_v2.py`

| 항목 | 값 |
|------|-----|
| 관찰 차원 | 27 |
| 보상 함수 | 4개 (단순화) |
| 기반 | Isaac Lab reach 예제 |

**변경 사항**:
- 보상 함수 7개 → 4개로 단순화
- reach 예제의 검증된 보상 형태 사용 (L2, tanh)
- 관찰 공간 정리 (36 → 27차원)

**보상 함수 구성**:
```python
# 4개의 단순화된 보상
rewards = {
    "position_error": RewTerm(weight=-0.5),    # L2 거리
    "position_fine": RewTerm(weight=+1.0),     # tanh 커널
    "orientation_error": RewTerm(weight=-0.3), # dot product
    "action_rate": RewTerm(weight=-0.001),
}
```

**학습 결과 (5,800 iteration)**:
```
position_fine: 0.006 → 0.72 (+11,254%)
orientation_error: -0.112 → -0.016 (+86%)
mean_reward: -2.03 → 6.90 (+440%)
```

**개선점**:
- 학습 안정성 향상 (noise_std 안정)
- 방향 학습 성공 (V1에서는 전혀 안됨)

**남은 문제점**:
- 여전히 측면에서 접근
- 정밀도 부족 (평균 3cm, 목표 2cm)
- 로컬 최적화에 갇힘

### 2.3 V3 환경 - Curriculum Learning (2025-12-16 ~ 12-17)

**파일**: `pen_grasp_env_v3.py`

| 항목 | 값 |
|------|-----|
| 특징 | Curriculum Learning |
| Stage 수 | 3단계 |
| 목표 성공률 | Stage별 85% → 90% → 95% |

**Stage 구성**:
```
Stage 1: 거리 < 10cm, dot < -0.70, 목표 85%
Stage 2: 거리 < 5cm, dot < -0.85, 목표 90%
Stage 3: 거리 < 2cm, dot < -0.95, 목표 95%
```

**결과**:
- Stage 1 달성은 비교적 쉬움
- Stage 2 이후 진전 없음
- **근본적 한계**: Manager 구조에서 상태 의존적 보상 구현 어려움

---

## 3. Manager-based의 근본적 한계

### 3.1 보상 함수 독립성 문제

Manager-based에서 각 RewardTerm은 **독립적으로 계산**됩니다:

```python
# Manager 내부 동작 (의사 코드)
total_reward = 0
for term in reward_terms:
    total_reward += term.func(env) * term.weight
```

**문제**:
- "먼저 위치 맞추고 → 그다음 정렬" 같은 **순차적 목표** 구현 어려움
- 각 Term이 서로의 상태를 모름
- 복잡한 조건부 로직 구현 제한적

### 3.2 상태 머신 구현의 어려움

펜 잡기 태스크의 이상적인 동작 순서:
```
1. PRE_GRASP: 펜 캡 위 7cm로 이동 + Z축 정렬
2. DESCEND: 정렬 유지하며 수직 하강
3. GRASP: 펜 잡기
```

**Manager-based 시도**:
```python
# 조건부 보상으로 상태 머신 흉내
def distance_reward(env):
    if distance < 0.1:  # 가까우면
        return alignment_bonus  # 정렬 보상
    else:
        return distance_bonus   # 거리 보상
```

**문제점**:
- 환경 간 상태 공유 어려움 (각 env가 어느 단계인지)
- 명시적인 상태 전이 로직 구현 불가
- 디버깅 어려움 (어느 Term이 문제인지 파악 힘듦)

### 3.3 텐서 연산 제약

Manager-based에서는 각 Term 함수가 `(num_envs,)` 형태의 텐서를 반환해야 합니다:

```python
def reward_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    # 복잡한 로직을 단일 텐서 반환으로 압축해야 함
    return torch.Tensor  # shape: (num_envs,)
```

**제약**:
- 중간 계산 결과 재사용 어려움
- 여러 조건의 조합 표현 제한적
- 환경별 다른 로직 적용 어려움

---

## 4. Direct 환경으로 전환

### 4.1 Direct 환경의 장점

| 항목 | Manager-based | Direct |
|------|---------------|--------|
| 상태 머신 | 어려움 | **자연스러움** |
| 조건부 로직 | 제한적 | **자유로움** |
| 디버깅 | 분산됨 | **집중됨** |
| 코드 가독성 | 분리됨 | **통합됨** |
| 커스터마이징 | Manager 제약 | **완전 자유** |

### 4.2 Direct 환경 구현 (e0509_direct_env.py)

**파일**: `pen_grasp_rl/envs/e0509_direct_env.py`

```python
class E0509DirectEnv(DirectRLEnv):
    # 상태 머신 상수
    PHASE_PRE_GRASP = 0
    PHASE_DESCEND = 1

    def __init__(self, cfg):
        super().__init__(cfg)
        # 환경별 현재 단계 추적
        self.current_phase = torch.zeros(self.num_envs, device=self.device)

    def _get_rewards(self):
        # 1. 공통 계산 (재사용 가능)
        distance = self._compute_distance()
        dot = self._compute_alignment()

        # 2. 단계별 다른 보상 (상태 머신)
        reward = torch.zeros(self.num_envs, device=self.device)

        # PRE_GRASP 단계
        pre_grasp_mask = (self.current_phase == self.PHASE_PRE_GRASP)
        reward[pre_grasp_mask] = self._pre_grasp_reward(distance, dot)[pre_grasp_mask]

        # DESCEND 단계
        descend_mask = (self.current_phase == self.PHASE_DESCEND)
        reward[descend_mask] = self._descend_reward(distance, dot)[descend_mask]

        # 3. 단계 전이 로직
        self._update_phase(distance, dot)

        return reward
```

### 4.3 상태 머신 구현

```
┌─────────────────────────────────────────────────────────┐
│                    상태 머신                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────────┐         ┌──────────────┐            │
│   │  PRE_GRASP   │ ──────→ │   DESCEND    │            │
│   │              │ 거리<3cm │              │            │
│   │ 펜캡 위 7cm  │ dot<-0.95│ 수직 하강    │            │
│   │ + Z축 정렬   │         │ + 정렬 유지  │            │
│   └──────────────┘         └──────────────┘            │
│         │                         │                     │
│         │ 보상:                   │ 보상:              │
│         │ - 거리 페널티 (-8.0)   │ - 거리 페널티 (-10.0)│
│         │ - 진행 보상 (+20.0)    │ - 하강 보상 (+15.0) │
│         │ - 정렬 보상 (+1.5)     │ - 정렬 유지 (+2.0)  │
│         │ - 전환 보너스 (+15.0)  │ - 성공 보너스(+100) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.4 Direct 환경 학습 결과

**V3 (10,000 iteration)**:
```
Mean Reward: 0 → 771.77 (최고)
Episode Length: 359 (max)
```

**문제점 발견 및 V4 개선**:
- 로봇 링크 겹침 → 작업 공간 기반 관절 한계 추가
- 과격한 움직임 → action_scale 0.1 → 0.05
- 특이점 페널티 부작용 → 비활성화 (관절 한계로 대체)

---

## 5. 결론

### 5.1 Manager-based가 적합한 경우

- 단순한 reach/track 태스크
- Isaac Lab 예제와 유사한 구조
- 빠른 프로토타이핑
- 표준화된 벤치마크

### 5.2 Direct가 필요한 경우

- **복잡한 순차적 태스크** (펜 잡기처럼)
- 상태 머신이 필요한 경우
- 조건부 보상 로직이 복잡한 경우
- 세밀한 제어 및 디버깅 필요

### 5.3 핵심 교훈

| 교훈 | 설명 |
|------|------|
| **구조 선택이 중요** | 태스크 복잡도에 맞는 환경 구조 선택 |
| **단순화의 함정** | Manager-based 단순화가 항상 좋은 것은 아님 |
| **상태 머신의 필요성** | 순차적 태스크에는 명시적 상태 관리 필요 |
| **디버깅 용이성** | Direct가 문제 파악에 훨씬 유리 |

### 5.4 최종 환경 비교

| 항목 | V1 (Manager) | V2 (Manager) | V3 (Manager) | Direct |
|------|--------------|--------------|--------------|--------|
| 관찰 차원 | 36 | 27 | 27 | **27** |
| 보상 함수 | 7개 | 4개 | 4개+Stage | **상태별** |
| 상태 머신 | X | X | X | **O** |
| 학습 안정성 | 발산 | 안정 | 안정 | **안정** |
| 정밀도 달성 | 실패 | 부분 | 부분 | **진행중** |
| 디버깅 | 어려움 | 중간 | 중간 | **쉬움** |

---

## 6. 참고 자료

- Isaac Lab 공식 문서: [Environment Design](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/index.html)
- 학습 로그: `docs/TRAINING_LOG.md`, `docs/TRAINING_LOG_V2.md`, `docs/TRAINING_LOG_V3.md`
- Direct 환경 로그: `DIRECT_TRAINING_LOG.md`

---

**작성일**: 2025-12-18
**작성자**: Claude (with human guidance)
