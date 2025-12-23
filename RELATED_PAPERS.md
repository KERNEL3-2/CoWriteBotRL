# FSM + RL 기반 Zero-shot Sim2Real 관련 논문

## 현재 학습 방식과 유사한 접근법
- Phase-based State Machine + RL
- Hierarchical Control (Low-level + High-level)
- Curriculum Learning
- Zero-shot Sim2Real Transfer

---

## 1. Model-based + RL for Tight Insertion (2025)
**핵심**: Residual policy (Model-based control + RL correction)

- **구조**: Classical controller (low-level) + RL residual policy (high-level)
- **Curriculum**: noise level, action magnitude 점진적 증가
- **Sim2Real**: IsaacGym 학습 → **Zero-shot real world 전이 성공**
- **특징**: Franka Panda로 학습 → Kuka iiwa 14로 zero-shot 전이

**링크**: https://arxiv.org/html/2505.11858

---

## 2. Hierarchical Sim2Real for Multi-Agent Manipulation
**핵심**: Low-level goal-reaching + High-level RL controller

- **구조**: 시뮬레이션에서 low-level skills 학습 → high-level RL action space로 사용
- **Sim2Real**: **Zero-shot transfer** + Domain randomization
- **특징**: 전체 hierarchical policy를 real world에 직접 전이

**링크**: https://www.researchgate.net/publication/335173858_Multi-Agent_Manipulation_via_Locomotion_using_Hierarchical_Sim2Real

---

## 3. Tactile-based Manipulation Sim2Real (March 2024)
**핵심**: Tactile RL + Zero-shot transfer

- **학습**: 시뮬레이션에서 22개 객체로 학습
- **Sim2Real**: **Zero-shot transfer** → 16개 unseen 실제 객체에서 테스트
- **특징**: 다양한 객체에 대한 일반화 성공

**링크**: https://arxiv.org/abs/2403.12170

---

## 4. Trifinger Dexterous Manipulation (IsaacGym)
**핵심**: GPU 시뮬레이션 → Real TriFinger 전이

- **환경**: IsaacGym
- **Sim2Real**: Dexterous manipulation 성공 사례
- **특징**: IsaacGymEnvs 공식 예제에 포함

**링크**: https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/rl_examples.md

---

## 5. Dual-Arm Assembly + Curriculum Learning (2024)
**핵심**: Hierarchical control + Reverse curriculum

- **구조**: Classical controller (low-level) + RL policy (high-level)
- **Curriculum**: 난이도 점진적 증가 (reverse curriculum)
- **Sim2Real**: Assembly task 전이

**링크**: https://www.mdpi.com/2075-1702/12/10/682

---

## 6. Humanoid Robot Vehicle Ingress (2024)
**핵심**: FSM + DRL 통합

- **구조**: Finite State Machine으로 단계 분리 + 각 상태별 다른 reward
- **핵심 인사이트**: "전체 과정에 동일한 reward를 쓰면 다양한 상태를 가진 task에 적합하지 않다"
- **특징**: 현재 학습 방식과 가장 유사한 접근법

**링크**: https://link.springer.com/article/10.1007/s13042-024-02407-w

---

## 7. Vision-Based Dexterous Manipulation on Humanoids (2025)
**핵심**: Divide-and-conquer policy distillation

- **Task**: grasp-and-reach, box lift, bimanual handover
- **구조**: automated real-to-sim tuning + contact/object goals reward
- **Sim2Real**: Zero-shot transfer

**링크**: https://arxiv.org/abs/2502.20396

---

## 8. TIAGo + Isaac Sim Sim2Real Gap Study (2024)
**핵심**: Isaac Sim/Gym 기반 sim2real gap 분석

- **환경**: Isaac Sim, Isaac Gym
- **내용**: Sim2Real gap 원인 분석 및 해결 방법

**링크**: https://arxiv.org/html/2403.07091v2

---

## 추가 참고 자료

### Deep RL for Robotics Survey (Annual Reviews)
- 실제 로봇에서 성공한 Deep RL 사례 종합
- https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-030323-022510

### EAGERx: Sim2Real Robot Learning Framework
- 통합 sim2real 프레임워크
- Domain randomization, delay simulation 지원
- https://arxiv.org/html/2407.04328

---

## 검색 키워드 (추가 조사용)
- "hierarchical reinforcement learning" manipulation sim2real
- "finite state machine" deep reinforcement learning robot
- "task decomposition" RL zero-shot transfer
- "residual policy" manipulation sim2real
- "curriculum learning" manipulation real robot
