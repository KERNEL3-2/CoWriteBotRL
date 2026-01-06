# 전문가 정책 (Expert Policy) 작업 로그

## 배경

### 문제점
- OSC 학습 실험 5A, 5B에서 dot mean (펜 정렬)이 개선되지 않음
- RL 학습된 정책이 우회 경로로 이동하는 경향
- 관절을 억지로 꺾는 동작 발생

### 해결 방향
- 규칙 기반 전문가 정책으로 최적 궤적 생성
- 모방 학습 (Behavior Cloning)으로 정책 학습
- Isaac Lab의 SkillGen과 동일한 접근 방식

---

## 생성된 파일

### 1. expert_policy.py
전문가 정책 및 데이터 수집 스크립트

**기능:**
- `SimpleExpertPolicy`: 캡 방향으로 직선 이동
- `ExpertPolicy`: 3단계 접근 (approach → descend → reach)
- 성능 평가 모드
- 궤적 수집 모드 (모방학습용)

**사용법:**
```bash
# 시각화 (눈으로 확인)
python expert_policy.py --mode play --num_envs 1

# 성능 평가
python expert_policy.py --mode eval --num_envs 64 --episodes 100

# 데이터 수집
python expert_policy.py --mode collect --num_envs 256 --episodes 1000 --output expert_data.pt
```

### 2. visualize_trajectories.py
수집된 궤적 시각화 스크립트

**기능:**
- 3D 궤적 플롯
- 2D 상단 뷰 (XY 평면)
- 통계 그래프 (에피소드 길이, 최종 거리, 경로 효율성)

**사용법:**
```bash
python visualize_trajectories.py --data expert_data.pt --output trajectory.png
```

---

## 전문가 정책 설계

### 관측값 구조 (27차원)
| 인덱스 | 항목 | 차원 |
|--------|------|------|
| 0:6 | joint_pos | 6 |
| 6:12 | joint_vel | 6 |
| 12:15 | grasp_pos_local | 3 |
| 15:18 | cap_pos_local | 3 |
| 18:21 | rel_pos (캡까지 상대위치) | 3 |
| 21:24 | pen_z (펜 축 방향) | 3 |
| 24 | perpendicular_dist | 1 |
| 25 | distance_to_cap | 1 |
| 26 | phase | 1 |

### SimpleExpertPolicy 알고리즘
```python
def get_action(obs):
    rel_pos = obs[:, 18:21]  # 캡까지 상대 위치
    dist = norm(rel_pos)

    # 정규화된 방향
    direction = rel_pos / dist

    # 거리에 비례하여 속도 조절
    speed_scale = clamp(dist / 0.1, min=0.2, max=1.0)

    return direction * move_speed * speed_scale
```

### 3단계 ExpertPolicy 알고리즘
1. **Phase 0 (Approach)**: 캡 위 5cm 지점으로 이동
2. **Phase 1 (Descend)**: 캡 위 1cm 지점으로 하강
3. **Phase 2 (Reach)**: 캡 위치 미세 조정

---

## 다음 단계

### 1. 전문가 성능 검증
```bash
python expert_policy.py --mode eval --num_envs 64 --episodes 100
```
- 목표: 성공률 95% 이상
- 평균 스텝 수 확인

### 2. 데이터 수집
```bash
python expert_policy.py --mode collect --num_envs 256 --episodes 5000 --output expert_data.pt
```
- 성공한 에피소드만 저장
- 다양한 펜 위치에서 수집

### 3. Behavior Cloning 학습
- Isaac Lab의 robomimic 통합 사용 가능
- 또는 직접 BC 스크립트 작성

### 4. 선택적 개선
- 효율성 페널티 추가 (action_penalty)
- 관절 속도 페널티 추가 (joint_vel_penalty)

---

## 참고 자료

- [Isaac Lab Imitation Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/index.html)
- [SkillGen Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)
- [MIT Underactuated Robotics - Imitation Learning](https://underactuated.mit.edu/imitation.html)

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-01-06 | expert_policy.py, visualize_trajectories.py 생성 |
