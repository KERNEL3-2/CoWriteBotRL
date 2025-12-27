# IK V2 작업 요약 (2024-12-18)

## 현재 상태: IK V2 학습 완료, 문제 발견

---

## 오늘 한 일

### 1. IK V2 환경 수정
- **ALIGN 단계 추가**: PRE_GRASP → ALIGN → DESCEND 3단계
- **펜 캡 위치 기준 생성**: 캡이 z=0.25~0.40 높이에 오도록
- **전환 조건 분리**:
  - PRE_GRASP → ALIGN: 거리 < 3cm (정렬 조건 없음)
  - ALIGN → DESCEND: dot < -0.95

### 2. IK V2 학습 결과
| 항목 | 값 |
|------|-----|
| 최고 Reward | 2229 (step 818) |
| 최종 Reward | 1105 (step 1999) |
| Episode Length | 359 (타임아웃) |
| 성공 횟수 | 0 |

### 3. Play 테스트 디버깅 결과
```
dist_pregrasp = 0.0796m (7.96cm) → 3cm 필요 ❌
dist_cap = 0.0198m (1.98cm) → 펜캡에 가까움 ✓
dot = -0.863 → -0.95 필요 ❌
```

---

## 발견된 문제

### 그리퍼가 펜캡 "위"가 아닌 "옆"으로 접근

```
의도한 경로:              실제 경로:

   그리퍼                    그리퍼 ──┐
      ↓                               ↓
  pre-grasp (캡 위 7cm)            펜캡 ← (옆에서 접근)
      ↓
    펜캡
```

- 펜캡까지 거리: 2cm (매우 가까움)
- pre-grasp 위치까지 거리: 8cm (멀다)
- **원인**: PRE_GRASP 보상에서 pregrasp_dist 페널티가 약함

---

## 내일 검토할 해결 방안

### 옵션 1: PRE_GRASP 거리 페널티 강화
```python
# 현재
rew_scale_pregrasp_dist = -8.0

# 강화 (예시)
rew_scale_pregrasp_dist = -15.0
```

### 옵션 2: 전환 조건 변경
```python
# 현재: pre-grasp 위치 기준
PRE_GRASP → ALIGN: dist_pregrasp < 3cm

# 변경: 펜캡 위치 기준 + 높이 조건
PRE_GRASP → ALIGN: dist_cap < 10cm AND grasp_z > cap_z + 5cm
```

### 옵션 3: TensorBoard extras 로깅 추가
학습 중 실시간으로 거리/정렬 모니터링
```python
self.extras["log"] = {
    "phase_pre_grasp_ratio": ...,
    "phase_align_ratio": ...,
    "phase_descend_ratio": ...,
    "dist_to_pregrasp": ...,
    "dist_to_cap": ...,
    "dot_product": ...,
}
```

---

## 파일 위치

| 파일 | 경로 |
|------|------|
| IK 환경 | `pen_grasp_rl/envs/e0509_ik_env.py` |
| 학습 스크립트 | `pen_grasp_rl/scripts/train_ik.py` |
| 테스트 스크립트 | `pen_grasp_rl/scripts/play_ik.py` |
| 체크포인트 | `/home/fhekwn549/e0509_ik/model_1999.pt` |
| 전체 로그 | `DIRECT_TRAINING_LOG.md` |

---

## 실행 명령어

### 학습 (별도 터미널)
```bash
source ~/isaacsim_env/bin/activate
cd ~/IsaacLab
python pen_grasp_rl/scripts/train_ik.py --headless --num_envs 4096 --max_iterations 5000
```

### 테스트 (디버깅 출력 포함)
```bash
cd ~/IsaacLab
python pen_grasp_rl/scripts/play_ik.py --checkpoint /home/fhekwn549/e0509_ik/model_1999.pt --num_envs 16
```

### 동기화 (코드 수정 후)
```bash
cp /home/fhekwn549/CoWriteBotRL/pen_grasp_rl/envs/e0509_ik_env.py /home/fhekwn549/IsaacLab/pen_grasp_rl/envs/
cp /home/fhekwn549/CoWriteBotRL/pen_grasp_rl/scripts/*.py /home/fhekwn549/IsaacLab/pen_grasp_rl/scripts/
```

---

## 핵심 질문 (내일 결정)

1. **pre-grasp 위치 개념 유지 vs 폐기?**
   - 유지: "위에서 접근" 전략 계속
   - 폐기: 바로 펜캡으로 접근하고 정렬만 신경

2. **보상 구조 어떻게 조정?**
   - pregrasp_dist 페널티 강화?
   - 다른 보상 축소?
   - 완전히 새로운 접근?

3. **모니터링 추가 먼저?**
   - extras 로깅 먼저 추가해서 학습 중 상황 파악
   - 그 다음 보상 조정
