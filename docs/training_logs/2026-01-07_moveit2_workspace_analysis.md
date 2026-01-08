# 2026-01-07 작업 요약

## 개요
BC/ACT 모델이 펜 위치에 반응하지 않는 문제를 해결하기 위해 데이터 양을 늘리고, MoveIt2 경로 계획의 성공/실패 분포를 분석했습니다.

---

## 0. 프로젝트 배경 (모방학습)

### Diffusion Policy 시도
- **목적**: MoveIt2로 생성한 전문가 궤적으로 모방학습
- **문제**: Diffusion Policy는 실제 로봇에서 수집한 데이터가 필요
- **결론**: 시뮬레이션 데이터만으로는 적용 어려워 포기

### BC/ACT로 방향 전환
- MoveIt2 궤적은 시뮬레이션에서 생성 가능
- BC(Behavioral Cloning)와 ACT(Action Chunking with Transformers)로 모방학습 진행
- 100개 궤적으로 학습했으나 펜 위치에 반응하지 않는 문제 발생 → 오늘 작업의 시작점

---

## 1. BC/ACT 실패 원인 분석

### 문제 상황
- ACT 모델을 100개 궤적으로 학습했으나, 펜 위치가 달라져도 로봇이 동일한 움직임을 보임
- Action 표준편차가 0.0008로 거의 변화 없음

### 원인 파악
1. **관측값 정규화 부재**: obs_std가 매우 작아 모델이 입력 변화를 감지 못함
2. **데이터 부족**: 100개 궤적으로는 25차원 입력 → 6차원 출력 매핑 학습에 불충분
3. **Open-loop vs Closed-loop 불일치**: MoveIt2는 시간 기반 궤적, BC는 상태 기반 정책

---

## 2. 데이터 수집 (1,000개 궤적)

### 작업 내용
- 기존 `collect_bc_trajectories.py` 스크립트로 1,000개 궤적 수집
- 저장 위치: `~/generated_trajectories_1000/`

### 결과
- 총 시도: 1,530회
- 성공: 1,000개 (65.4%)
- 실패: 530개 (34.6%)

### 이유
- 100개 → 1,000개로 10배 증가시켜 모델이 펜 위치 변화를 학습할 수 있는 충분한 데이터 확보
- OPTIMUS 등 관련 논문에서도 수천 개 궤적 사용

---

## 3. 3D 시각화 스크립트 생성

### 생성 파일
`scripts/visualize_planning_3d.py`

### 기능
1. **3D 시각화**: Plotly를 사용한 인터랙티브 3D scatter plot
2. **기울기별 시각화**: 0-10°, 10-20°, 20-30° 세 구간으로 나눠서 표시
3. **영역별 분석**: X, Z, 기울기별 성공률 출력
4. **거리 기반 분석**: 로봇 베이스로부터의 XY/3D 거리별 성공률

### 이유
- 경로 계획 실패 원인을 파악하기 위해 공간적 분포 시각화 필요
- 로봇 workspace 경계가 구형으로 나타날 것으로 예상했으나, 실제로는 기울기가 주된 요인임을 발견

---

## 4. HDF5 변환 스크립트 생성

### 생성 파일
`scripts/convert_npz_to_hdf5.py`

### 기능
- MoveIt2가 생성한 npz 궤적들을 robomimic 형식의 HDF5로 변환
- Train/Valid 분할 (9:1)
- 관측값: joint_pos, joint_vel, ee_pos, ee_quat, pen_pos, pen_axis

### 결과
- 출력: `data/pen_grasp_1000.hdf5`
- 총 데모: 1,000개
- 총 스텝: 32,343개
- Train: 900개, Valid: 100개

### 이유
- ACT 학습 스크립트가 HDF5 형식을 사용
- robomimic 호환 형식으로 변환하여 다른 IL 알고리즘에도 활용 가능

---

## 5. 정규화 통계 업데이트

### 작업 내용
- 1,000개 궤적 데이터로 `norm_stats.npz` 재계산
- obs_mean, obs_std, action_mean, action_std 포함

### 이유
- 100개 데이터의 통계와 1,000개 데이터의 통계가 다를 수 있음
- 정규화가 제대로 되어야 모델이 입력 변화에 반응

---

## 6. ACT 학습 설정 업데이트

### 변경 사항
- `train_act.py`의 데이터 경로를 `pen_grasp_1000.hdf5`로 변경

### 현재 상태
- 학습 진행 중 (학습용 노트북에서 실행)

---

## 7. MoveIt2 Workspace 분석 결과

### 핵심 발견
1. **기울기가 가장 중요**: Tilt 0-10°는 88% 성공, 10° 이상은 53%로 급감
2. **거리 영향은 제한적**: 대부분 65-70%, 0.70m 이상에서만 42%로 급락
3. **구형 경계 부재**: 실패가 위치보다 orientation(자세)에 의존

### 의미
- IL 학습 시 기울기가 큰 펜 위치는 데이터가 상대적으로 적음
- 높은 성공률이 필요하면 Tilt 0-10° 범위로 제한 권장

---

## 생성/수정된 파일 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `scripts/visualize_planning_3d.py` | 신규 | 3D 시각화 및 분석 도구 |
| `scripts/convert_npz_to_hdf5.py` | 신규 | npz → HDF5 변환 |
| `data/pen_grasp_1000.hdf5` | 신규 | 1,000개 궤적 학습 데이터 |
| `pen_grasp_rl/imitation_learning/norm_stats.npz` | 수정 | 정규화 통계 업데이트 |
| `pen_grasp_rl/imitation_learning/train_act.py` | 수정 | 데이터 경로 변경 |

---

## 다음 단계

1. ~~ACT 학습 완료 후 테스트 - 펜 위치에 반응하는지 확인~~ ✅ 완료
2. ~~필요시 IL 큐레이션 적용~~ → BC+RL fine-tuning으로 방향 전환
3. ~~성능이 부족하면 DAgger 또는 RL fine-tuning 고려~~ ✅ BC+RL 진행

---

## 2026-01-08 후속 작업

### BC+RL Fine-tuning 진행
- ACT 시뮬레이션 테스트: 1.8% 성공률로 부족
- Residual Policy Learning 적용: BC + RL 결합
- 자세한 내용: `../2026-01-08_bc_rl_finetuning/training_log.md` 참고
