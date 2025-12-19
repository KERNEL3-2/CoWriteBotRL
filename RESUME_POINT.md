# 작업 재개 지점

**마지막 작업일**: 2024-12-19

## 현재 상태

### IK V4 + 펜 각도 랜덤화 (Z축 원뿔 방식)

코드 수정 완료, **학습 대기 중**

**변경사항**:
- 기존 roll/pitch 개별 방식 → Z축 기준 원뿔 각도 방식
- `pen_tilt_max = 0.52` (최대 30도 기울기)
- `pen_yaw_range = (-3.14, 3.14)` (전체 회전)

**파일 위치**:
- 환경: `pen_grasp_rl/envs/e0509_ik_env_v4.py`
- 학습: `pen_grasp_rl/scripts/train_ik_v4.py`
- 테스트: `pen_grasp_rl/scripts/play_ik_v4.py`

## 이전 시도 (발산)

roll/pitch 방식으로 체크포인트에서 이어서 학습 → **발산**
- 원인: Noise Std 폭발 (10.5+), Adaptive LR 폭주
- 교훈: 새 도메인 추가 시 체크포인트 사용 주의

## 다음 할 일

### 1. 학습 실행 (다른 노트북에서)

```bash
# 1. git pull
cd ~/CoWriteBotRL && git pull

# 2. IsaacLab으로 동기화
cp pen_grasp_rl/envs/e0509_ik_env_v4.py ~/IsaacLab/pen_grasp_rl/envs/

# 3. 학습 시작 (처음부터 새로, 체크포인트 없이!)
cd ~/IsaacLab && source ~/isaacsim_env/bin/activate
python pen_grasp_rl/scripts/train_ik_v4.py --headless --num_envs 4096
```

### 2. 학습 결과 확인

- Mean Reward가 양수로 수렴하는지 확인
- Noise Std가 안정적인지 확인 (0.2~0.5 범위)
- Value Loss가 한자릿수~두자릿수인지 확인

### 3. Play 테스트

```bash
python pen_grasp_rl/scripts/play_ik_v4.py --checkpoint /path/to/model.pt --num_envs 32
```

### 4. (선택) Fixed LR로 변경

발산이 다시 발생하면 `train_ik_v4.py`에서:
```python
algorithm = RslRlPpoAlgorithmCfg(
    learning_rate=1e-4,      # 3e-4 → 1e-4
    schedule="fixed",        # "adaptive" → "fixed"
    ...
)
```

## 참고 문서

- 상세 로그: `DIRECT_TRAINING_LOG.md`
- 발산 그래프: `images/e0509_ik_v4_angle_diverged.png`
- V4 학습 그래프: `images/e0509_ik_v4_training.png`

## 핵심 포인트

1. **체크포인트 없이 처음부터 학습** (Distribution Shift 방지)
2. **Z축 원뿔 각도**: 정확히 최대 30도 기울기 제한
3. **발산 시 Fixed LR 사용** 고려
