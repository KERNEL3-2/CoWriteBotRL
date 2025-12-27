# YOLO 펜 감지 작업 로그

## 개요
- **목표**: HSV 기반 펜 감지 → YOLO 기반으로 전환 (정확도 향상)
- **최종 목표**: 세그멘테이션으로 정확한 펜 끝점(캡/팁) 감지 → 3D 자세(pitch/yaw) 계산

---

## 완료된 작업

### 1. Eye-to-Hand 캘리브레이션
- **결과**: 1.47cm 오차
- **파일**: `/home/fhekwn549/doosan_ws/src/e0509_sim2real/scripts/`

### 2. 이미지 수집
- **스크립트**: `collect_images_simple.py`
- **수집량**: 100+ 장
- **저장 위치**: RealSense 카메라로 촬영

### 3. Roboflow 라벨링
- **프로젝트**: `pen-detect-vakdp/blue_pen`
- **라벨링 방식**: 폴리곤 (Auto-label with SAM)
- **URL**: https://universe.roboflow.com/pen-detect-vakdp/blue_pen/dataset/1

### 4. YOLOv8 Detection 학습 (완료)
- **모델**: yolov8n.pt
- **결과**: 99.5% mAP50
- **문제점**: 바운딩 박스만으로는 정확한 펜 끝점 감지 불가

### 5. 세그멘테이션 데이터셋 정리 (완료)
- **문제**: 혼합 라벨 형식 (박스 5개 + 세그먼트 97개)
- **해결**: 박스 형식 5개 파일 제거
  - `pen_0068_jpg.rf.*.txt`
  - `pen_0033_jpg.rf.*.txt`
  - `pen_0097_jpg.rf.*.txt`
  - `pen_0049_jpg.rf.*.txt`
  - `pen_0010_jpg.rf.*.txt`
- **현재 상태**: 96개 순수 세그멘테이션 라벨

---

## 다음 작업 (TODO)

### 1. YOLO 세그멘테이션 학습
```bash
cd ~
yolo segment train data=~/pen_dataset/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
```

### 2. pen_detector_yolo.py 업데이트
- 세그멘테이션 모델 로드
- 마스크에서 펜 끝점(캡/팁) 추출
- 정확한 3D pitch/yaw 계산

### 3. Sim2Real 통합
- 세그멘테이션 기반 펜 감지 → 로봇 제어

---

## 주요 파일 위치

| 파일 | 경로 |
|------|------|
| 펜 감지기 | `/home/fhekwn549/doosan_ws/src/e0509_sim2real/scripts/pen_detector_yolo.py` |
| 데이터셋 | `~/pen_dataset/` |
| data.yaml | `~/pen_dataset/data.yaml` |
| HSV 튜닝 | `/home/fhekwn549/doosan_ws/src/e0509_sim2real/scripts/tune_hsv.py` |

---

## 데이터셋 구조
```
~/pen_dataset/
├── data.yaml          # train/val 경로 설정
├── train/
│   ├── images/        # 학습 이미지
│   └── labels/        # 세그멘테이션 라벨 (96개)
└── test/
    └── images/        # 테스트 이미지
```

### data.yaml 내용
```yaml
train: ../train/images
val: ../train/images    # valid 폴더 없어서 train 재사용
test: ../test/images
nc: 1
names: ['pen']
```

---

## 참고사항
- 학습은 별도 터미널에서 실행 (Claude 타임아웃 문제)
- 세그멘테이션 모델: `yolov8n-seg.pt` 사용 (detection용 `yolov8n.pt` 아님)
