# CoWriteBotRL

Doosan E0509 + RH-P12-RN-A 그리퍼를 이용한 펜 잡기 강화학습 프로젝트

## 개요

Isaac Lab 기반 강화학습으로 로봇이 펜을 적절한 자세로 잡는 방법을 학습합니다.

## 구조

```
CoWriteBotRL/
├── pen_grasp_rl/
│   ├── envs/                    # 강화학습 환경
│   │   └── pen_grasp_env.py
│   ├── scripts/
│   │   ├── train.py             # 학습 스크립트
│   │   ├── play.py              # 체크포인트 테스트
│   │   └── digital_twin.py      # Isaac Sim 디지털 트윈
│   └── models/                  # USD 모델 파일
└── e0509_gripper_isaac/         # 로봇 USD 파일
```

## 의존성

- Isaac Lab (Isaac Sim 4.5+)
- RSL-RL
- e0509_gripper_description (ROS2 패키지)

## 사용법

### 학습
```bash
source ~/isaacsim_env/bin/activate
cd ~/CoWriteBotRL
python pen_grasp_rl/scripts/train.py
```

### 체크포인트 테스트
```bash
source ~/isaacsim_env/bin/activate
cd ~/CoWriteBotRL
python pen_grasp_rl/scripts/play.py --checkpoint <checkpoint_path>
```

### Digital Twin (실제 로봇 시각화)

실제 로봇의 움직임을 Isaac Sim에서 실시간으로 시각화합니다.

**터미널 1: 로봇 실행**
```bash
ros2 launch e0509_gripper_description bringup.launch.py mode:=virtual
```

**터미널 2: ROS2 Bridge** (e0509_gripper_description 패키지)
[e0509_gripper_description](https://github.com/fhekwn549/e0509_gripper_description)
```bash
source /opt/ros/humble/setup.bash
source ~/doosan_ws/install/setup.bash
cd ~/doosan_ws/src/e0509_gripper_description/scripts
python3 digital_twin_bridge.py
```

**터미널 3: Isaac Sim 디지털 트윈**
```bash
source ~/isaacsim_env/bin/activate
cd ~/CoWriteBotRL
python pen_grasp_rl/scripts/digital_twin.py
```

## License

Apache-2.0
