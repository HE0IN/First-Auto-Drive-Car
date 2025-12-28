# 🚗 First-Auto-Drive-Car

카메라 영상과 조이스틱 입력값을 학습하여 자율주행하는 RC카 프로젝트

> 전남대학교 전자통신공학부 학부 프로젝트 (2022.09 ~ 2022.12)

---

## 📌 프로젝트 소개

사람이 PS4 조이스틱으로 RC카를 조작할 때의 **카메라 영상**과 **조향/속도값**을 수집하고, 이를 TensorFlow CNN 모델로 학습시켜 카메라 영상만으로 자율주행이 가능한 RC카를 개발했습니다.

차체 외관은 Autodesk Fusion 360으로 직접 설계하고 3D 프린팅으로 제작했습니다.

---

## ✨ 주요 기능

- **데이터 수집** - PS4 조이스틱 조작 시 카메라 영상과 조향/속도값을 동시에 저장
- **데이터 밸런싱** - 직진 데이터 과다 수집 문제 해결을 위한 샘플링
- **이미지 증강** - Pan, Zoom, Brightness 랜덤 적용으로 학습 데이터 다양화
- **End-to-End 학습** - NVIDIA 자율주행 모델 아키텍처 기반 CNN 구현
- **실시간 자율주행** - 학습된 모델로 카메라 영상 기반 조향/속도 제어
- **안전 기능** - 초음파 센서 장애물 감지 및 빨간색 정지신호 인식

---

## 🛠 Tech Stack

| 분류 | 기술 |
|:---:|:---|
| **Hardware** | Raspberry Pi 4, Pi Camera, DC Motor, L298N Driver, PS4 Controller, HC-SR04 Ultrasonic |
| **AI/ML** | TensorFlow, Keras, NumPy, OpenCV, imgaug |
| **Language** | Python |
| **Design** | Autodesk Fusion 360, 3D Printing |

---

## 📁 파일 구조

```
First-Auto-Drive-Car/
│
├── 🎮 데이터 수집 (라즈베리파이)
│   ├── DataCollectionMain.py     # 데이터 수집 메인 실행
│   ├── DataCollectionModule.py   # 이미지/CSV 저장 모듈
│   ├── JoyStickModule.py         # PS4 조이스틱 입력 처리
│   ├── MotorModule.py            # DC 모터 PWM 제어
│   └── WebcamModule.py           # 카메라 영상 캡처
│
├── 🧠 모델 학습 (PC)
│   ├── Training.py               # 모델 학습 실행
│   ├── Training_Add.py           # 기존 모델에 추가 학습
│   ├── utils.py                  # 데이터 로드, 전처리, 모델 정의
│   └── utils_Add.py              # 추가 학습용 유틸리티
│
├── 🚗 자율주행 실행 (라즈베리파이)
│   ├── RunMain.py                # 자율주행 실행 (기본)
│   ├── Final.py                  # 자율주행 실행 (최종) - 초음파/정지신호 감지
│   ├── WebcamModule_Final.py     # 카메라 + 빨간색 정지신호 감지
│   └── testmodel.py              # 모델 테스트용
│
└── 📂 data/                      # 수집된 학습 데이터
    ├── IMG0/, IMG1/, ...             # 이미지 폴더
    └── log_0.csv, log_1.csv, ...     # 조향/속도 레이블
```

---

## 📄 파일별 설명

### 🎮 데이터 수집

| 파일 | 설명 |
|:---|:---|
| `DataCollectionMain.py` | 데이터 수집 메인 실행 파일. 조이스틱 입력을 받아 모터를 제어하고, Share 버튼으로 녹화 시작/종료 |
| `DataCollectionModule.py` | 이미지를 폴더에 저장하고 조향/속도값을 CSV로 기록 |
| `JoyStickModule.py` | pygame 기반 PS4 조이스틱 입력 처리. X/O/△/□로 속도 단계 조절 |
| `MotorModule.py` | L298N 모터 드라이버 PWM 제어. 조향값과 속도를 좌/우 모터 출력으로 변환 |
| `WebcamModule.py` | OpenCV로 카메라 영상 캡처 및 리사이즈 |

### 🧠 모델 학습

| 파일 | 설명 |
|:---|:---|
| `Training.py` | 처음부터 새 모델 학습. 데이터 경로, 로그 인덱스, 모델명 설정 |
| `Training_Add.py` | 기존 학습된 모델(.h5)에 추가 데이터로 Fine-tuning |
| `utils.py` | 핵심 유틸리티 - 데이터 로드, 밸런싱, 이미지 증강, 전처리, CNN 모델 생성 |

### 🚗 자율주행 실행

| 파일 | 설명 |
|:---|:---|
| `RunMain.py` | 기본 자율주행 실행. 카메라 → 모델 예측 → 모터 제어 |
| `Final.py` | 최종 자율주행. 초음파 센서(30cm 장애물 감지) + 빨간색 정지신호 인식 추가 |
| `WebcamModule_Final.py` | HSV 색공간에서 빨간색 객체 감지 기능이 추가된 카메라 모듈 |

---

## 🔄 시스템 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. 데이터 수집 (라즈베리파이)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   python DataCollectionMain.py                                  │
│                                                                 │
│   • PS4 조이스틱으로 RC카 조작                                    │
│   • Share 버튼: 녹화 시작/종료                                    │
│   • X/O/△/□: 속도 단계 (0.4 / 0.6 / 0.8 / 1.0)                  │
│   • 출력: IMG폴더 + log.csv                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       2. 모델 학습 (PC)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   python Training.py                                            │
│                                                                 │
│   • 데이터 로드 및 밸런싱                                         │
│   • 이미지 증강 (Pan, Zoom, Brightness)                          │
│   • 전처리: RGB→YUV, GaussianBlur, Resize(200x66)               │
│   • CNN 학습 (NVIDIA End-to-End 아키텍처)                        │
│   • 출력: model.h5                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. 자율주행 (라즈베리파이)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   python Final.py                                               │
│                                                                 │
│   • 카메라 영상 → 전처리 → 모델 예측                               │
│   • 예측값: [Steering, Throttle] → 모터 제어                      │
│   • 초음파: 30cm 이내 장애물 → 정지                                │
│   • 색상: 빨간색 정지신호 → 정지                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 모델 아키텍처

NVIDIA End-to-End Learning 논문 기반 CNN 모델

```
Input: (66, 200, 3)
        │
        ▼
┌───────────────────┐
│ Conv2D 24 (5x5)   │ stride=2, ELU
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Conv2D 36 (5x5)   │ stride=2, ELU, Dropout(0.2)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Conv2D 48 (5x5)   │ stride=2, ELU
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Conv2D 64 (3x3)   │ ELU
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Conv2D 64 (3x3)   │ ELU
└─────────┬─────────┘
          ▼
┌───────────────────┐
│     Flatten       │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Dense 100         │ ELU, Dropout(0.3)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Dense 50          │ ELU
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Dense 10          │ ELU
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Dense 2           │ Output: [Steering, Throttle]
└───────────────────┘
```

---

## 🚀 실행 방법

### 환경 설정

**라즈베리파이**
```bash
pip install opencv-python numpy tensorflow pygame RPi.GPIO
```

**PC (학습용)**
```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn imgaug
```

### 1. 데이터 수집

```bash
# 라즈베리파이에서 실행
python DataCollectionMain.py
```

### 2. 모델 학습

```bash
# PC에서 실행
python Training.py
```

`Training.py` 설정:
```python
data_path = "your/data/path"      # 데이터 경로
data_index = [0, 9]               # 로그 파일 범위 (log_0 ~ log_9)
new_model_name = 'model.h5'       # 저장할 모델명
```

### 3. 자율주행

```bash
# 라즈베리파이에서 실행
python Final.py
```

`Final.py` 설정:
```python
model = load_model('/path/to/model.h5')
throttleSen = 0.525    # 속도 민감도
steeringSen = 0.225    # 조향 민감도
```

---

## 🔧 하드웨어 핀 연결

| 모듈 | GPIO |
|:---:|:---:|
| Motor ENA | 0 |
| Motor IN1A | 6 |
| Motor IN2A | 5 |
| Motor ENB | 26 |
| Motor IN1B | 19 |
| Motor IN2B | 13 |
| Ultrasonic TRIG | 23 |
| Ultrasonic ECHO | 24 |

---

## 📝 배운 점

- 임베디드 환경에서의 딥러닝 모델 배포 및 최적화
- End-to-End Learning 방식의 자율주행 시스템 이해
- 실시간 영상 처리와 하드웨어 제어의 동기화
- 데이터 불균형 문제 해결 (직진 데이터 언더샘플링)
- Autodesk Fusion 360을 활용한 3D 모델링 및 프린팅

---

## 📄 License

This project is licensed under the MIT License.
#   - F i r s t - A u t o - D r i v e - C a r 
 
 
