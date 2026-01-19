# 설비 이상 동작 검출 시스템 (MovAnal)

## 1. 프로젝트 개요

### 1.1 목적
조립라인 설비 내부를 촬영한 동영상을 분석하여 **정상 동작 패턴을 학습**하고, 이를 기반으로 **이상 동작을 실시간으로 검출**하는 시스템 개발

### 1.2 핵심 특징
- **비지도 학습 기반**: 정상 데이터만으로 학습 (이상 샘플 불필요)
- **범용성**: 다양한 설비 형태에 적용 가능한 유연한 구조
- **실시간 처리**: 영상 스트림에서 즉시 이상 감지
- **적응형 학습**: 조명 변화, 그림자 등 환경 변화 대응

---

## 2. 시스템 요구사항

### 2.1 기능 요구사항

| 구분 | 요구사항 | 우선순위 |
|------|----------|----------|
| FR-01 | 동영상 파일 및 실시간 스트림 입력 지원 | 필수 |
| FR-02 | 정상 동작 패턴 학습 기능 | 필수 |
| FR-03 | 실시간 이상 동작 검출 | 필수 |
| FR-04 | 속도 변화 감지 | 필수 |
| FR-05 | 비정상 움직임 패턴 감지 | 필수 |
| FR-06 | 조명/그림자 변화 대응 | 필수 |
| FR-07 | 이상 발생 시 알림 및 로깅 | 필수 |
| FR-08 | 학습 모델 저장 및 로드 | 필수 |
| FR-09 | 설비별 독립 모델 관리 | 권장 |
| FR-10 | 웹 기반 모니터링 대시보드 | 선택 |

### 2.2 비기능 요구사항

| 구분 | 요구사항 | 목표값 |
|------|----------|--------|
| NFR-01 | 처리 지연시간 | < 100ms (30fps 기준) |
| NFR-02 | 검출 정확도 | Precision > 90%, Recall > 85% |
| NFR-03 | 오탐율 (False Positive) | < 5% |
| NFR-04 | GPU 메모리 사용량 | < 4GB |
| NFR-05 | 동시 처리 카메라 수 | 최소 4대 (단일 서버) |

### 2.3 환경 조건

```
카메라: 고정형 (위치/각도 불변)
해상도: 1920x1080 (Full HD) 권장
프레임율: 30fps
조명: 기본 일정, 설비 동작에 따른 그림자 변화 존재
동작 특성: 반복 패턴 존재 (사이클 타임 기반)
```

---

## 3. 시스템 아키텍처

### 3.1 전체 구성도

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MovAnal System                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Camera  │───▶│ Video Input  │───▶│ Preprocessor │              │
│  │  Stream  │    │   Module     │    │              │              │
│  └──────────┘    └──────────────┘    └──────┬───────┘              │
│                                              │                       │
│                                              ▼                       │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                  Feature Extraction                       │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │      │
│  │  │  Optical   │  │   Motion   │  │    Appearance      │  │      │
│  │  │   Flow     │  │   History  │  │    Features        │  │      │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │      │
│  └──────────────────────────┬───────────────────────────────┘      │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              Anomaly Detection Engine                     │      │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐  │      │
│  │  │   Autoencoder   │    │   Temporal Pattern Model    │  │      │
│  │  │   (Spatial)     │    │   (LSTM/Transformer)        │  │      │
│  │  └─────────────────┘    └─────────────────────────────┘  │      │
│  └──────────────────────────┬───────────────────────────────┘      │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐      │
│  │   Anomaly    │───▶│    Alert     │───▶│   Dashboard /    │      │
│  │   Scoring    │    │   Manager    │    │   Logging        │      │
│  └──────────────┘    └──────────────┘    └──────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 모듈 구성

#### 3.2.1 Video Input Module
- 역할: 영상 소스 관리 및 프레임 추출
- 지원 입력:
  - RTSP 스트림 (IP 카메라)
  - USB 카메라 (DirectShow/V4L2)
  - 동영상 파일 (MP4, AVI, MKV)
- 버퍼링 및 프레임 동기화

#### 3.2.2 Preprocessor
- 역할: 영상 전처리 및 정규화
- 기능:
  - 해상도 조정 (처리 효율화)
  - 노이즈 제거
  - **조명 정규화** (히스토그램 평활화, CLAHE)
  - ROI(Region of Interest) 설정

#### 3.2.3 Feature Extraction
- 역할: 이상 검출을 위한 특징 추출

| 특징 유형 | 추출 방법 | 검출 대상 |
|-----------|-----------|-----------|
| Optical Flow | Farneback / RAFT | 속도 변화, 움직임 방향 |
| Motion History | 프레임 차분 누적 | 동작 패턴, 정지 감지 |
| Appearance | CNN Encoder | 형태 변화, 이물질 |
| Temporal | 시계열 임베딩 | 주기성, 타이밍 이상 |

#### 3.2.4 Anomaly Detection Engine
- 역할: 정상 패턴 학습 및 이상 판별

**핵심 모델 구조:**
```
1. Convolutional Autoencoder (CAE)
   - 입력: 전처리된 프레임 시퀀스
   - 출력: 재구성 오차 (Reconstruction Error)
   - 원리: 정상 패턴만 학습 → 이상 시 재구성 실패 → 높은 오차

2. Variational Autoencoder (VAE)
   - 잠재 공간에서의 확률적 모델링
   - 이상 샘플의 likelihood 기반 검출

3. LSTM/Transformer 기반 시계열 모델
   - 입력: 연속 프레임의 특징 벡터
   - 출력: 다음 프레임 예측
   - 원리: 예측 오차가 크면 이상 판정
```

#### 3.2.5 Anomaly Scoring
- 역할: 이상 점수 산출 및 임계값 판정
- 방법:
  - 재구성 오차 기반 점수
  - Z-score 정규화
  - 동적 임계값 (Moving Average 기반)
  - 앙상블 스코어링

#### 3.2.6 Alert Manager
- 역할: 이상 감지 시 알림 처리
- 기능:
  - 로그 기록 (시간, 프레임, 점수, 영역)
  - 이상 구간 영상 클립 저장
  - 외부 시스템 연동 (REST API, MQTT)
  - 알림 중복 방지 (Debouncing)

---

## 4. 데이터 처리 파이프라인

### 4.1 학습 단계 (Training Phase)

```
[정상 동작 영상]
       │
       ▼
┌─────────────────┐
│ 영상 분할       │  → 설비 사이클 단위로 분할
│ (Cycle Split)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 전처리          │  → 정규화, 노이즈 제거, 조명 보정
│ (Preprocess)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 특징 추출       │  → Optical Flow, Motion History
│ (Extract)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 모델 학습       │  → Autoencoder 학습
│ (Train Model)   │     - 재구성 학습
└────────┬────────┘     - 시계열 패턴 학습
         │
         ▼
┌─────────────────┐
│ 임계값 설정     │  → 검증 데이터로 최적 임계값 결정
│ (Calibrate)     │
└────────┬────────┘
         │
         ▼
   [학습된 모델 저장]
```

### 4.2 추론 단계 (Inference Phase)

```
[실시간 영상 스트림]
       │
       ▼
┌─────────────────┐
│ 프레임 버퍼     │  → 연속 N 프레임 유지
│ (Frame Buffer)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 전처리          │  → 학습과 동일한 전처리
│ (Preprocess)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 특징 추출       │  → 실시간 특징 계산
│ (Extract)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 이상 점수 계산  │  → 모델 추론
│ (Inference)     │     - 재구성 오차 계산
└────────┬────────┘     - 예측 오차 계산
         │
         ▼
┌─────────────────┐
│ 판정            │  → 임계값 비교
│ (Decision)      │     - 연속성 검사
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
 [정상]    [이상]
              │
              ▼
        ┌─────────────────┐
        │ 알림 처리       │
        │ (Alert)         │
        └─────────────────┘
```

---

## 5. 이상 검출 알고리즘 상세

### 5.1 속도 변화 검출

**방법: Optical Flow 기반 속도 분석**

```python
# 개념적 알고리즘
1. 연속 프레임 간 Optical Flow 계산
2. Flow 벡터의 크기(magnitude) 계산
3. ROI 영역별 평균 속도 산출
4. 정상 속도 분포와 비교
   - 급격한 속도 증가/감소 감지
   - 특정 영역의 속도 이상 감지
```

**검출 대상:**
- 설비 동작 속도 저하/과속
- 갑작스러운 정지
- 비정상적 진동

### 5.2 비정상 패턴 검출

**방법 1: Autoencoder 재구성 오차**

```
원리:
- 정상 데이터로만 학습된 Autoencoder는 정상 패턴을 잘 재구성
- 이상 패턴 입력 시 재구성 품질 저하 → 높은 오차

Anomaly Score = MSE(Original, Reconstructed)
```

**방법 2: 시계열 예측 오차**

```
원리:
- LSTM/Transformer로 다음 프레임 특징 예측
- 예측과 실제의 차이가 크면 이상

Anomaly Score = ||Predicted - Actual||
```

### 5.3 조명 변화 대응 전략

```
1. 전처리 단계
   - Adaptive Histogram Equalization (CLAHE)
   - 배경 모델링 및 차분

2. 학습 단계
   - 다양한 조명 조건의 정상 데이터 포함
   - Data Augmentation (밝기, 대비 변화)

3. 추론 단계
   - 조명 변화 감지 시 가중치 조정
   - 그림자 영역 마스킹 옵션
```

### 5.4 이상 판정 로직

```
┌─────────────────────────────────────────────────────────┐
│                  Anomaly Decision Logic                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  score = w1 * reconstruction_error                       │
│        + w2 * prediction_error                           │
│        + w3 * speed_anomaly_score                        │
│                                                          │
│  if score > threshold:                                   │
│      if consecutive_anomaly_count >= min_frames:         │
│          → ANOMALY DETECTED                              │
│      else:                                               │
│          → INCREMENT counter                             │
│  else:                                                   │
│      → RESET counter                                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 기술 스택

### 6.1 개발 환경

| 구분 | 기술 | 버전 |
|------|------|------|
| 언어 | Python | 3.10+ |
| 딥러닝 | PyTorch | 2.0+ |
| 영상처리 | OpenCV | 4.8+ |
| 데이터처리 | NumPy, Pandas | - |
| 시각화 | Matplotlib, Plotly | - |
| 설정관리 | Hydra / YAML | - |
| 로깅 | Python logging, TensorBoard | - |

### 6.2 선택적 기술

| 구분 | 기술 | 용도 |
|------|------|------|
| 모델 최적화 | ONNX Runtime, TensorRT | 추론 가속화 |
| 웹 서버 | FastAPI | REST API 제공 |
| 대시보드 | Streamlit / Gradio | 모니터링 UI |
| 메시지큐 | Redis / MQTT | 알림 연동 |
| 컨테이너 | Docker | 배포 |

### 6.3 하드웨어 권장사양

**최소 사양 (개발/테스트):**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB
- GPU: NVIDIA GTX 1660 (6GB)
- Storage: SSD 256GB

**권장 사양 (운영):**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 32GB
- GPU: NVIDIA RTX 3060 (12GB) 이상
- Storage: SSD 512GB+

---

## 7. 프로젝트 구조

```
MovAnal/
├── docs/                      # 문서
│   ├── SYSTEM_DESIGN.md       # 본 문서
│   └── API.md                 # API 명세
│
├── src/
│   ├── __init__.py
│   ├── main.py                # 진입점
│   │
│   ├── input/                 # 영상 입력 모듈
│   │   ├── __init__.py
│   │   ├── video_source.py    # 영상 소스 추상화
│   │   ├── camera.py          # 카메라 연결
│   │   └── file_reader.py     # 파일 읽기
│   │
│   ├── preprocessing/         # 전처리 모듈
│   │   ├── __init__.py
│   │   ├── normalizer.py      # 정규화
│   │   ├── roi_manager.py     # ROI 관리
│   │   └── augmentation.py    # 데이터 증강
│   │
│   ├── features/              # 특징 추출 모듈
│   │   ├── __init__.py
│   │   ├── optical_flow.py    # Optical Flow
│   │   ├── motion_history.py  # Motion History
│   │   └── extractor.py       # 통합 추출기
│   │
│   ├── models/                # 딥러닝 모델
│   │   ├── __init__.py
│   │   ├── autoencoder.py     # Autoencoder 모델
│   │   ├── temporal.py        # 시계열 모델
│   │   └── ensemble.py        # 앙상블
│   │
│   ├── detection/             # 이상 검출 모듈
│   │   ├── __init__.py
│   │   ├── scorer.py          # 이상 점수 계산
│   │   ├── detector.py        # 이상 판정
│   │   └── calibrator.py      # 임계값 캘리브레이션
│   │
│   ├── alert/                 # 알림 모듈
│   │   ├── __init__.py
│   │   ├── manager.py         # 알림 관리
│   │   ├── logger.py          # 로깅
│   │   └── notifier.py        # 외부 알림
│   │
│   └── utils/                 # 유틸리티
│       ├── __init__.py
│       ├── config.py          # 설정 관리
│       ├── visualization.py   # 시각화
│       └── metrics.py         # 성능 지표
│
├── configs/                   # 설정 파일
│   ├── default.yaml           # 기본 설정
│   └── equipment/             # 설비별 설정
│
├── models/                    # 학습된 모델 저장
│
├── data/                      # 데이터 (gitignore)
│   ├── raw/                   # 원본 영상
│   ├── processed/             # 전처리 데이터
│   └── logs/                  # 로그
│
├── tests/                     # 테스트
│   ├── unit/
│   └── integration/
│
├── scripts/                   # 실행 스크립트
│   ├── train.py               # 학습 실행
│   ├── inference.py           # 추론 실행
│   └── evaluate.py            # 평가
│
├── requirements.txt           # 의존성
├── setup.py                   # 패키지 설정
└── README.md                  # 프로젝트 소개
```

---

## 8. 개발 단계 (Roadmap)

### Phase 1: 기반 구축 (Foundation)
- [ ] 프로젝트 구조 설정
- [ ] 영상 입력 모듈 개발 (파일, 카메라)
- [ ] 기본 전처리 파이프라인
- [ ] Optical Flow 기반 특징 추출
- [ ] 기본 시각화 도구

### Phase 2: 핵심 모델 (Core Model)
- [ ] Convolutional Autoencoder 구현
- [ ] 학습 파이프라인 구축
- [ ] 재구성 오차 기반 이상 검출
- [ ] 기본 임계값 설정

### Phase 3: 고도화 (Enhancement)
- [ ] LSTM/Transformer 시계열 모델 추가
- [ ] 앙상블 스코어링
- [ ] 동적 임계값 조정
- [ ] 조명 변화 대응 강화

### Phase 4: 운영화 (Production)
- [ ] 실시간 추론 최적화
- [ ] 알림 시스템 연동
- [ ] 모니터링 대시보드
- [ ] 설비별 모델 관리

### Phase 5: 확장 (Extension)
- [ ] 다중 카메라 지원
- [ ] 모델 자동 업데이트
- [ ] 클라우드 연동
- [ ] REST API 제공

---

## 9. 고려사항 및 리스크

### 9.1 기술적 고려사항

| 항목 | 고려사항 | 대응방안 |
|------|----------|----------|
| 조명 변화 | 그림자, 밝기 변화로 오탐 발생 가능 | CLAHE, 다양한 조건 학습, 적응형 임계값 |
| 정상 패턴 다양성 | 정상 동작도 변동이 있음 | 충분한 학습 데이터, VAE 활용 |
| 실시간 처리 | GPU 연산 병목 | 모델 경량화, TensorRT, 배치 처리 |
| 사이클 타임 변동 | 설비 속도 변동 | 동적 시간 정렬 (DTW), 정규화 |
| 오탐/미탐 균형 | 민감도와 특이도 트레이드오프 | 다단계 임계값, 확인 대기 시간 |

### 9.2 운영 리스크

| 리스크 | 영향 | 완화방안 |
|--------|------|----------|
| 과다 알림 (Alert Fatigue) | 운영자 무시 | 알림 그룹핑, 심각도 분류 |
| 모델 노후화 | 검출 성능 저하 | 주기적 재학습, 성능 모니터링 |
| 신규 이상 유형 | 미검출 | 지속적 피드백, 모델 업데이트 |

---

## 10. 성능 평가 지표

### 10.1 검출 성능

```
Precision = TP / (TP + FP)    # 이상 판정 중 실제 이상 비율
Recall    = TP / (TP + FN)    # 실제 이상 중 검출된 비율
F1-Score  = 2 * (P * R) / (P + R)

목표: Precision > 90%, Recall > 85%, F1 > 87%
```

### 10.2 실시간 성능

```
Latency   = 프레임 입력 ~ 판정 완료 시간
Throughput = 초당 처리 프레임 수

목표: Latency < 100ms, Throughput >= 30fps
```

### 10.3 시스템 안정성

```
Uptime    = 정상 운영 시간 / 전체 시간
MTBF      = 평균 고장 간격

목표: Uptime > 99.5%
```

---

## 11. 용어 정의

| 용어 | 정의 |
|------|------|
| 이상 (Anomaly) | 학습된 정상 패턴에서 벗어난 동작 |
| 정상 패턴 | 설비의 일반적인 반복 동작 |
| Optical Flow | 연속 프레임 간 픽셀의 이동 벡터 |
| Autoencoder | 입력을 압축 후 복원하는 신경망 |
| 재구성 오차 | 원본과 복원된 데이터의 차이 |
| ROI | Region of Interest, 관심 영역 |
| 임계값 | 이상 판정 기준값 |
| Latency | 처리 지연 시간 |

---

## 12. 참고 자료

### 12.1 관련 논문
- "Deep Learning for Anomaly Detection: A Survey" (Chalapathy & Chawla, 2019)
- "Video Anomaly Detection with Sparse Coding" (Cong et al., 2011)
- "Memorizing Normality to Detect Anomaly" (Park et al., 2020)

### 12.2 오픈소스 참조
- [PyTorch Video](https://github.com/facebookresearch/pytorchvideo)
- [Anomalib](https://github.com/openvinotoolkit/anomalib)

---

*문서 버전: 1.0*
*작성일: 2026-01-19*
*상태: 초안*
