# MovAnal 구현 작업 계획서

## 개요

본 문서는 SYSTEM_DESIGN.md의 기획안을 기반으로 각 Phase별 구체적인 구현 작업을 정의합니다.

---

## Phase 1: 기반 구축 (Foundation)

### 목표
프로젝트의 기본 구조를 설정하고, 영상 입력 및 전처리 파이프라인을 구축합니다.

### 1.1 프로젝트 환경 설정

#### Task 1.1.1: 프로젝트 구조 생성
```
작업 내용:
- 디렉토리 구조 생성 (src/, configs/, tests/, scripts/, data/, models/)
- 각 모듈별 __init__.py 파일 생성
- .gitignore 설정 (data/, models/, __pycache__, .env 등)

산출물:
- 전체 디렉토리 구조
- .gitignore 파일
```

#### Task 1.1.2: 의존성 관리 설정
```
작업 내용:
- requirements.txt 작성
  - opencv-python>=4.8.0
  - torch>=2.0.0
  - torchvision>=0.15.0
  - numpy>=1.24.0
  - pandas>=2.0.0
  - pyyaml>=6.0
  - matplotlib>=3.7.0
  - tqdm>=4.65.0
- setup.py 작성 (패키지 설치용)
- pyproject.toml 작성 (선택)

산출물:
- requirements.txt
- setup.py
```

#### Task 1.1.3: 설정 관리 시스템 구축
```
작업 내용:
- configs/default.yaml 작성
  - video: 해상도, fps, 버퍼 크기
  - preprocessing: 정규화 파라미터
  - model: 기본 하이퍼파라미터
  - detection: 임계값, 연속 프레임 수
  - logging: 레벨, 출력 경로
- src/utils/config.py 구현
  - YAML 로드/저장 함수
  - 설정 병합 함수 (default + custom)
  - 설정 검증 함수

산출물:
- configs/default.yaml
- src/utils/config.py
```

### 1.2 영상 입력 모듈 개발

#### Task 1.2.1: VideoSource 추상 클래스 정의
```python
# src/input/video_source.py
작업 내용:
- VideoSource ABC(Abstract Base Class) 정의
  - open(): 소스 열기
  - read(): 프레임 읽기
  - close(): 소스 닫기
  - get_info(): 메타데이터 반환 (fps, resolution, frame_count)
  - __iter__, __next__: 이터레이터 지원

산출물:
- src/input/video_source.py
```

#### Task 1.2.2: 파일 입력 구현
```python
# src/input/file_reader.py
작업 내용:
- FileVideoSource 클래스 구현
  - cv2.VideoCapture 래핑
  - 지원 포맷: MP4, AVI, MKV
  - 프레임 번호 탐색 기능 (seek)
  - 구간 반복 재생 기능

테스트:
- 파일 열기/닫기
- 프레임 순차 읽기
- 특정 프레임 탐색

산출물:
- src/input/file_reader.py
- tests/unit/test_file_reader.py
```

#### Task 1.2.3: 카메라 입력 구현
```python
# src/input/camera.py
작업 내용:
- CameraSource 클래스 구현
  - USB 카메라 지원 (device index)
  - RTSP 스트림 지원 (URL)
  - 연결 재시도 로직
  - 버퍼링 관리 (최신 프레임 유지)

테스트:
- USB 카메라 연결
- RTSP 스트림 연결
- 연결 끊김 처리

산출물:
- src/input/camera.py
- tests/unit/test_camera.py
```

#### Task 1.2.4: 프레임 버퍼 구현
```python
# src/input/frame_buffer.py
작업 내용:
- FrameBuffer 클래스 구현
  - collections.deque 기반 순환 버퍼
  - 최대 N 프레임 유지
  - 스레드 안전 (threading.Lock)
  - 타임스탬프 관리

산출물:
- src/input/frame_buffer.py
```

### 1.3 전처리 파이프라인 개발

#### Task 1.3.1: 기본 정규화 구현
```python
# src/preprocessing/normalizer.py
작업 내용:
- FrameNormalizer 클래스 구현
  - 해상도 조정 (resize)
  - 픽셀값 정규화 (0-1 또는 -1~1)
  - 그레이스케일 변환 옵션
  - CLAHE 적용 (조명 정규화)

함수:
- normalize(frame) -> normalized_frame
- denormalize(normalized) -> frame
- apply_clahe(frame) -> enhanced_frame

산출물:
- src/preprocessing/normalizer.py
- tests/unit/test_normalizer.py
```

#### Task 1.3.2: ROI 관리자 구현
```python
# src/preprocessing/roi_manager.py
작업 내용:
- ROIManager 클래스 구현
  - ROI 정의 (x, y, width, height)
  - 다중 ROI 지원
  - ROI 영역 추출
  - ROI 설정 저장/로드 (JSON/YAML)
  - 시각화 (ROI 박스 그리기)

산출물:
- src/preprocessing/roi_manager.py
```

#### Task 1.3.3: 전처리 파이프라인 통합
```python
# src/preprocessing/pipeline.py
작업 내용:
- PreprocessingPipeline 클래스 구현
  - 전처리 단계 체이닝
  - 설정 기반 파이프라인 구성
  - 배치 처리 지원

사용 예:
pipeline = PreprocessingPipeline([
    Resize(640, 480),
    CLAHE(),
    Normalize(),
    ROICrop(roi)
])
processed = pipeline(frame)

산출물:
- src/preprocessing/pipeline.py
```

### 1.4 특징 추출 모듈 개발

#### Task 1.4.1: Optical Flow 추출기 구현
```python
# src/features/optical_flow.py
작업 내용:
- OpticalFlowExtractor 클래스 구현
  - Farneback 알고리즘 (cv2.calcOpticalFlowFarneback)
  - Dense optical flow 계산
  - Flow magnitude 및 angle 계산
  - Flow 시각화 (HSV 컬러맵)

출력:
- flow_magnitude: 속도 크기 맵
- flow_angle: 방향 맵
- flow_rgb: 시각화 이미지

산출물:
- src/features/optical_flow.py
- tests/unit/test_optical_flow.py
```

#### Task 1.4.2: Motion History 추출기 구현
```python
# src/features/motion_history.py
작업 내용:
- MotionHistoryExtractor 클래스 구현
  - 프레임 차분 계산
  - 누적 모션 히스토리 이미지 (MHI)
  - 모션 에너지 이미지 (MEI)
  - 시간 감쇠 파라미터 조정

산출물:
- src/features/motion_history.py
```

#### Task 1.4.3: 특징 추출기 통합
```python
# src/features/extractor.py
작업 내용:
- FeatureExtractor 클래스 구현
  - 여러 특징 추출기 조합
  - 특징 벡터 연결 (concatenation)
  - 특징 정규화

산출물:
- src/features/extractor.py
```

### 1.5 기본 시각화 도구 개발

#### Task 1.5.1: 실시간 시각화 구현
```python
# src/utils/visualization.py
작업 내용:
- RealtimeVisualizer 클래스 구현
  - 원본 프레임 표시
  - Optical Flow 오버레이
  - ROI 표시
  - FPS 표시
  - cv2.imshow 기반

산출물:
- src/utils/visualization.py
```

### 1.6 Phase 1 통합 테스트

#### Task 1.6.1: 데모 스크립트 작성
```python
# scripts/demo_phase1.py
작업 내용:
- 영상 파일 입력 → 전처리 → 특징 추출 → 시각화
- 전체 파이프라인 동작 확인
- 처리 속도 측정

산출물:
- scripts/demo_phase1.py
```

---

## Phase 2: 핵심 모델 (Core Model)

### 목표
Convolutional Autoencoder를 구현하고 정상 패턴 학습 및 이상 검출 기능을 구축합니다.

### 2.1 Autoencoder 모델 구현

#### Task 2.1.1: Encoder 네트워크 설계
```python
# src/models/autoencoder.py
작업 내용:
- ConvEncoder 클래스 구현
  - 입력: (B, C, H, W) 프레임 시퀀스
  - Conv2d + BatchNorm + ReLU 블록
  - 다운샘플링 (stride=2 또는 MaxPool)
  - 잠재 벡터 출력

구조 예시:
Input (3, 256, 256)
→ Conv(64) → Conv(128) → Conv(256) → Conv(512)
→ Flatten → FC → Latent (512)

산출물:
- src/models/autoencoder.py (Encoder 부분)
```

#### Task 2.1.2: Decoder 네트워크 설계
```python
# src/models/autoencoder.py
작업 내용:
- ConvDecoder 클래스 구현
  - 입력: 잠재 벡터
  - ConvTranspose2d로 업샘플링
  - 원본 크기로 복원
  - Sigmoid 출력 (0-1 범위)

구조 예시:
Latent (512)
→ FC → Reshape
→ ConvT(512) → ConvT(256) → ConvT(128) → ConvT(64)
→ Conv(3) → Output (3, 256, 256)

산출물:
- src/models/autoencoder.py (Decoder 부분)
```

#### Task 2.1.3: Autoencoder 통합 클래스
```python
# src/models/autoencoder.py
작업 내용:
- ConvAutoencoder 클래스 구현
  - Encoder + Decoder 조합
  - forward() 메서드
  - encode() 메서드 (잠재 벡터만 반환)
  - decode() 메서드 (복원만 수행)
  - 모델 저장/로드 메서드

산출물:
- src/models/autoencoder.py (완성)
- tests/unit/test_autoencoder.py
```

### 2.2 학습 파이프라인 구축

#### Task 2.2.1: 데이터셋 클래스 구현
```python
# src/data/dataset.py
작업 내용:
- VideoFrameDataset 클래스 구현
  - torch.utils.data.Dataset 상속
  - 영상 파일에서 프레임 추출
  - 전처리 파이프라인 적용
  - __getitem__: 단일 프레임 또는 프레임 시퀀스

- SequenceDataset 클래스 구현
  - 연속 N 프레임을 하나의 샘플로
  - 슬라이딩 윈도우 방식

산출물:
- src/data/dataset.py
```

#### Task 2.2.2: 학습 루프 구현
```python
# src/training/trainer.py
작업 내용:
- AutoencoderTrainer 클래스 구현
  - 모델, 옵티마이저, 손실함수 관리
  - train_epoch() 메서드
  - validate() 메서드
  - 체크포인트 저장/로드
  - 학습 로그 (loss, epoch)
  - TensorBoard 연동 (선택)

손실 함수:
- MSE Loss (기본)
- SSIM Loss (구조적 유사도)
- Perceptual Loss (선택)

산출물:
- src/training/trainer.py
```

#### Task 2.2.3: 학습 스크립트 작성
```python
# scripts/train.py
작업 내용:
- 명령줄 인자 처리 (argparse)
  - --config: 설정 파일 경로
  - --data: 학습 데이터 경로
  - --epochs: 학습 에폭 수
  - --output: 모델 저장 경로
- 데이터 로더 생성
- 모델 초기화
- 학습 실행
- 최종 모델 저장

사용 예:
python scripts/train.py --config configs/default.yaml --data data/raw/normal_videos --epochs 100

산출물:
- scripts/train.py
```

### 2.3 이상 검출 기능 구현

#### Task 2.3.1: 재구성 오차 계산기 구현
```python
# src/detection/scorer.py
작업 내용:
- ReconstructionScorer 클래스 구현
  - 입력 프레임과 재구성 프레임 비교
  - 픽셀별 오차 계산 (MSE, MAE)
  - 영역별 오차 집계
  - 이상 점수 정규화

출력:
- anomaly_score: 전체 이상 점수
- error_map: 픽셀별 오차 맵 (시각화용)

산출물:
- src/detection/scorer.py
```

#### Task 2.3.2: 이상 탐지기 구현
```python
# src/detection/detector.py
작업 내용:
- AnomalyDetector 클래스 구현
  - 모델 로드
  - 실시간 추론
  - 점수 기반 판정
  - 연속성 검사 (N 프레임 연속 이상 시 알림)

메서드:
- detect(frame) -> (is_anomaly, score, error_map)
- detect_batch(frames) -> List[(is_anomaly, score)]

산출물:
- src/detection/detector.py
```

#### Task 2.3.3: 임계값 설정 도구 구현
```python
# src/detection/calibrator.py
작업 내용:
- ThresholdCalibrator 클래스 구현
  - 검증 데이터로 점수 분포 분석
  - 백분위 기반 임계값 설정
  - ROC 곡선 분석 (이상 샘플 있을 경우)
  - 임계값 저장/로드

산출물:
- src/detection/calibrator.py
```

### 2.4 Phase 2 통합 테스트

#### Task 2.4.1: 추론 스크립트 작성
```python
# scripts/inference.py
작업 내용:
- 학습된 모델 로드
- 영상 입력 (파일 또는 카메라)
- 실시간 이상 검출
- 결과 시각화 (원본 + 재구성 + 오차맵)
- 이상 발생 시 콘솔 출력

산출물:
- scripts/inference.py
```

#### Task 2.4.2: 평가 스크립트 작성
```python
# scripts/evaluate.py
작업 내용:
- 테스트 데이터셋 평가
- 재구성 오차 분포 분석
- 시각화 (히스토그램, 시계열)
- 성능 지표 출력

산출물:
- scripts/evaluate.py
```

---

## Phase 3: 고도화 (Enhancement)

### 목표
시계열 모델 추가, 앙상블 스코어링, 동적 임계값 조정으로 검출 성능을 향상합니다.

### 3.1 시계열 모델 추가

#### Task 3.1.1: LSTM 기반 예측 모델 구현
```python
# src/models/temporal.py
작업 내용:
- TemporalLSTM 클래스 구현
  - 입력: 연속 프레임의 특징 벡터 시퀀스
  - LSTM 레이어 (다층 가능)
  - 다음 프레임 특징 예측
  - 예측 오차 기반 이상 점수

구조:
Input (seq_len, feature_dim)
→ LSTM(hidden_size) × num_layers
→ FC → Output (feature_dim)

산출물:
- src/models/temporal.py
```

#### Task 3.1.2: Transformer 기반 예측 모델 구현 (선택)
```python
# src/models/temporal.py
작업 내용:
- TemporalTransformer 클래스 구현
  - 입력: 프레임 시퀀스
  - Positional Encoding
  - Multi-head Self-Attention
  - 다음 프레임 예측

산출물:
- src/models/temporal.py (Transformer 추가)
```

#### Task 3.1.3: 시계열 모델 학습 파이프라인
```python
# src/training/temporal_trainer.py
작업 내용:
- TemporalTrainer 클래스 구현
  - 시퀀스 데이터 학습
  - 예측 손실 함수
  - 학습/검증 루프

산출물:
- src/training/temporal_trainer.py
```

### 3.2 앙상블 스코어링

#### Task 3.2.1: 앙상블 스코어러 구현
```python
# src/models/ensemble.py
작업 내용:
- EnsembleScorer 클래스 구현
  - 다중 모델 관리
  - 가중 평균 스코어링
  - 투표 기반 판정 (선택)

스코어 계산:
score = w1 * ae_score + w2 * lstm_score + w3 * flow_score

산출물:
- src/models/ensemble.py
```

#### Task 3.2.2: 가중치 최적화
```python
# src/detection/weight_optimizer.py
작업 내용:
- 검증 데이터로 최적 가중치 탐색
- 그리드 서치 또는 베이지안 최적화
- 최적 가중치 저장

산출물:
- src/detection/weight_optimizer.py
```

### 3.3 동적 임계값 조정

#### Task 3.3.1: 적응형 임계값 구현
```python
# src/detection/adaptive_threshold.py
작업 내용:
- AdaptiveThreshold 클래스 구현
  - 이동 평균 기반 임계값
  - 시간대별 임계값 조정
  - 이상치 제외 평균

수식:
threshold = mean(recent_scores) + k * std(recent_scores)

산출물:
- src/detection/adaptive_threshold.py
```

### 3.4 조명 변화 대응 강화

#### Task 3.4.1: 조명 정규화 고도화
```python
# src/preprocessing/lighting.py
작업 내용:
- LightingNormalizer 클래스 구현
  - 다중 CLAHE 파라미터 지원
  - 히스토그램 매칭
  - 그림자 검출 및 보정

산출물:
- src/preprocessing/lighting.py
```

#### Task 3.4.2: 데이터 증강 구현
```python
# src/preprocessing/augmentation.py
작업 내용:
- VideoAugmentor 클래스 구현
  - 밝기 변화
  - 대비 변화
  - 노이즈 추가
  - 그림자 시뮬레이션

산출물:
- src/preprocessing/augmentation.py
```

### 3.5 Phase 3 통합 테스트

#### Task 3.5.1: 성능 비교 평가
```
작업 내용:
- 단일 모델 vs 앙상블 성능 비교
- 고정 임계값 vs 동적 임계값 비교
- 조명 정규화 효과 분석

산출물:
- 성능 비교 리포트
```

---

## Phase 4: 운영화 (Production)

### 목표
실시간 추론 최적화, 알림 시스템 연동, 모니터링 대시보드를 구축합니다.

### 4.1 실시간 추론 최적화

#### Task 4.1.1: 모델 최적화
```python
# src/optimization/model_optimizer.py
작업 내용:
- ONNX 변환
- TensorRT 최적화 (NVIDIA GPU)
- 양자화 (INT8/FP16)
- 추론 속도 벤치마크

산출물:
- src/optimization/model_optimizer.py
- 최적화된 모델 파일
```

#### Task 4.1.2: 추론 엔진 구현
```python
# src/inference/engine.py
작업 내용:
- InferenceEngine 클래스 구현
  - 최적화된 모델 로드
  - 배치 추론 지원
  - GPU 메모리 관리
  - 비동기 추론 (선택)

산출물:
- src/inference/engine.py
```

### 4.2 알림 시스템 연동

#### Task 4.2.1: 알림 관리자 구현
```python
# src/alert/manager.py
작업 내용:
- AlertManager 클래스 구현
  - 이상 이벤트 수신
  - 알림 중복 방지 (debouncing)
  - 심각도 분류 (warning, critical)
  - 알림 큐 관리

산출물:
- src/alert/manager.py
```

#### Task 4.2.2: 로깅 시스템 구현
```python
# src/alert/logger.py
작업 내용:
- AnomalyLogger 클래스 구현
  - 이상 이벤트 로깅
  - 로그 포맷 (JSON)
  - 로그 로테이션
  - 이상 구간 영상 저장

로그 내용:
- 타임스탬프
- 이상 점수
- 프레임 번호
- 영상 클립 경로

산출물:
- src/alert/logger.py
```

#### Task 4.2.3: 외부 알림 연동
```python
# src/alert/notifier.py
작업 내용:
- Notifier 추상 클래스
- WebhookNotifier (REST API 호출)
- MQTTNotifier (MQTT 발행)
- EmailNotifier (선택)

산출물:
- src/alert/notifier.py
```

### 4.3 모니터링 대시보드

#### Task 4.3.1: REST API 서버 구현
```python
# src/api/server.py
작업 내용:
- FastAPI 기반 서버
- 엔드포인트:
  - GET /status: 시스템 상태
  - GET /metrics: 실시간 지표
  - GET /events: 최근 이상 이벤트
  - POST /config: 설정 변경
  - GET /stream: 영상 스트림 (MJPEG)

산출물:
- src/api/server.py
- src/api/routes.py
```

#### Task 4.3.2: 대시보드 UI 구현
```python
# src/dashboard/app.py
작업 내용:
- Streamlit 또는 Gradio 기반 UI
- 실시간 영상 표시
- 이상 점수 그래프
- 이벤트 히스토리
- 설정 조정 패널

산출물:
- src/dashboard/app.py
```

### 4.4 설비별 모델 관리

#### Task 4.4.1: 모델 레지스트리 구현
```python
# src/models/registry.py
작업 내용:
- ModelRegistry 클래스 구현
  - 설비별 모델 저장/로드
  - 모델 버전 관리
  - 모델 메타데이터 관리

산출물:
- src/models/registry.py
```

### 4.5 Phase 4 통합 테스트

#### Task 4.5.1: 통합 운영 테스트
```
작업 내용:
- 24시간 연속 운영 테스트
- 메모리 누수 검사
- 알림 정상 동작 확인
- 대시보드 기능 검증

산출물:
- 운영 테스트 리포트
```

---

## Phase 5: 확장 (Extension)

### 목표
다중 카메라 지원, 모델 자동 업데이트, 클라우드 연동 기능을 추가합니다.

### 5.1 다중 카메라 지원

#### Task 5.1.1: 멀티 스트림 매니저 구현
```python
# src/input/multi_stream.py
작업 내용:
- MultiStreamManager 클래스 구현
  - 다중 카메라 동시 처리
  - 스트림별 독립 파이프라인
  - 자원 할당 관리

산출물:
- src/input/multi_stream.py
```

#### Task 5.1.2: 병렬 추론 구현
```python
# src/inference/parallel.py
작업 내용:
- 멀티 스레드/프로세스 추론
- GPU 자원 분배
- 결과 통합

산출물:
- src/inference/parallel.py
```

### 5.2 모델 자동 업데이트

#### Task 5.2.1: 온라인 학습 구현
```python
# src/training/online_learner.py
작업 내용:
- OnlineLearner 클래스 구현
  - 새로운 정상 데이터 수집
  - 점진적 학습 (Incremental Learning)
  - 모델 성능 모니터링
  - 자동 재학습 트리거

산출물:
- src/training/online_learner.py
```

### 5.3 클라우드 연동

#### Task 5.3.1: 클라우드 스토리지 연동
```python
# src/cloud/storage.py
작업 내용:
- 이상 영상 클라우드 업로드
- 모델 파일 동기화
- AWS S3 / Azure Blob / GCP Storage 지원

산출물:
- src/cloud/storage.py
```

### 5.4 REST API 확장

#### Task 5.4.1: 완전한 API 구현
```python
# src/api/
작업 내용:
- 모델 관리 API
- 학습 트리거 API
- 다중 카메라 관리 API
- API 문서화 (OpenAPI/Swagger)

산출물:
- src/api/ 확장
- docs/API.md
```

---

## 부록: 작업 우선순위 매트릭스

### 필수 (Must Have)
| Phase | Task | 설명 |
|-------|------|------|
| 1 | 1.1.1 ~ 1.2.4 | 프로젝트 기반, 영상 입력 |
| 1 | 1.3.1 ~ 1.3.3 | 전처리 파이프라인 |
| 1 | 1.4.1 | Optical Flow |
| 2 | 2.1.1 ~ 2.1.3 | Autoencoder 모델 |
| 2 | 2.2.1 ~ 2.2.3 | 학습 파이프라인 |
| 2 | 2.3.1 ~ 2.3.3 | 이상 검출 |

### 권장 (Should Have)
| Phase | Task | 설명 |
|-------|------|------|
| 1 | 1.4.2 ~ 1.4.3 | Motion History, 특징 통합 |
| 3 | 3.1.1 | LSTM 시계열 모델 |
| 3 | 3.2.1 | 앙상블 스코어링 |
| 4 | 4.2.1 ~ 4.2.3 | 알림 시스템 |

### 선택 (Could Have)
| Phase | Task | 설명 |
|-------|------|------|
| 3 | 3.1.2 | Transformer 모델 |
| 4 | 4.1.1 | 모델 최적화 (TensorRT) |
| 4 | 4.3.1 ~ 4.3.2 | 대시보드 |
| 5 | 전체 | 확장 기능 |

---

## 일정 추정 (참고용)

| Phase | 예상 작업량 |
|-------|------------|
| Phase 1 | 기반 구축 - 중간 규모 |
| Phase 2 | 핵심 모델 - 큰 규모 |
| Phase 3 | 고도화 - 중간 규모 |
| Phase 4 | 운영화 - 큰 규모 |
| Phase 5 | 확장 - 중간 규모 |

---

*문서 버전: 1.0*
*작성일: 2026-01-19*
*기반 문서: SYSTEM_DESIGN.md v1.0*
