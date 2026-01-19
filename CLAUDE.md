# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MovAnal is a facility anomaly detection system that analyzes video footage from assembly line equipment to detect abnormal operations in real-time using unsupervised learning. The project uses Convolutional Autoencoders and LSTM models to learn normal operation patterns and identify deviations.

**Current Status**: Design/planning phase - only documentation exists, no source code yet.

## Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.0+
- **Computer Vision**: OpenCV 4.8+
- **Data Processing**: NumPy, Pandas
- **Configuration**: Hydra / YAML
- **Optional**: ONNX Runtime, TensorRT, FastAPI, Streamlit/Gradio

## Planned Commands (to be implemented)

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Lint
pylint src/
black src/

# Training
python scripts/train.py --config configs/default.yaml --data data/raw/videos --epochs 100

# Inference
python scripts/inference.py --model models/autoencoder.pt --input video.mp4

# Phase 1 demo
python scripts/demo_phase1.py
```

## Architecture

The system follows a modular pipeline architecture:

```
Video Input → Preprocessing → Feature Extraction → Anomaly Detection → Alert
```

### Key Modules (under `src/`)

- **input/**: Video sources (file, camera, RTSP), frame buffering
- **preprocessing/**: Normalization (CLAHE), ROI management, data augmentation
- **features/**: Optical Flow (Farneback), Motion History extraction
- **models/**: Convolutional Autoencoder, LSTM/Transformer temporal models, ensemble scoring
- **detection/**: Reconstruction error scoring, threshold calibration, adaptive thresholds
- **alert/**: Event logging, external notifications (REST, MQTT)
- **inference/**: Optimized inference engine, parallel multi-stream processing

### Anomaly Detection Algorithm

Dual-model approach combining:
1. **Autoencoder**: High reconstruction error indicates anomaly
2. **LSTM/Transformer**: High prediction error for next frame indicates anomaly

```
score = w1 * reconstruction_error + w2 * prediction_error + w3 * speed_anomaly_score
if score > threshold and consecutive_anomaly_count >= min_frames: ANOMALY
```

## Development Phases

1. **Phase 1 (Foundation)**: Project setup, video input, preprocessing, Optical Flow, visualization
2. **Phase 2 (Core Model)**: Autoencoder implementation, training pipeline, basic anomaly detection
3. **Phase 3 (Enhancement)**: LSTM/Transformer models, ensemble scoring, dynamic thresholds
4. **Phase 4 (Production)**: Inference optimization, alerts, dashboard, model registry
5. **Phase 5 (Extension)**: Multi-camera, online learning, cloud integration, REST API

## Key Documentation

- [SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md): Complete system architecture, module specifications, algorithms
- [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md): Detailed task breakdown for each phase with code examples

## Performance Targets

- Latency: < 100ms per frame (30fps)
- Precision: > 90%, Recall: > 85%
- False Positive Rate: < 5%
- GPU Memory: < 4GB
