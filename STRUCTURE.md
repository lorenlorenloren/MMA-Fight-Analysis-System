# Project Structure

## MMA Fight Analysis System

```
mma-fight-analysis/
├── config/
│   ├── model_config.yaml          # YOLO, pose, action recognition models config
│   └── analytics_config.yaml      # Metrics, thresholds, video processing settings
├── src/
│   ├── __init__.py                # Package initialization
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── video_loader.py        # Frame extraction and preprocessing
│   │   └── cage_detector.py       # Octagon detection & homography
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── fighter_tracker.py     # YOLO + BoT-SORT tracking
│   │   └── reid_module.py         # Re-identification for fighters
│   ├── pose/
│   │   ├── __init__.py
│   │   ├── pose_estimator.py      # YOLO-Pose or MediaPipe wrapper
│   │   └── triangulation.py       # Multi-view 3D pose estimation
│   ├── action_recognition/
│   │   ├── __init__.py
│   │   ├── rule_based.py          # Strike/grappling heuristics
│   │   ├── learned_model.py       # LSTM/ST-GCN models
│   │   └── train.py               # Model training pipeline
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Strike count, speed, distance metrics
│   │   ├── aggregator.py          # Per-round/fighter summaries
│   │   └── cage_control.py        # Position analysis
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── overlay.py             # Real-time overlay rendering
│   │   ├── minimap.py             # Octagon mini-map visualization
│   │   └── heatmap.py             # Activity heatmaps
│   └── pipeline.py                # End-to-end orchestration
├── data/
│   ├── raw/                       # Input fight videos
│   ├── processed/                 # Preprocessed data
│   ├── annotations/               # Ground truth labels
│   └── cache/                     # Intermediate pickle/JSON results
├── models/
│   ├── yolov8x.pt                 # Detection model weights
│   ├── action_recognition.pth     # Action classification model
│   └── checkpoints/               # Training checkpoints
├── notebooks/
│   ├── eda.ipynb                  # Exploratory data analysis
│   ├── pose_validation.ipynb      # Pose estimation verification
│   ├── action_recognition_demo.ipynb
│   └── end_to_end_analysis.ipynb  # Complete pipeline demo
├── tests/
│   ├── test_detection.py
│   ├── test_pose.py
│   ├── test_metrics.py
│   └── test_visualization.py
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── .env.example                   # Environment variable template
├── README.md                      # Main documentation
├── STRUCTURE.md                   # This file
└── PROMPT.md                      # Original project prompt
```

## Key Components

### Preprocessing
- Frame-by-frame video decoding with FPS normalization
- Octagon edge detection and homography for coordinate normalization

### Detection & Tracking
- YOLO-based fighter detection
- BoT-SORT persistent tracking across occlusions
- Re-identification via embeddings and appearance matching

### Pose Estimation
- YOLO-Pose for 21-point keypoint extraction
- Ground vs standing position classification
- Temporal smoothing via Kalman filters

### Action Recognition
- Rule-based: Hand velocity, body rotation for strikes
- Learned: LSTM/ST-GCN for grappling sequences
- 15 action classes (strikes, grappling, defense, resets)

### Analytics
- Strike metrics: count, accuracy, speed (pixel→world conversion)
- Spatial: fighter distance, movement speed, cage control
- Temporal: per-round aggregation, activity intensity

### Visualization
- Real-time overlays: bounding boxes, skeletons, action labels
- Mini-map of octagon with fighter positions and trails
- Heatmaps of activity over time

## Dependencies

- **Detection**: PyTorch, Ultralytics YOLO
- **CV**: OpenCV, NumPy, SciPy
- **Pose**: YOLO-Pose or MediaPipe
- **ML**: PyTorch, Scikit-learn
- **Utils**: Loguru, PyYAML, Pandas
- **Dev**: Pytest, Black, Flake8, MyPy

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Configure models
cp .env.example .env
edit config/model_config.yaml

# Run end-to-end pipeline
python src/pipeline.py --video data/raw/fight.mp4 --output results/fight_analysis.json

# Run tests
pytest tests/
```

## Performance

- **Detection**: 280 FPS (YOLOv8x on A100)
- **Pose**: ~50 FPS (single-GPU)
- **Action Recognition**: ~30 FPS (temporal aggregation)
- **Target**: 0.5x real-time for offline analysis
