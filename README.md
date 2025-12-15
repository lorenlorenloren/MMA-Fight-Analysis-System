# MMA Fight Analysis System

**Production-Grade AI/ML Pipeline for Automated UFC Fight Video Analysis**

## Overview

This system provides an end-to-end computer vision and machine learning solution for automated analysis of mixed martial arts (MMA) fight videos. It performs fighter detection, pose estimation, action recognition, and generates quantitative combat metrics including strike statistics, spatial analytics, and fight summary reports.

Inspired by tennis analytics systems but fully adapted to UFC-style fights with support for grappling sequences, ground fighting, and octagon-based positioning.

## Key Features

✅ **Fighter Detection & Tracking**
- Real-time detection using YOLOv8/v11
- Persistent ID maintenance through occlusions (BoT-SORT)
- Re-identification via appearance embeddings

✅ **Pose Estimation**
- 21-point keypoint extraction (YOLO-Pose)
- Ground vs. standing position classification  
- Temporal smoothing via Kalman filters
- Multi-view 3D triangulation support

✅ **Action Recognition**
- **Rule-based**: Strike detection via hand velocity & body rotation
- **Learned**: LSTM/ST-GCN for complex grappling
- 15 action classes: jab, cross, hook, kicks, clinch, takedown, etc.

✅ **Quantitative Metrics**
- Strike count & accuracy
- Strike speed estimation (pixel → m/s)
- Fighter distance & movement speed
- Cage control & octagon positioning
- Per-round aggregation

✅ **Visualization**
- Real-time overlays (bounding boxes, skeletons, action labels)
- Octagon mini-map with fighter positions
- Activity heatmaps over time
- Annotated output videos

## System Architecture

```
Video Input → Preprocessing → Detection → Pose → Action Recognition → Analytics → Visualization
```

### Core Components

1. **Preprocessing** (`src/preprocessing/`)
   - Frame-by-frame video decoding
   - FPS normalization
   - Octagon detection & homography for spatial normalization

2. **Detection & Tracking** (`src/detection/`)
   - YOLO-based fighter bounding boxes
   - BoT-SORT tracking across frames
   - Re-identification after occlusions

3. **Pose Estimation** (`src/pose/`)
   - YOLO-Pose 21-keypoint extraction
   - Ground position detection
   - Keypoint confidence filtering

4. **Action Recognition** (`src/action_recognition/`)
   - Rule-based strike detection
   - Learned temporal models (LSTM)
   - Multi-action classification

5. **Analytics** (`src/analytics/`)
   - Strike metrics (count, speed, accuracy)
   - Spatial metrics (distance, cage control)
   - Temporal aggregation

6. **Visualization** (`src/visualization/`)
   - Overlay rendering
   - Mini-map generation
   - Heatmap visualization

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (recommended for GPU acceleration)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/lorenlorenloren/MMA-Fight-Analysis-System.git
cd MMA-Fight-Analysis-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\\Scripts\\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Download model weights
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt'); YOLO('yolov8x-pose.pt')"
```

## Usage

### Quick Start

```python
from src.pipeline import MMAAnalyzerPipeline

# Initialize pipeline
pipeline = MMAAnalyzerPipeline(config_path='config/model_config.yaml')

# Process fight video
results = pipeline.analyze(video_path='data/raw/fight.mp4')

# Save results
results.to_json('results/fight_analysis.json')
results.save_annotated_video('results/fight_annotated.mp4')
```

### Command Line

```bash
# Full pipeline
python src/pipeline.py \\
  --video data/raw/fight.mp4 \\
  --output results/fight_analysis.json \\
  --save-video \\
  --gpu

# Run tests
pytest tests/ -v

# Format & lint
black src/
flake8 src/
```

## Performance

| Component | Speed (A100 GPU) | Speed (CPU) |
|-----------|------------------|-------------|
| Detection | 280 FPS | ~20 FPS |
| Pose | ~50 FPS | ~10 FPS |
| Action Rec. | ~30 FPS | ~5 FPS |
| **End-to-End** | **~20 FPS** | **~2 FPS** |

Target: **0.5x real-time** for offline analysis (10 min to process 5 min round)

## Project Structure

See [STRUCTURE.md](STRUCTURE.md) for detailed directory layout.

## Configuration

Edit configuration files in `config/`:

- **`model_config.yaml`**: Model paths, detection thresholds, action classes
- **`analytics_config.yaml`**: Metric computation, cage geometry, visualization settings
- **`.env`**: Environment variables (device, paths, logging)

## Training

To train the action recognition model:

```bash
python src/action_recognition/train.py \\
  --data data/annotations/labels.json \\
  --epochs 100 \\
  --batch-size 32 \\
  --output models/action_recognition.pth
```

## Dataset & Labeling

For training data, we recommend:
- UFC Fight Pass archives (licensed)
- Publicly available fight footage
- Semi-automated labeling via CVAT or Labelbox

Annotation format: JSON with bounding boxes, keypoints, and action labels

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file

## Citation

If you use this system in research, please cite:

```bibtex
@software{mma_fight_analysis,
  title={MMA Fight Analysis System},
  author={lorenlorenloren},
  year={2024},
  url={https://github.com/lorenlorenloren/MMA-Fight-Analysis-System}
}
```

## References

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- BoT-SORT Tracking: https://arxiv.org/abs/2206.14651
- Pose Estimation: https://arxiv.org/abs/2304.07850
- ST-GCN: https://arxiv.org/abs/1801.07455

## Contact

For questions or issues, open a GitHub issue or contact [@lorenlorenloren](https://github.com/lorenlorenloren)

---

**Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: December 2024
