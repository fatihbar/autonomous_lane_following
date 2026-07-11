# Autonomous Lane Following System

A professional-grade autonomous lane following system for closed-track vehicles using deep learning and classical computer vision with fallback safety mechanisms.

## Features

### 🎯 Core Perception
- **YOLO-based Lane Segmentation**: Real-time lane detection using YOLOv8 segmentation
- **Hough Transform Fallback**: Classical fallback when YOLO confidence is low
- **Dual-Source Fusion**: Intelligent combination of YOLO and Hough detections
- **Virtual Lane Generation**: Safety fallback using historical lane data when detection fails

### 🤖 Planning & Control
- **Pure Pursuit Steering**: Smooth path tracking algorithm
- **Confidence-based Speed Control**: Adaptive speed based on detection confidence
- **Real-time Path Planning**: Continuous path generation from detected lane centerline

### 🛡️ Safety & Reliability
- **Multi-layer Safety Supervisor**: Monitors system health at highest priority
- **LiDAR Integration Ready**: Optional obstacle detection and emergency braking
- **Fallback Chain**: YOLO → Hough → Virtual Lane → Emergency Stop
- **Performance Monitoring**: Real-time FPS, latency, and confidence tracking
- **Dry-Run Mode**: Safe testing without actual vehicle control

### 📊 Diagnostics & Visualization
- **Real-time Lane Visualization**: Live overlay of detected lanes on video
- **AI Explanations**: System explains its decisions and lane loss causes
- **Status Dashboard**: FPS, latency, confidence metrics
- **Debug Mode**: Detailed logging and visualization of all detection steps

## System Architecture

```
Video Input
    ↓
┌─────────────────────┐
│ YOLO Lane Detector  │
└──────────┬──────────┘
           ↓
    ┌──────────────┐
    │ Confidence   │
    │ Check?       │
    └──┬───────┬───┘
       │ High  │ Low
       ↓       ↓
    YOLO  ┌─────────────────────┐
           │ Hough Fallback      │
           └──────────┬──────────┘
                      ↓
              ┌───────────────┐
              │ Confidence?   │
              └──┬────────┬───┘
                 │ OK     │ Low
                 ↓        ↓
              Hough  ┌──────────────────┐
                     │ Virtual Lane Gen │
                     └────────┬─────────┘
                              ↓
        ┌─────────────────────────────────────┐
        │    Sensor Fusion                    │
        │  (YOLO + Hough + Virtual + LiDAR)  │
        └────────────┬────────────────────────┘
                     ↓
        ┌─────────────────────────────────────┐
        │    Path Planner                     │
        │  (Lookahead Point Generation)       │
        └────────────┬────────────────────────┘
                     ↓
        ┌─────────────────────────────────────┐
        │    Vehicle Controller               │
        │  (Pure Pursuit + Speed Control)     │
        └────────────┬────────────────────────┘
                     ↓
        ┌─────────────────────────────────────┐
        │    Safety Supervisor                │
        │  (Override & Intervention)          │
        └────────────┬────────────────────────┘
                     ↓
            Control Command
         (Steering + Throttle/Brake)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 4GB+ RAM (8GB+ recommended for real-time performance)

### Setup

```bash
# Clone repository
git clone https://github.com/fatihbar/autonomous_lane_following.git
cd autonomous_lane_following

# Install dependencies
pip install -r requirements.txt

# Optional: Install ROS2 support
pip install -r requirements_ros2.txt

# Install package
pip install -e .
```

## Quick Start

### Test with Video File

```bash
# Run on video file
python scripts/run_standalone.py \
  --model models/lane_segmentation.pt \
  --video path/to/video.mp4 \
  --output output_videos/result.mp4

# Run on webcam
python scripts/run_standalone.py \
  --model models/lane_segmentation.pt \
  --video 0  # 0 = default camera
```

### Test Components

```bash
# Test lane detection on single image
python scripts/test_detectors.py \
  --model models/lane_segmentation.pt \
  --image path/to/image.jpg \
  --test yolo  # or 'hough' or 'all'
```

## Configuration

All components are configured via YAML files in `config/` directory:

- **perception.yaml**: YOLO model and lane detection parameters
- **hough_fallback.yaml**: Hough Transform settings
- **fusion.yaml**: Multi-sensor fusion weights and thresholds
- **controller.yaml**: Vehicle control parameters
- **safety.yaml**: Safety supervisor thresholds
- **visualization.yaml**: Visualization and debug settings
- **sensors.yaml**: ROS2 topic mapping and sensor configuration

### Key Parameters

```yaml
# Perception - YOLO confidence for primary lane detection
perception.model.confidence_threshold: 0.5

# Safety - Maximum time without lane detection before emergency stop
safety.thresholds.max_no_lane_time_s: 2.0

# Safety - Maximum duration of virtual lane fallback
safety.thresholds.max_virtual_lane_time_s: 3.0

# Controller - Maximum vehicle speed
controller.speed.max_mps: 2.0

# Controller - Reduce speed when confidence is low
controller.confidence_based_speed.low: 0.8
```

## Project Structure

```
autonomous_lane_following/
├── core/                      # Core algorithms
│   ├── data_types.py         # All message definitions
│   ├── yolo_lane_detector.py # YOLO segmentation
│   ├── hough_lane_detector.py # Hough Transform fallback
│   ├── lane_mask_processor.py # Extract geometry from mask
│   ├── lane_geometry.py       # Build lane geometry
│   ├── virtual_lane_generator.py # Virtual lane fallback
│   ├── lidar_obstacle_detector.py # LiDAR integration
│   ├── sensor_fusion.py       # Multi-sensor fusion
│   ├── path_planner.py        # Path planning
│   ├── controller.py          # Pure Pursuit steering
│   ├── safety_supervisor.py   # Safety monitoring
│   ├── ai_diagnostics.py      # System explanations
│   └── logger.py              # Logging utilities
│
├── adapters/                  # Input/output adapters
│   ├── video_source.py        # Video file/camera input
│   └── standalone_bus.py      # Non-ROS pub/sub
│
├── utils/                     # Utility functions
│   ├── bev_utils.py           # Bird's eye view transforms
│   ├── kalman_filter.py       # Smoothing filters
│   ├── low_pass_filter.py     # Signal filtering
│   ├── geometry_utils.py      # Geometry calculations
│   └── safety_utils.py        # Safety helpers
│
├── visualization/             # Visualization
│   └── lane_visualizer.py     # Lane and status overlay
│
├── scripts/                   # Executable scripts
│   ├── run_standalone.py      # Main application
│   └── test_detectors.py      # Component testing
│
├── config/                    # Configuration files (YAML)
│   ├── perception.yaml
│   ├── hough_fallback.yaml
│   ├── fusion.yaml
│   ├── controller.yaml
│   ├── safety.yaml
│   ├── visualization.yaml
│   └── sensors.yaml
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System design
│   ├── CALIBRATION.md         # Camera calibration guide
│   ├── DEPLOYMENT.md          # ROS2 deployment guide
│   └── TROUBLESHOOTING.md     # Common issues
│
├── tests/                     # Unit and integration tests
├── models/                    # YOLO model files (download separately)
├── logs/                      # Runtime logs (generated)
├── output_videos/             # Output video files (generated)
│
├── requirements.txt           # Core dependencies
├── requirements_ros2.txt      # ROS2 optional dependencies
├── setup.py                   # Package setup
├── pyproject.toml             # Build configuration
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## Performance

### Benchmarks (on RTX 3060)
- YOLO inference: ~20-30ms (FP16)
- Hough fallback: ~5-10ms
- Sensor fusion: ~2ms
- Control computation: ~1ms
- **Total latency: ~25-45ms**
- **Achievable FPS: 25-40 fps** at 640x480 resolution

### Memory Usage
- Peak VRAM: ~4GB (with YOLO model)
- Peak RAM: ~2GB

## Data Types

All system components communicate via dataclasses defined in `core/data_types.py`:

- `DetectionSource`: Enum for detection source (YOLO, HOUGH, VIRTUAL_LANE, LOST)
- `SystemMode`: Enum for operating mode (MANUAL, ASSISTED, AUTONOMOUS, DEGRADED, EMERGENCY_STOP)
- `LaneGeometry`: Complete lane with left/right boundaries and centerline
- `FusionResult`: Result of multi-sensor fusion with confidence
- `ControlCommand`: Steering angle and throttle/brake commands
- `SafetyStatus`: System health and safety state

See `core/data_types.py` for complete definitions.

## Safety Features

### Fallback Chain
1. **Primary**: YOLO segmentation (high confidence)
2. **Secondary**: Hough Transform (if YOLO fails)
3. **Tertiary**: Virtual lane from history (if both fail)
4. **Emergency**: Full stop (if all fail > 2 seconds)

### Safety Constraints
- Lane confidence threshold: 0.3
- Fusion confidence threshold: 0.25
- Maximum inference latency: 100ms
- Minimum FPS: 10
- Maximum time without lane: 2 seconds
- Maximum virtual lane duration: 3 seconds

### Intervention Points
- Emergency stop if LiDAR detects obstacle < 0.5m
- Speed reduction in degraded mode
- Full brake if confidence drops unexpectedly
- Watchdog timeout (configurable)

## ROS2 Integration

For ROS2 deployment, see `docs/DEPLOYMENT.md`:

```bash
# Build ROS2 package
colcon build --packages-select autonomous_lane_following

# Run ROS2 node
ros2 run autonomous_lane_following lane_follower_node
```

All topics are configured in `config/sensors.yaml`.

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_lane_detection.py -v
```

### Code Style

```bash
# Format code
black . --line-length 100

# Type checking
mypy core/ --ignore-missing-imports

# Linting
pylint core/
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

All components write detailed logs to `logs/` directory.

## Common Issues

See `docs/TROUBLESHOOTING.md` for:
- GPU not detected
- Low FPS performance
- Lane detection failures
- Control command issues

## Model Acquisition

### Getting the YOLO Model

The repository requires a pre-trained YOLO lane segmentation model. Options:

1. **Train your own**:
   ```bash
   yolo segment train data=path/to/dataset.yaml model=yolov8s-seg.pt epochs=100
   ```

2. **Use provided weights** (download separately):
   - Place in `models/lane_segmentation.pt`

3. **Download example model**:
   ```bash
   mkdir -p models
   # Download your trained model here
   ```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this system in research, please cite:

```bibtex
@software{autonomous_lane_following_2024,
  title = {Autonomous Lane Following System},
  author = {Autonomous Systems Team},
  year = {2024},
  url = {https://github.com/fatihbar/autonomous_lane_following}
}
```

## Support

For issues and questions:
1. Check `docs/TROUBLESHOOTING.md`
2. Search existing GitHub issues
3. Open a new GitHub issue with:
   - System info (GPU, OS, Python version)
   - Error message and logs
   - Minimal reproducible example
   - Input video/image if relevant

## Roadmap

- [ ] Full ROS2 integration
- [ ] Multi-lane detection
- [ ] LiDAR-primary mode
- [ ] Weather robustness (rain, snow, fog)
- [ ] Night mode support
- [ ] Real vehicle deployment
- [ ] TensorRT optimization
- [ ] Quantization support (INT8)

## Acknowledgments

- Built with [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics)
- Computer Vision using [OpenCV](https://opencv.org/)
- Inspired by autonomous vehicle research community

---

**Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: 2024
