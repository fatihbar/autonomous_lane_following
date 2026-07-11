# Architecture Overview

## System Design Philosophy

This autonomous lane following system is designed with **safety-first** and **reliability** as core principles:

1. **Layered Perception**: Multiple detection methods with intelligent fallback
2. **Safety Supervision**: Highest-priority safety module overrides all decisions
3. **Explainability**: AI diagnostics explain system decisions and failures
4. **Graceful Degradation**: Reduces capability rather than failing catastrophically

## Component Hierarchy

### Layer 1: Perception (Detection)

#### YOLO Lane Detector (`core/yolo_lane_detector.py`)
- **Purpose**: Primary lane detection using deep learning
- **Input**: RGB image (any resolution)
- **Output**: Lane segmentation mask with confidence
- **Properties**:
  - Runs on GPU (CUDA/CPU fallback)
  - Fast inference (20-30ms @ 640x480)
  - Confidence-based
  - Memory efficient with FP16 support

#### Hough Lane Detector (`core/hough_lane_detector.py`)
- **Purpose**: Fallback when YOLO confidence is low
- **Input**: RGB image
- **Output**: Left and right lane point lists
- **Properties**:
  - CPU-only (very fast ~5-10ms)
  - Robust to different lighting
  - No model weights needed
  - Works in degraded visibility

#### Lane Mask Processor (`core/lane_mask_processor.py`)
- **Purpose**: Convert YOLO mask to lane geometry
- **Operations**:
  - Morphological cleaning (open/close)
  - Contour extraction
  - Polynomial fitting (quadratic)
  - Boundary separation (left/right)
  - Centerline generation

#### Lane Geometry Builder (`core/lane_geometry.py`)
- **Purpose**: Create lane geometry from Hough or other sources
- **Operations**:
  - Point-to-boundary conversion
  - Polynomial fitting
  - Centerline calculation
  - Lane width estimation

### Layer 2: Fallback Mechanisms

#### Virtual Lane Generator (`core/virtual_lane_generator.py`)
- **Purpose**: Generate lane from history when real detection fails
- **Mechanism**:
  - Maintains 30-frame history
  - Uses most recent valid detection
  - Confidence degradation (~50%)
  - Maximum 3-second duration
  - Triggers safety intervention if exceeded

#### LiDAR Obstacle Detector (`core/lidar_obstacle_detector.py`)
- **Purpose**: Optional obstacle detection for emergency braking
- **Implementation**:
  - ROI filtering (forward 5m, width 3m)
  - Height-based filtering
  - Distance measurement
  - Emergency stop trigger (< 0.5m)

### Layer 3: Sensor Fusion (`core/sensor_fusion.py`)

**Priority-based fusion algorithm**:

```
IF yolo_confidence >= 0.5:
    source = YOLO
    IF hough_confidence >= 0.3:
        source = YOLO_HOUGH (blended)
ELSIF hough_confidence >= 0.3:
    source = HOUGH
ELSIF virtual_lane_available:
    source = VIRTUAL_LANE
ELSE:
    source = LOST
    confidence = 0.0

IF lidar_clear AND distance_safe:
    confidence_boost = +0.2
```

### Layer 4: Planning

#### Path Planner (`core/path_planner.py`)
- **Purpose**: Generate smooth paths for control
- **Inputs**: Lane geometry, image height
- **Outputs**: 
  - Centerline point list
  - Lookahead point (1m ahead)
  - Curvature estimate
- **Algorithm**:
  - Extract centerline points
  - Polynomial interpolation
  - Lookahead point calculation
  - Curvature estimation (2nd derivative)

### Layer 5: Control

#### Vehicle Controller (`core/controller.py`)
- **Purpose**: Generate steering and speed commands
- **Algorithm**: Pure Pursuit steering
- **Components**:
  - Lateral control: Pure Pursuit → steering angle
  - Longitudinal control: Confidence-based speed
  - Smoothing: Exponential smoothing (α=0.3)
- **Safety limits**:
  - Max steering: ±35°
  - Max speed: 2.0 m/s
  - Speed reduction in degraded mode

### Layer 6: Safety Supervision

#### Safety Supervisor (`core/safety_supervisor.py`)
**Highest priority - can override all control decisions**

**Monitored metrics**:
- Lane confidence (threshold: 0.3)
- Fusion confidence (threshold: 0.25)
- System FPS (minimum: 10)
- Inference latency (maximum: 100ms)
- Time without lane (maximum: 2s)
- Virtual lane duration (maximum: 3s)

**Actions**:
1. **Mode switching**: AUTONOMOUS → DEGRADED → EMERGENCY_STOP
2. **Command modification**: Reduce speed, apply brakes
3. **Intervention**: Force stop if constraints violated
4. **Logging**: Record all safety events

### Layer 7: Diagnostics

#### AI Diagnostics (`core/ai_diagnostics.py`)
- **Purpose**: Explain system decisions to users
- **Capabilities**:
  - Lane loss reason explanation
  - Current source identification
  - System status reporting
  - Debug information generation

## Data Flow

```
Frame Input (BGR image)
    ↓
[YOLO Detector] ────→ Mask + Confidence
    ↓
[Mask Processor] ────→ YOLO Geometry
    ↓
[Hough Detector] ────→ Hough Geometry (parallel)
    ↓
[Geometry Builder] ────→ Hough Geometry formatted
    ↓
[Virtual Lane Gen] ────→ Virtual Geometry (if needed)
    ↓
[Sensor Fusion] ────→ Final Lane Geometry + Source
    ↓
[Path Planner] ────→ Path with Lookahead
    ↓
[Controller] ────→ Steering + Speed Command
    ↓
[Safety Supervisor] ────→ VALIDATED Command
    ↓
Vehicle Control Output
```

## Configuration System

All components are configured via `config/*.yaml` files:

- **Decoupled design**: Change behavior without code changes
- **Easy experimentation**: Tune thresholds and parameters
- **Multiple profiles**: Different configs for different scenarios

Example:
```yaml
# config/perception.yaml
model:
  confidence_threshold: 0.5  # When to trust YOLO
  imgsz: 640                 # Input size

roi:
  top_ratio: 0.0             # Search from top
  bottom_ratio: 1.0          # To bottom
```

## Error Handling Strategy

**Principle**: Fail gracefully with fallback

```python
try:
    detection = yolo_detector.detect(image)
    if detection.confidence < threshold:
        detection = hough_detector.detect(image)
    if no_detection:
        detection = virtual_lane_generator.generate()
    if still_no_detection:
        safety_supervisor.emergency_stop()
except Exception as e:
    logger.error(f"Detection failed: {e}")
    safety_supervisor.emergency_stop()
```

## Performance Optimization

### Latency Budget (Target: 33ms for 30fps)
- YOLO inference: 20-30ms
- Mask processing: 2-3ms
- Fusion: 1-2ms
- Planning: 1ms
- Control: 1ms
- **Total**: ~25-37ms ✓

### Memory Optimization
- YOLO model: FP16 precision (50% smaller)
- Input resizing: 640x480 instead of full resolution
- History pruning: Keep only last 30 frames
- Circular buffers: Constant memory usage

### Parallelization
- YOLO inference: GPU
- Hough processing: CPU (no blocking)
- Visualization: Separate thread (optional)

## Testing Strategy

### Unit Tests
```python
# Test individual components
test_yolo_detector(model_path, image)
test_hough_detector(image)
test_sensor_fusion(yolo_geom, hough_geom)
test_pure_pursuit_controller(path)
test_safety_supervisor(metrics)
```

### Integration Tests
```python
# Test full pipeline
test_full_system(video_path)
test_fallback_chain()
test_safety_interventions()
```

### Benchmarks
```python
# Performance testing
bench_yolo_inference()
bench_hough_inference()
bench_end_to_end_latency()
bench_memory_usage()
```

## Extension Points

### Add New Detection Method
1. Implement detector class
2. Inherit from base interface
3. Register in sensor fusion
4. Add to fallback chain

### Add New Safety Constraint
1. Add metric to SafetyStatus
2. Add threshold to safety.yaml
3. Add check in SafetySupervisor
4. Add logging

### Add New Control Strategy
1. Implement controller class
2. Add configuration section
3. Integrate in main loop
4. Add visualization

## Dependencies

### Core
- `numpy`: Numerical computing
- `opencv-python`: Computer vision
- `PyYAML`: Configuration management

### AI
- `torch`: Deep learning framework
- `ultralytics`: YOLO implementation
- `torchvision`: Vision utilities

### Optional
- `rclpy`: ROS2 Python client (for robot integration)
- `sensor_msgs`: ROS2 messages
- `cv_bridge`: ROS2 OpenCV bridge

## Future Improvements

1. **Multi-lane support**: Detect multiple lanes simultaneously
2. **Object tracking**: Maintain lane identity across frames
3. **Weather adaptation**: Train on rain/snow/fog data
4. **Night vision**: IR camera support
5. **Terrain mapping**: 3D lane geometry
6. **Fleet coordination**: V2V communication
7. **ML-based fallback**: Train fallback detector on YOLO failures
8. **Real-time calibration**: Auto-camera calibration

## References

- Pure Pursuit: [Path Tracking for Autonomous Vehicles](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1.pdf)
- YOLO: [You Only Look Once](https://arxiv.org/abs/1905.12356)
- Hough Transform: [Standard Computer Vision Technique](https://en.wikipedia.org/wiki/Hough_transform)
- Safety Principles: [ISO 26262 Functional Safety](https://en.wikipedia.org/wiki/ISO_26262)
