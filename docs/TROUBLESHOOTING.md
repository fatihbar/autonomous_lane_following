# Troubleshooting Guide

## Common Issues and Solutions

### GPU / CUDA Issues

**Problem**: CUDA not detected, using CPU
```
WARNING: GPU not available, using CPU for inference
```

**Solutions**:
1. Verify NVIDIA GPU installed:
   ```bash
   nvidia-smi
   ```

2. Install CUDA toolkit:
   ```bash
   # Ubuntu
   sudo apt-get install cuda
   
   # Then reinstall torch
   pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu118
   ```

3. Set CUDA_VISIBLE_DEVICES:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   python scripts/run_standalone.py ...
   ```

### Performance Issues

**Problem**: Low FPS (< 10 fps)

**Solutions**:
1. **Reduce input size**:
   ```yaml
   # config/perception.yaml
   model:
     imgsz: 480  # Instead of 640
   ```

2. **Use FP16 precision**:
   ```yaml
   model:
     use_half: true
   ```

3. **Disable visualization**:
   ```yaml
   # config/visualization.yaml
   visualization:
     enabled: false
   ```

4. **Check system resources**:
   ```bash
   # Monitor GPU
   nvidia-smi -l 1
   
   # Monitor CPU/RAM
   top
   ```

**Problem**: High inference latency (> 100ms)

**Solutions**:
1. Check YOLO batch size (should be 1)
2. Reduce image resolution (see above)
3. Close background applications
4. Use dedicated GPU if available
5. Check for disk I/O bottlenecks:
   ```bash
   iostat -x 1
   ```

### Lane Detection Issues

**Problem**: Lane not detected (LOST mode)

**Diagnosis**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with visualization
python scripts/run_standalone.py --video path/to/video.mp4
# Watch confidence metrics
```

**Solutions**:

1. **Low lighting**:
   - Check exposure settings (if using USB camera)
   - Pre-process image with histogram equalization:
   ```python
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   enhanced = clahe.apply(gray_image)
   ```

2. **Poor contrast**:
   - Adjust YOLO confidence threshold lower:
   ```yaml
   model:
     confidence_threshold: 0.4  # From 0.5
   ```

3. **Road markings not trained**:
   - Hough fallback should work if YOLO fails
   - Increase Hough confidence threshold if needed:
   ```yaml
   # config/hough_fallback.yaml
   confidence:
     threshold: 0.3  # Or lower
   ```

4. **Model issues**:
   - Verify model file exists:
   ```bash
   ls -lh models/lane_segmentation.pt
   ```
   - Check model compatibility:
   ```python
   from ultralytics import YOLO
   model = YOLO("models/lane_segmentation.pt")
   print(model.info())
   ```

**Problem**: Flickering lane detection

**Solutions**:
1. Increase smoothing alpha:
   ```yaml
   path_planning:
     smoothing:
       alpha: 0.5  # Higher = more smoothing
   ```

2. Increase virtual lane history:
   ```yaml
   fusion:
     virtual_lane:
       history_size: 60  # From 30
   ```

### Control Issues

**Problem**: Vehicle not responding to steering commands

**Check**:
1. Verify dry-run mode is disabled:
   ```yaml
   # config/safety.yaml
   vehicle_control:
     dry_run: false  # For real vehicle
     enable_vehicle_output: true
   ```

2. Check control command output:
   ```bash
   # Add debug print in scripts/run_standalone.py
   print(f"Steering: {result['control_cmd'].steering_angle_deg}°")
   print(f"Throttle: {result['control_cmd'].throttle_command}")
   ```

3. Verify vehicle is in autonomous mode

**Problem**: Steering oscillates

**Solutions**:
1. Reduce steering smoothing alpha:
   ```yaml
   controller:
     pure_pursuit:
       steering_smoothing_alpha: 0.2  # From 0.3
   ```

2. Increase lookahead distance:
   ```yaml
   controller:
     pure_pursuit:
       lookahead_distance_m: 1.5  # From 1.0
   ```

3. Check for noisy lane detections

### Safety Issues

**Problem**: Emergency stop triggered unexpectedly

**Check**:
1. **No lane detected for > 2s**:
   ```bash
   # Look for messages like:
   # WARNING: No lane for 2.5s > 2.0s
   ```
   Solution: Lower confidence threshold or improve detection

2. **Virtual lane exceeded 3s**:
   ```bash
   # Look for messages like:
   # WARNING: Virtual lane for 3.5s > 3.0s
   ```
   Solution: Improve real detection or extend virtual lane duration

3. **Low FPS (< 10)**:
   Solution: Optimize performance (see above)

4. **High inference latency (> 100ms)**:
   Solution: Reduce resolution, use FP16, optimize code

**Problem**: System in DEGRADED mode instead of AUTONOMOUS

**Reason**: Fusion confidence < 0.25

**Solutions**:
1. Check lane confidence in visualization
2. Improve YOLO detection (better model, better training data)
3. Ensure Hough fallback is working
4. Lower confidence thresholds (with caution):
   ```yaml
   fusion:
     confidence_thresholds:
       min_fusion: 0.20  # From 0.25
   ```

### Memory Issues

**Problem**: Out of memory (OOM) error

**Solutions**:
1. Reduce YOLO model size:
   ```python
   # Use nano or small version
   detector = YOLOLaneDetector(
       model_path="models/yolov8n-seg.pt",  # nano
   )
   ```

2. Reduce virtual lane history:
   ```yaml
   fusion:
     virtual_lane:
       history_size: 10  # From 30
   ```

3. Disable visualization:
   ```yaml
   visualization:
     enabled: false
   ```

4. Process lower resolution video:
   ```bash
   # Convert video to lower resolution
   ffmpeg -i input.mp4 -vf scale=480:-1 input_480p.mp4
   ```

### Video Input Issues

**Problem**: Video file not opening

**Solutions**:
1. Check file path:
   ```bash
   python scripts/run_standalone.py --video path/to/video.mp4
   # Error shows correct path
   ```

2. Check codec support:
   ```bash
   # Try with different format
   ffmpeg -i input.avi -c:v libx264 -crf 23 output.mp4
   ```

3. Check file integrity:
   ```bash
   ffmpeg -v error -i input.mp4 -f null -
   ```

**Problem**: Camera not opening

**Solutions**:
1. Check camera permissions:
   ```bash
   # Linux
   sudo usermod -a -G video $USER
   ```

2. Test camera directly:
   ```bash
   # Using OpenCV
   python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read())"
   ```

3. Try different camera index:
   ```bash
   python scripts/run_standalone.py --video 1  # Try index 1, 2, etc.
   ```

### Logging and Debugging

**Enable verbose logging**:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Check logs**:
```bash
# View recent logs
tail -f logs/autonomous_lane_following_*.log

# Filter by error
grep ERROR logs/*.log
```

**Enable debug visualization**:
```yaml
# config/visualization.yaml
debug:
  view: true
  publish_debug_image: true
```

### Getting Help

If issue persists:

1. **Collect diagnostics**:
   ```bash
   # System info
   uname -a
   nvidia-smi
   python --version
   pip list
   ```

2. **Save relevant logs**:
   ```bash
   tar -czf logs.tar.gz logs/
   ```

3. **Create minimal reproducible example**

4. **Open GitHub issue** with:
   - System information
   - Error messages and stack traces
   - Steps to reproduce
   - Input video/image (if possible)
   - Configuration YAML files

---

**Note**: Most issues can be resolved by:
1. Checking logs first
2. Verifying configuration
3. Testing individual components
4. Reducing complexity (resolution, features)
