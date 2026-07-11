"""
Core data types for autonomous lane following system.
Defines all message structures and enums.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum
import numpy as np
from datetime import datetime


class DetectionSource(Enum):
    """Lane detection source"""
    YOLO = "YOLO"
    HOUGH = "HOUGH"
    YOLO_HOUGH = "YOLO_HOUGH"
    VIRTUAL_LANE = "VIRTUAL_LANE"
    LIDAR_ASSISTED = "LIDAR_ASSISTED"
    LOST = "LOST"


class SystemMode(Enum):
    """Overall system operational mode"""
    MANUAL = "MANUAL"
    ASSISTED = "ASSISTED"
    AUTONOMOUS = "AUTONOMOUS"
    DEGRADED = "DEGRADED"
    HOUGH_FALLBACK = "HOUGH_FALLBACK"
    VIRTUAL_LANE = "VIRTUAL_LANE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class LanePoint:
    """A point on lane boundary"""
    x: float  # pixel x or world x
    y: float  # pixel y or world y
    confidence: float = 1.0
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class LaneMask:
    """Lane segmentation mask from YOLO"""
    mask: np.ndarray  # binary mask
    confidence: float  # model confidence 0-1
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def area(self) -> int:
        """Get pixel count in mask"""
        return int(np.sum(self.mask))


@dataclass
class LaneBoundary:
    """Left or right lane boundary"""
    points: List[LanePoint]  # ordered points along boundary
    confidence: float  # confidence 0-1
    polynomial: Optional[np.ndarray] = None  # poly coefficients if fitted
    is_virtual: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def point_at_y(self, y: float) -> Optional[float]:
        """Get x coordinate at given y using polynomial"""
        if self.polynomial is None:
            return None
        return float(np.polyval(self.polynomial, y))


@dataclass
class LaneGeometry:
    """Complete lane geometry"""
    left_boundary: Optional[LaneBoundary]
    right_boundary: Optional[LaneBoundary]
    centerline: Optional[LaneBoundary]
    lane_width_pixels: float = 0.0
    confidence: float = 0.0
    source: DetectionSource = DetectionSource.LOST
    timestamp: datetime = field(default_factory=datetime.now)
    debug_info: dict = field(default_factory=dict)


@dataclass
class YOLODetectionResult:
    """Result from YOLO lane detector"""
    lane_mask: Optional[LaneMask]
    is_valid: bool
    confidence: float  # 0-1
    mask_area: int
    processing_time_ms: float
    device_used: str  # "cuda" or "cpu"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HoughDetectionResult:
    """Result from Hough Transform fallback"""
    left_line_points: List[LanePoint]
    right_line_points: List[LanePoint]
    is_valid: bool
    confidence: float  # 0-1
    line_segments: List[Tuple[float, float, float, float]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ObstacleInfo:
    """Obstacle detected by LiDAR"""
    distance_m: float
    angle_rad: float
    height_m: float
    confidence: float
    
    def is_critical(self, threshold_m: float = 0.5) -> bool:
        return self.distance_m < threshold_m


@dataclass
class LiDARDetectionResult:
    """Result from LiDAR obstacle detection"""
    obstacles: List[ObstacleInfo]
    free_space_mask: Optional[np.ndarray]  # bird's eye view
    emergency_stop_required: bool
    min_distance_m: float
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FusionResult:
    """Result of multi-sensor fusion"""
    final_lane: LaneGeometry
    lane_source: DetectionSource
    fusion_confidence: float
    virtual_lane_active: bool
    virtual_lane_duration_s: float
    lidar_assisted: bool
    timestamp: datetime = field(default_factory=datetime.now)
    source_contributions: dict = field(default_factory=dict)  # {"YOLO": 0.7, "HOUGH": 0.3}


@dataclass
class Path:
    """Planned path for vehicle"""
    centerline_points: List[Tuple[float, float]]  # list of (x, y) in vehicle frame
    lookahead_point: Tuple[float, float]
    lookahead_distance_m: float
    curvature_estimate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ControlCommand:
    """Control command for vehicle"""
    steering_angle_deg: float  # positive = right
    throttle_command: float  # 0-1
    brake_command: float  # 0-1
    target_speed_mps: float
    confidence: float
    is_emergency_stop: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyStatus:
    """Current safety status"""
    system_mode: SystemMode
    emergency_stop_active: bool
    lane_confidence: float
    fusion_confidence: float
    inference_latency_ms: float
    current_fps: float
    virtual_lane_duration_s: float
    no_lane_duration_s: float
    last_warning: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemDiagnostics:
    """System health and diagnostics"""
    yolo_fps: float
    hough_fps: float
    fusion_fps: float
    controller_fps: float
    total_fps: float
    yolo_inference_ms: float
    hough_inference_ms: float
    fusion_inference_ms: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    active_modules: List[str]  # which modules are active
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
