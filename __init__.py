"""
Autonomous Lane Following Package

A professional-grade autonomous lane following system for closed-track vehicles
with deep learning perception, safety supervision, and multiple fallback mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Fatih Barlas"
__email__ = "94764881+fatihbar@users.noreply.github.com"
__license__ = "MIT"

# Core imports
from core.data_types import (
    DetectionSource,
    SystemMode,
    LaneGeometry,
    FusionResult,
    ControlCommand,
    SafetyStatus,
)
from core.yolo_lane_detector import YOLOLaneDetector
from core.hough_lane_detector import HoughLaneDetector
from core.sensor_fusion import SensorFusion
from core.controller import VehicleController
from core.safety_supervisor import SafetySupervisor

__all__ = [
    "DetectionSource",
    "SystemMode",
    "LaneGeometry",
    "FusionResult",
    "ControlCommand",
    "SafetyStatus",
    "YOLOLaneDetector",
    "HoughLaneDetector",
    "SensorFusion",
    "VehicleController",
    "SafetySupervisor",
]
