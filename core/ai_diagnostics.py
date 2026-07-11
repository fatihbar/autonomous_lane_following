"""
AI diagnostics - explain system behavior and decisions.
"""

from typing import List, Optional
from datetime import datetime

from .data_types import (
    FusionResult,
    SafetyStatus,
    DetectionSource,
    SystemDiagnostics,
)
from .logger import setup_logger

logger = setup_logger(__name__)


class AIDiagnostics:
    """
    Provide diagnostics and explanations for system decisions.
    """
    
    @staticmethod
    def explain_lane_loss(fusion_result: FusionResult) -> str:
        """
        Explain why lane detection is lost.
        
        Args:
            fusion_result: Fusion result
            
        Returns:
            Explanation string
        """
        if fusion_result.lane_source != DetectionSource.LOST:
            return "Lane is detected"
        
        reasons = []
        
        if fusion_result.fusion_confidence < 0.2:
            reasons.append("All detection sources failed")
        
        if fusion_result.virtual_lane_active and fusion_result.virtual_lane_duration_s > 2.0:
            reasons.append("Virtual lane active too long")
        
        if not reasons:
            reasons.append("Unknown cause")
        
        return " | ".join(reasons)
    
    @staticmethod
    def explain_lane_source(fusion_result: FusionResult) -> str:
        """
        Explain which source is being used.
        
        Args:
            fusion_result: Fusion result
            
        Returns:
            Explanation string
        """
        source_map = {
            DetectionSource.YOLO: "YOLO segmentation (primary)",
            DetectionSource.HOUGH: "Hough Transform fallback",
            DetectionSource.YOLO_HOUGH: "YOLO + Hough fusion",
            DetectionSource.VIRTUAL_LANE: "Virtual lane from history",
            DetectionSource.LIDAR_ASSISTED: "LiDAR-assisted detection",
            DetectionSource.LOST: "Lane lost - emergency mode",
        }
        
        base = source_map.get(fusion_result.lane_source, "Unknown source")
        confidence_str = f" (confidence: {fusion_result.fusion_confidence:.2f})"
        
        return base + confidence_str
    
    @staticmethod
    def get_system_status(safety_status: SafetyStatus) -> List[str]:
        """
        Get human-readable system status.
        
        Args:
            safety_status: Safety status
            
        Returns:
            List of status strings
        """
        status = []
        status.append(f"Mode: {safety_status.system_mode.value}")
        status.append(f"FPS: {safety_status.current_fps:.1f}")
        status.append(f"Lane confidence: {safety_status.lane_confidence:.2f}")
        status.append(f"Fusion confidence: {safety_status.fusion_confidence:.2f}")
        status.append(f"Inference latency: {safety_status.inference_latency_ms:.1f}ms")
        
        if safety_status.virtual_lane_active:
            status.append(f"Virtual lane: {safety_status.virtual_lane_duration_s:.1f}s active")
        
        if safety_status.last_warning:
            status.append(f"Warning: {safety_status.last_warning}")
        
        return status
