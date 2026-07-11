"""
Multi-sensor fusion for lane detection.
Combines YOLO, Hough, Virtual lane, and LiDAR data.
"""

import numpy as np
from typing import Optional, Dict
from datetime import datetime
import time

from .data_types import (
    LaneGeometry,
    FusionResult,
    DetectionSource,
    YOLODetectionResult,
    HoughDetectionResult,
    LiDARDetectionResult,
)
from .logger import setup_logger

logger = setup_logger(__name__)


class SensorFusion:
    """
    Fuse lane detections from multiple sources.
    """
    
    def __init__(
        self,
        yolo_weight: float = 0.7,
        hough_weight: float = 0.4,
        virtual_lane_weight: float = 0.2,
        lidar_weight: float = 0.3,
        yolo_confidence_threshold: float = 0.5,
        hough_confidence_threshold: float = 0.3,
        min_fusion_confidence: float = 0.25,
    ):
        """
        Initialize sensor fusion.
        
        Args:
            yolo_weight: Weight for YOLO detections
            hough_weight: Weight for Hough detections
            virtual_lane_weight: Weight for virtual lane
            lidar_weight: Weight for LiDAR validation
            yolo_confidence_threshold: Min confidence for YOLO to be primary
            hough_confidence_threshold: Min confidence for Hough to be used
            min_fusion_confidence: Minimum fusion confidence for valid result
        """
        self.yolo_weight = yolo_weight
        self.hough_weight = hough_weight
        self.virtual_lane_weight = virtual_lane_weight
        self.lidar_weight = lidar_weight
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.hough_confidence_threshold = hough_confidence_threshold
        self.min_fusion_confidence = min_fusion_confidence
    
    def fuse(
        self,
        yolo_geometry: Optional[LaneGeometry],
        hough_geometry: Optional[LaneGeometry],
        virtual_geometry: Optional[LaneGeometry],
        lidar_result: Optional[LiDARDetectionResult],
        virtual_duration_s: float = 0.0,
    ) -> FusionResult:
        """
        Fuse multiple lane detections.
        
        Args:
            yolo_geometry: YOLO detection result
            hough_geometry: Hough Transform result
            virtual_geometry: Virtual lane result
            lidar_result: LiDAR obstacle detection
            virtual_duration_s: Duration virtual lane has been active
            
        Returns:
            FusionResult with best estimate and source
        """
        start_time = time.time()
        
        try:
            # Determine best source
            source = DetectionSource.LOST
            final_geometry = None
            contributions = {}
            lidar_assisted = False
            
            # Priority-based fusion
            if yolo_geometry is not None and yolo_geometry.confidence >= self.yolo_confidence_threshold:
                # YOLO is primary
                source = DetectionSource.YOLO
                final_geometry = yolo_geometry
                contributions["YOLO"] = self.yolo_weight
                
                # Try to fuse with Hough if available
                if (
                    hough_geometry is not None
                    and hough_geometry.confidence >= self.hough_confidence_threshold
                ):
                    source = DetectionSource.YOLO_HOUGH
                    contributions["HOUGH"] = self.hough_weight * 0.5  # Lower weight as fusion
            
            elif hough_geometry is not None and hough_geometry.confidence >= self.hough_confidence_threshold:
                # Hough fallback
                source = DetectionSource.HOUGH
                final_geometry = hough_geometry
                contributions["HOUGH"] = self.hough_weight
            
            elif virtual_geometry is not None:
                # Virtual lane fallback
                source = DetectionSource.VIRTUAL_LANE
                final_geometry = virtual_geometry
                contributions["VIRTUAL"] = self.virtual_lane_weight
            
            else:
                # No detection - LOST
                source = DetectionSource.LOST
                final_geometry = None
            
            # Calculate fusion confidence
            if final_geometry is not None:
                fusion_confidence = final_geometry.confidence
                
                # LiDAR validation
                if lidar_result is not None and not lidar_result.emergency_stop_required:
                    fusion_confidence = min(
                        1.0,
                        fusion_confidence * (1 + self.lidar_weight * 0.2)
                    )
                    lidar_assisted = True
            else:
                fusion_confidence = 0.0
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return FusionResult(
                final_lane=final_geometry if final_geometry is not None else self._empty_geometry(),
                lane_source=source,
                fusion_confidence=fusion_confidence,
                virtual_lane_active=source == DetectionSource.VIRTUAL_LANE,
                virtual_lane_duration_s=virtual_duration_s,
                lidar_assisted=lidar_assisted,
                source_contributions=contributions,
            )
        
        except Exception as e:
            logger.error(f"Error during fusion: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return FusionResult(
                final_lane=self._empty_geometry(),
                lane_source=DetectionSource.LOST,
                fusion_confidence=0.0,
                virtual_lane_active=False,
                virtual_lane_duration_s=0.0,
                lidar_assisted=False,
                source_contributions={},
            )
    
    @staticmethod
    def _empty_geometry() -> LaneGeometry:
        """
        Create empty lane geometry.
        """
        return LaneGeometry(
            left_boundary=None,
            right_boundary=None,
            centerline=None,
            confidence=0.0,
            source=DetectionSource.LOST,
        )
