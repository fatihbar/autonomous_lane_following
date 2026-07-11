"""
Virtual lane generation from historical centerline data.
Activates when both YOLO and Hough detections fail.
"""

import numpy as np
from typing import Optional, List, Deque
from collections import deque
from datetime import datetime
import time

from .data_types import LaneGeometry, LaneBoundary, LanePoint, DetectionSource
from .logger import setup_logger

logger = setup_logger(__name__)


class VirtualLaneGenerator:
    """
    Generate virtual lanes when real detection fails.
    Uses historical lane data and motion prediction.
    """
    
    def __init__(
        self,
        history_size: int = 30,
        max_virtual_duration_s: float = 3.0,
        smoothing_alpha: float = 0.3,
    ):
        """
        Initialize virtual lane generator.
        
        Args:
            history_size: Number of frames to keep in history
            max_virtual_duration_s: Maximum duration of virtual lane before safety intervention
            smoothing_alpha: Exponential smoothing factor for centerline
        """
        self.history_size = history_size
        self.max_virtual_duration_s = max_virtual_duration_s
        self.smoothing_alpha = smoothing_alpha
        
        # History of lane geometries
        self.geometry_history: Deque[LaneGeometry] = deque(maxlen=history_size)
        
        # Virtual lane state
        self.virtual_lane_active = False
        self.virtual_lane_start_time: Optional[datetime] = None
        self.last_good_geometry: Optional[LaneGeometry] = None
    
    def update(
        self,
        geometry: Optional[LaneGeometry],
        current_time: Optional[datetime] = None,
    ) -> tuple[Optional[LaneGeometry], float, bool]:
        """
        Update virtual lane generator.
        
        Args:
            geometry: Current detected geometry (can be None)
            current_time: Current time (uses now() if None)
            
        Returns:
            (geometry_to_use, virtual_duration_s, is_virtual)
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Update history if valid detection
        if geometry is not None and geometry.source != DetectionSource.LOST:
            self.geometry_history.append(geometry)
            self.last_good_geometry = geometry
            self.virtual_lane_active = False
            self.virtual_lane_start_time = None
            return geometry, 0.0, False
        
        # No valid detection - activate virtual lane if history available
        if self.last_good_geometry is None or len(self.geometry_history) == 0:
            # No history, cannot generate virtual lane
            logger.warning("No history available for virtual lane generation")
            return None, 0.0, False
        
        # Start virtual lane if not already active
        if not self.virtual_lane_active:
            self.virtual_lane_active = True
            self.virtual_lane_start_time = current_time
            logger.info("Virtual lane activated")
        
        # Calculate duration
        virtual_duration_s = (
            current_time - self.virtual_lane_start_time
        ).total_seconds()
        
        # Check if exceeded maximum duration
        if virtual_duration_s > self.max_virtual_duration_s:
            logger.warning(
                f"Virtual lane exceeded max duration: {virtual_duration_s:.2f}s > {self.max_virtual_duration_s}s"
            )
            return None, virtual_duration_s, True
        
        # Generate virtual lane from history
        virtual_geometry = self._generate_from_history()
        
        return virtual_geometry, virtual_duration_s, True
    
    def _generate_from_history(self) -> Optional[LaneGeometry]:
        """
        Generate virtual lane from historical centerlines.
        
        Returns:
            Virtual LaneGeometry
        """
        if not self.geometry_history:
            return None
        
        try:
            # Use most recent geometry as base
            recent = self.geometry_history[-1]
            
            if recent.centerline is None:
                return None
            
            # Create smoothed virtual centerline
            # Simple approach: use recent centerline with reduced confidence
            virtual_centerline = LaneBoundary(
                points=recent.centerline.points,
                confidence=max(0.3, recent.centerline.confidence * 0.5),  # Reduce confidence
                polynomial=recent.centerline.polynomial,
                is_virtual=True,
            )
            
            virtual_geometry = LaneGeometry(
                left_boundary=None,
                right_boundary=None,
                centerline=virtual_centerline,
                lane_width_pixels=recent.lane_width_pixels,
                confidence=max(0.2, recent.confidence * 0.4),  # Reduced confidence
                source=DetectionSource.VIRTUAL_LANE,
            )
            
            return virtual_geometry
        
        except Exception as e:
            logger.error(f"Error generating virtual lane: {e}")
            return None
    
    def reset(self):
        """
        Reset virtual lane history.
        """
        self.geometry_history.clear()
        self.virtual_lane_active = False
        self.virtual_lane_start_time = None
        self.last_good_geometry = None
        logger.info("Virtual lane generator reset")
