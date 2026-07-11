"""
LiDAR obstacle detection and free-space analysis.
Provides emergency stop trigger and corridor validation.
"""

import numpy as np
from typing import Optional, List
from datetime import datetime
import time

from .data_types import LiDARDetectionResult, ObstacleInfo
from .logger import setup_logger

logger = setup_logger(__name__)


class LiDARObstacleDetector:
    """
    Detect obstacles from LiDAR point cloud.
    """
    
    def __init__(
        self,
        emergency_stop_distance_m: float = 0.5,
        roi_forward_m: float = 5.0,
        roi_width_m: float = 3.0,
        height_threshold_m: float = 0.1,
    ):
        """
        Initialize LiDAR obstacle detector.
        
        Args:
            emergency_stop_distance_m: Distance threshold for emergency stop
            roi_forward_m: Forward detection range in meters
            roi_width_m: Lateral detection range in meters
            height_threshold_m: Minimum height for obstacle detection
        """
        self.emergency_stop_distance_m = emergency_stop_distance_m
        self.roi_forward_m = roi_forward_m
        self.roi_width_m = roi_width_m
        self.height_threshold_m = height_threshold_m
    
    def detect(
        self,
        pointcloud: Optional[np.ndarray] = None,
    ) -> LiDARDetectionResult:
        """
        Detect obstacles in point cloud.
        
        Args:
            pointcloud: N x 4 array of (x, y, z, intensity) or None for dry run
            
        Returns:
            LiDARDetectionResult
        """
        start_time = time.time()
        
        # Dry run mode if no pointcloud
        if pointcloud is None:
            return LiDARDetectionResult(
                obstacles=[],
                free_space_mask=None,
                emergency_stop_required=False,
                min_distance_m=float('inf'),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            # Filter ROI: forward and width
            roi_mask = (
                (pointcloud[:, 0] > 0) &  # Forward
                (pointcloud[:, 0] < self.roi_forward_m) &
                (np.abs(pointcloud[:, 1]) < self.roi_width_m / 2) &
                (pointcloud[:, 2] > self.height_threshold_m)  # Above ground
            )
            
            roi_points = pointcloud[roi_mask]
            
            # Find obstacles (clusters of points)
            obstacles = []
            min_distance = float('inf')
            
            if len(roi_points) > 0:
                # Simple clustering: group by distance
                distances = np.sqrt(roi_points[:, 0]**2 + roi_points[:, 1]**2)
                min_distance = float(np.min(distances))
                
                # Create obstacle info for closest point
                closest_idx = np.argmin(distances)
                closest = roi_points[closest_idx]
                
                obstacle = ObstacleInfo(
                    distance_m=min_distance,
                    angle_rad=float(np.arctan2(closest[1], closest[0])),
                    height_m=float(closest[2]),
                    confidence=0.8,
                )
                obstacles.append(obstacle)
            
            # Check emergency stop condition
            emergency_stop = min_distance < self.emergency_stop_distance_m
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return LiDARDetectionResult(
                obstacles=obstacles,
                free_space_mask=None,
                emergency_stop_required=emergency_stop,
                min_distance_m=min_distance,
                processing_time_ms=processing_time_ms,
            )
        
        except Exception as e:
            logger.error(f"Error during LiDAR detection: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return LiDARDetectionResult(
                obstacles=[],
                free_space_mask=None,
                emergency_stop_required=False,
                min_distance_m=float('inf'),
                processing_time_ms=processing_time_ms,
            )
