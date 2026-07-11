"""
Path planning from detected lane centerline.
Generates smooth paths and lookahead points.
"""

import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime
import time

from .data_types import LaneGeometry, Path
from .logger import setup_logger

logger = setup_logger(__name__)


class PathPlanner:
    """
    Plan vehicle path from detected lane.
    """
    
    def __init__(
        self,
        lookahead_distance_m: float = 1.0,
        lookahead_distance_pixels: float = 100,
        smoothing_alpha: float = 0.3,
    ):
        """
        Initialize path planner.
        
        Args:
            lookahead_distance_m: Lookahead distance in meters (for info)
            lookahead_distance_pixels: Lookahead distance in pixels
            smoothing_alpha: Exponential smoothing factor
        """
        self.lookahead_distance_m = lookahead_distance_m
        self.lookahead_distance_pixels = lookahead_distance_pixels
        self.smoothing_alpha = smoothing_alpha
        self.last_path: Optional[Path] = None
    
    def plan(
        self,
        geometry: Optional[LaneGeometry],
        image_height: int,
    ) -> Optional[Path]:
        """
        Plan path from lane geometry.
        
        Args:
            geometry: Detected lane geometry
            image_height: Image height for lookahead point calculation
            
        Returns:
            Path with centerline points and lookahead
        """
        start_time = time.time()
        
        if geometry is None or geometry.centerline is None:
            return None
        
        try:
            centerline = geometry.centerline
            
            # Convert centerline points to list
            centerline_points = [(p.x, p.y) for p in centerline.points]
            
            if len(centerline_points) == 0:
                return None
            
            # Find lookahead point
            # Use bottom of image as vehicle position
            vehicle_y = image_height
            lookahead_y = max(0, vehicle_y - self.lookahead_distance_pixels)
            
            # Get x coordinate at lookahead y
            if centerline.polynomial is not None:
                try:
                    lookahead_x = float(np.polyval(centerline.polynomial, lookahead_y))
                except:
                    lookahead_x = centerline_points[-1][0]
            else:
                # Interpolate from points
                lookahead_x = self._interpolate_x_at_y(centerline_points, lookahead_y)
            
            lookahead_point = (lookahead_x, lookahead_y)
            
            # Estimate curvature at lookahead point
            curvature = self._estimate_curvature(centerline)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            path = Path(
                centerline_points=centerline_points,
                lookahead_point=lookahead_point,
                lookahead_distance_m=self.lookahead_distance_m,
                curvature_estimate=curvature,
            )
            
            self.last_path = path
            return path
        
        except Exception as e:
            logger.error(f"Error planning path: {e}")
            return self.last_path
    
    @staticmethod
    def _interpolate_x_at_y(
        points: List[Tuple[float, float]],
        y_target: float,
    ) -> float:
        """
        Interpolate x coordinate at target y.
        
        Args:
            points: List of (x, y) tuples
            y_target: Target y coordinate
            
        Returns:
            Interpolated x coordinate
        """
        if not points:
            return 0.0
        
        # Find two points bracketing y_target
        for i in range(len(points) - 1):
            y1, y2 = points[i][1], points[i + 1][1]
            
            if min(y1, y2) <= y_target <= max(y1, y2):
                # Linear interpolation
                if y2 == y1:
                    return points[i][0]
                
                x1, x2 = points[i][0], points[i + 1][0]
                t = (y_target - y1) / (y2 - y1)
                return x1 + t * (x2 - x1)
        
        # Target outside range, return closest point
        return points[-1][0] if points else 0.0
    
    @staticmethod
    def _estimate_curvature(centerline) -> float:
        """
        Estimate path curvature from centerline polynomial.
        
        Args:
            centerline: LaneBoundary with polynomial
            
        Returns:
            Curvature estimate
        """
        if centerline.polynomial is None or len(centerline.polynomial) < 3:
            return 0.0
        
        try:
            # For quadratic polynomial ax^2 + bx + c
            # Curvature ~ 2*a at the vertex
            return float(2 * centerline.polynomial[0])
        except:
            return 0.0
