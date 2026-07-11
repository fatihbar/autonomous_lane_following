"""
Lane geometry utilities for combining detections and creating geometry.
"""

import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime

from .data_types import LaneBoundary, LaneGeometry, LanePoint, DetectionSource, HoughDetectionResult
from .logger import setup_logger

logger = setup_logger(__name__)


class LaneGeometryBuilder:
    """
    Build lane geometry from various detection sources.
    """
    
    def __init__(
        self,
        polyfit_degree: int = 2,
        lane_width_pixels: float = 200,
    ):
        """
        Initialize geometry builder.
        
        Args:
            polyfit_degree: Polynomial degree for fitting
            lane_width_pixels: Default lane width
        """
        self.polyfit_degree = polyfit_degree
        self.lane_width_pixels = lane_width_pixels
    
    def from_hough_result(
        self,
        hough_result: HoughDetectionResult,
        image_height: int,
        image_width: int,
    ) -> LaneGeometry:
        """
        Create lane geometry from Hough detection result.
        
        Args:
            hough_result: HoughDetectionResult
            image_height: Image height
            image_width: Image width
            
        Returns:
            LaneGeometry
        """
        try:
            # Convert points to boundaries
            left_boundary = None
            right_boundary = None
            
            if len(hough_result.left_line_points) > 0:
                left_boundary = self._points_to_boundary(
                    hough_result.left_line_points,
                    hough_result.confidence,
                )
            
            if len(hough_result.right_line_points) > 0:
                right_boundary = self._points_to_boundary(
                    hough_result.right_line_points,
                    hough_result.confidence,
                )
            
            # Create centerline
            centerline = self._create_centerline(left_boundary, right_boundary)
            
            # Estimate lane width
            lane_width = self._estimate_lane_width_from_boundaries(
                left_boundary,
                right_boundary,
            )
            
            return LaneGeometry(
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                centerline=centerline,
                lane_width_pixels=lane_width,
                confidence=hough_result.confidence,
                source=DetectionSource.HOUGH,
            )
        
        except Exception as e:
            logger.error(f"Error creating geometry from Hough result: {e}")
            return LaneGeometry(
                left_boundary=None,
                right_boundary=None,
                centerline=None,
                confidence=0.0,
                source=DetectionSource.LOST,
            )
    
    def _points_to_boundary(
        self,
        points: List[LanePoint],
        confidence: float,
    ) -> Optional[LaneBoundary]:
        """
        Convert list of LanePoint to LaneBoundary with polynomial fit.
        
        Args:
            points: List of LanePoint
            confidence: Confidence value
            
        Returns:
            LaneBoundary with fitted polynomial
        """
        if len(points) == 0:
            return None
        
        if len(points) < self.polyfit_degree + 1:
            return LaneBoundary(points=points, confidence=confidence)
        
        try:
            # Convert to numpy array
            points_array = np.array([[p.x, p.y] for p in points])
            
            # Sort by y
            points_array = points_array[points_array[:, 1].argsort()]
            
            # Fit polynomial
            poly = np.polyfit(points_array[:, 1], points_array[:, 0], self.polyfit_degree)
            
            return LaneBoundary(
                points=points,
                confidence=confidence,
                polynomial=poly,
            )
        
        except Exception as e:
            logger.warning(f"Could not fit polynomial: {e}")
            return LaneBoundary(points=points, confidence=confidence)
    
    @staticmethod
    def _create_centerline(
        left: Optional[LaneBoundary],
        right: Optional[LaneBoundary],
    ) -> Optional[LaneBoundary]:
        """
        Create centerline from left and right boundaries.
        """
        if left is None and right is None:
            return None
        
        if left is not None and right is not None:
            center_points = []
            # Use right boundary points as reference
            for point_r in right.points:
                x_l = left.point_at_y(point_r.y)
                if x_l is not None:
                    center_x = (x_l + point_r.x) / 2
                    center_points.append(LanePoint(x=center_x, y=point_r.y))
            
            if len(center_points) > 0:
                try:
                    points_array = np.array([[p.x, p.y] for p in center_points])
                    poly = np.polyfit(points_array[:, 1], points_array[:, 0], 2)
                    return LaneBoundary(
                        points=center_points,
                        confidence=min(left.confidence, right.confidence),
                        polynomial=poly,
                    )
                except:
                    return LaneBoundary(
                        points=center_points,
                        confidence=min(left.confidence, right.confidence),
                    )
        
        # Use single boundary
        boundary = left if left is not None else right
        return LaneBoundary(
            points=boundary.points,
            confidence=boundary.confidence,
            polynomial=boundary.polynomial,
        )
    
    @staticmethod
    def _estimate_lane_width_from_boundaries(
        left: Optional[LaneBoundary],
        right: Optional[LaneBoundary],
    ) -> float:
        """
        Estimate lane width from boundaries.
        """
        if left is None or right is None:
            return 200.0
        
        widths = []
        for point_r in right.points:
            x_l = left.point_at_y(point_r.y)
            if x_l is not None:
                widths.append(point_r.x - x_l)
        
        if widths:
            return float(np.median(widths))
        return 200.0
