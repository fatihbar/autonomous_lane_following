"""
Process YOLO lane mask to extract lane boundaries and geometry.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime

from .data_types import LaneMask, LaneBoundary, LaneGeometry, LanePoint, DetectionSource
from .logger import setup_logger

logger = setup_logger(__name__)


class LaneMaskProcessor:
    """
    Extract lane boundaries from YOLO segmentation mask.
    """
    
    def __init__(
        self,
        min_lane_area: int = 100,
        polyfit_degree: int = 2,
        lane_width_pixels: float = 200,
        morph_kernel_size: int = 5,
    ):
        """
        Initialize lane mask processor.
        
        Args:
            min_lane_area: Minimum pixels for valid lane
            polyfit_degree: Polynomial degree for lane fitting
            lane_width_pixels: Expected lane width for missing side estimation
            morph_kernel_size: Kernel size for morphological operations
        """
        self.min_lane_area = min_lane_area
        self.polyfit_degree = polyfit_degree
        self.lane_width_pixels = lane_width_pixels
        self.morph_kernel_size = morph_kernel_size
    
    def process(
        self,
        lane_mask: LaneMask,
        image_shape: Tuple[int, int],
    ) -> LaneGeometry:
        """
        Process lane mask to extract geometry.
        
        Args:
            lane_mask: YOLO lane mask
            image_shape: (height, width) of original image
            
        Returns:
            LaneGeometry with left/right/center boundaries
        """
        try:
            # Check mask validity
            if lane_mask.area() < self.min_lane_area:
                logger.debug(f"Mask area too small: {lane_mask.area()} < {self.min_lane_area}")
                return LaneGeometry(
                    left_boundary=None,
                    right_boundary=None,
                    centerline=None,
                    confidence=0.0,
                    source=DetectionSource.LOST,
                )
            
            # Clean mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            cleaned_mask = cv2.morphologyEx(lane_mask.mask, cv2.MORPH_OPEN, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return LaneGeometry(
                    left_boundary=None,
                    right_boundary=None,
                    centerline=None,
                    confidence=0.0,
                    source=DetectionSource.LOST,
                )
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Find points in contour
            points = largest_contour.squeeze()
            if points.ndim == 1:
                # Single point
                return LaneGeometry(
                    left_boundary=None,
                    right_boundary=None,
                    centerline=None,
                    confidence=0.0,
                    source=DetectionSource.LOST,
                )
            
            # Separate left and right lane points (by x coordinate)
            h, w = image_shape
            center_x = w // 2
            
            left_mask_points = points[points[:, 0] < center_x]
            right_mask_points = points[points[:, 0] >= center_x]
            
            # Extract lane boundaries
            left_boundary = None
            right_boundary = None
            
            if len(left_mask_points) > 0:
                left_boundary = self._fit_lane_boundary(left_mask_points, lane_mask.confidence)
            
            if len(right_mask_points) > 0:
                right_boundary = self._fit_lane_boundary(right_mask_points, lane_mask.confidence)
            
            # Create centerline
            centerline = self._create_centerline(left_boundary, right_boundary)
            
            # Estimate lane width
            if left_boundary and right_boundary:
                lane_width = self._estimate_lane_width(left_boundary, right_boundary)
            else:
                lane_width = self.lane_width_pixels
            
            return LaneGeometry(
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                centerline=centerline,
                lane_width_pixels=lane_width,
                confidence=lane_mask.confidence,
                source=DetectionSource.YOLO,
            )
        
        except Exception as e:
            logger.error(f"Error processing lane mask: {e}")
            return LaneGeometry(
                left_boundary=None,
                right_boundary=None,
                centerline=None,
                confidence=0.0,
                source=DetectionSource.LOST,
            )
    
    def _fit_lane_boundary(
        self,
        points: np.ndarray,
        confidence: float,
    ) -> LaneBoundary:
        """
        Fit polynomial to lane boundary points.
        
        Args:
            points: Array of (x, y) points
            confidence: Detection confidence
            
        Returns:
            LaneBoundary with fitted polynomial
        """
        if len(points) < self.polyfit_degree + 1:
            # Not enough points
            lane_points = [LanePoint(x=float(p[0]), y=float(p[1])) for p in points]
            return LaneBoundary(points=lane_points, confidence=confidence)
        
        # Fit polynomial
        try:
            poly = np.polyfit(points[:, 1], points[:, 0], self.polyfit_degree)
            lane_points = [LanePoint(x=float(p[0]), y=float(p[1])) for p in points]
            
            return LaneBoundary(
                points=lane_points,
                confidence=confidence,
                polynomial=poly,
            )
        except Exception as e:
            logger.warning(f"Could not fit polynomial: {e}")
            lane_points = [LanePoint(x=float(p[0]), y=float(p[1])) for p in points]
            return LaneBoundary(points=lane_points, confidence=confidence)
    
    @staticmethod
    def _create_centerline(
        left: Optional[LaneBoundary],
        right: Optional[LaneBoundary],
    ) -> Optional[LaneBoundary]:
        """
        Create centerline from left and right boundaries.
        
        Args:
            left: Left boundary
            right: Right boundary
            
        Returns:
            Centerline LaneBoundary
        """
        if left is None and right is None:
            return None
        
        if left is not None and right is not None:
            # Average of left and right
            center_points = []
            # Use right boundary y values
            for point_r in right.points:
                x_l = left.point_at_y(point_r.y)
                if x_l is not None:
                    center_x = (x_l + point_r.x) / 2
                    center_points.append(LanePoint(x=center_x, y=point_r.y))
            
            if len(center_points) > 0:
                # Fit polynomial to centerline
                points_array = np.array([[p.x, p.y] for p in center_points])
                try:
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
        
        # Use single boundary as centerline
        boundary = left if left is not None else right
        return LaneBoundary(
            points=boundary.points,
            confidence=boundary.confidence,
            polynomial=boundary.polynomial,
        )
    
    @staticmethod
    def _estimate_lane_width(
        left: LaneBoundary,
        right: LaneBoundary,
    ) -> float:
        """
        Estimate lane width from boundaries.
        
        Args:
            left: Left boundary
            right: Right boundary
            
        Returns:
            Estimated lane width in pixels
        """
        widths = []
        for point_r in right.points:
            x_l = left.point_at_y(point_r.y)
            if x_l is not None:
                widths.append(point_r.x - x_l)
        
        if widths:
            return float(np.median(widths))
        return 0.0
