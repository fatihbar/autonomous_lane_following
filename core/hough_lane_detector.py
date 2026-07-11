"""
Hough Transform-based lane detection as fallback.
Provides robust lane detection when YOLO confidence is low.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
import time

from .data_types import HoughDetectionResult, LanePoint
from .logger import setup_logger

logger = setup_logger(__name__)


class HoughLaneDetector:
    """
    Lane detection using Hough Transform as fallback.
    """
    
    def __init__(
        self,
        canny_low: float = 50,
        canny_high: float = 150,
        hough_rho: float = 1.0,
        hough_theta: float = np.pi / 180,
        hough_threshold: int = 40,
        min_line_length: float = 40,
        max_line_gap: float = 80,
        min_slope_abs: float = 0.3,
        max_slope_abs: float = 3.0,
        lane_width_pixels: float = 200,
    ):
        """
        Initialize Hough lane detector.
        
        Args:
            canny_low: Canny edge low threshold
            canny_high: Canny edge high threshold
            hough_rho: Hough transform rho parameter
            hough_theta: Hough transform theta parameter
            hough_threshold: Hough transform threshold
            min_line_length: Minimum line segment length
            max_line_gap: Maximum gap to connect line segments
            min_slope_abs: Minimum absolute slope for lane lines
            max_slope_abs: Maximum absolute slope for lane lines
            lane_width_pixels: Expected lane width in pixels
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_slope_abs = min_slope_abs
        self.max_slope_abs = max_slope_abs
        self.lane_width_pixels = lane_width_pixels
    
    def detect(
        self,
        image: np.ndarray,
        roi_top_ratio: float = 0.55,
        roi_bottom_ratio: float = 1.0,
    ) -> HoughDetectionResult:
        """
        Detect lanes using Hough Transform.
        
        Args:
            image: Input BGR image (H, W, 3)
            roi_top_ratio: Top of ROI as fraction of height
            roi_bottom_ratio: Bottom of ROI as fraction of height
            
        Returns:
            HoughDetectionResult with left/right lane points
        """
        start_time = time.time()
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            # Apply ROI
            h, w = edges.shape
            roi_top = int(h * roi_top_ratio)
            roi_bottom = int(h * roi_bottom_ratio)
            
            roi_edges = np.zeros_like(edges)
            roi_edges[roi_top:roi_bottom, :] = edges[roi_top:roi_bottom, :]
            
            # Hough Line Transform
            lines = cv2.HoughLinesP(
                roi_edges,
                rho=self.hough_rho,
                theta=self.hough_theta,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap,
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            if lines is None or len(lines) == 0:
                return HoughDetectionResult(
                    left_line_points=[],
                    right_line_points=[],
                    is_valid=False,
                    confidence=0.0,
                    line_segments=[],
                    processing_time_ms=processing_time_ms,
                )
            
            # Extract and categorize line segments
            left_segments = []
            right_segments = []
            center_x = w // 2
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Skip horizontal lines
                if y2 == y1:
                    continue
                
                # Calculate slope
                slope = (x2 - x1) / (y2 - y1)
                slope_abs = abs(slope)
                
                # Filter by slope
                if slope_abs < self.min_slope_abs or slope_abs > self.max_slope_abs:
                    continue
                
                # Categorize as left or right
                mid_x = (x1 + x2) / 2
                
                if mid_x < center_x and slope > 0:  # Left lane (positive slope going up)
                    left_segments.append(((x1, y1), (x2, y2), slope))
                elif mid_x >= center_x and slope < 0:  # Right lane (negative slope going up)
                    right_segments.append(((x1, y1), (x2, y2), slope))
            
            # Convert segments to points
            left_points = self._segments_to_points(left_segments, roi_top, h)
            right_points = self._segments_to_points(right_segments, roi_top, h)
            
            # Calculate confidence
            confidence = 0.0
            if len(left_points) > 0 or len(right_points) > 0:
                # Confidence based on number of detected segments and consistency
                segment_count = len(left_segments) + len(right_segments)
                confidence = min(1.0, segment_count / 10.0)
            
            return HoughDetectionResult(
                left_line_points=left_points,
                right_line_points=right_points,
                is_valid=confidence > 0.2,
                confidence=confidence,
                line_segments=[(float(s[0][0]), float(s[0][1]), float(s[1][0]), float(s[1][1])) for seg in left_segments + right_segments for s in [seg]],
                processing_time_ms=processing_time_ms,
            )
        
        except Exception as e:
            logger.error(f"Error during Hough detection: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return HoughDetectionResult(
                left_line_points=[],
                right_line_points=[],
                is_valid=False,
                confidence=0.0,
                line_segments=[],
                processing_time_ms=processing_time_ms,
            )
    
    @staticmethod
    def _segments_to_points(
        segments: List[Tuple],
        roi_top: int,
        image_height: int,
    ) -> List[LanePoint]:
        """
        Convert line segments to ordered points.
        
        Args:
            segments: List of ((x1,y1), (x2,y2), slope) tuples
            roi_top: Top of ROI
            image_height: Image height
            
        Returns:
            List of LanePoint objects sorted by y coordinate
        """
        if not segments:
            return []
        
        # Collect all points
        all_points = []
        for (x1, y1), (x2, y2), _ in segments:
            all_points.append((x1, y1))
            all_points.append((x2, y2))
        
        if not all_points:
            return []
        
        # Sort by y coordinate (top to bottom)
        all_points.sort(key=lambda p: p[1])
        
        # Create LanePoint objects
        return [LanePoint(x=float(x), y=float(y), confidence=1.0) for x, y in all_points]
