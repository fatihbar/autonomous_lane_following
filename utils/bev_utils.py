"""
Utilities for BEV (Bird's Eye View) transformations.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

from ..core.logger import setup_logger

logger = setup_logger(__name__)


class BEVTransformer:
    """
    Transform between camera view and bird's eye view.
    """
    
    def __init__(
        self,
        src_points: Optional[List[Tuple[float, float]]] = None,
        dst_points: Optional[List[Tuple[float, float]]] = None,
        output_width: int = 640,
        output_height: int = 480,
    ):
        """
        Initialize BEV transformer.
        
        Args:
            src_points: Source points in camera view (4 points)
            dst_points: Destination points in BEV (4 points)
            output_width: BEV output width
            output_height: BEV output height
        """
        self.output_width = output_width
        self.output_height = output_height
        
        # Default perspective points if not provided
        if src_points is None:
            # Trapezoid in camera view (bottom wider)
            self.src_points = np.array([
                [100, 400],
                [540, 400],
                [0, 480],
                [640, 480],
            ], dtype=np.float32)
        else:
            self.src_points = np.array(src_points, dtype=np.float32)
        
        if dst_points is None:
            # Rectangle in BEV
            self.dst_points = np.array([
                [100, 0],
                [540, 0],
                [0, 480],
                [640, 480],
            ], dtype=np.float32)
        else:
            self.dst_points = np.array(dst_points, dtype=np.float32)
        
        # Calculate transformation matrix
        self.transform_matrix = cv2.getPerspectiveTransform(
            self.src_points,
            self.dst_points,
        )
        self.inverse_matrix = cv2.getPerspectiveTransform(
            self.dst_points,
            self.src_points,
        )
    
    def to_bev(self, image: np.ndarray) -> np.ndarray:
        """
        Transform image to bird's eye view.
        
        Args:
            image: Input image
            
        Returns:
            BEV image
        """
        return cv2.warpPerspective(
            image,
            self.transform_matrix,
            (self.output_width, self.output_height),
        )
    
    def from_bev(self, image: np.ndarray) -> np.ndarray:
        """
        Transform from BEV back to camera view.
        
        Args:
            image: BEV image
            
        Returns:
            Camera view image
        """
        return cv2.warpPerspective(
            image,
            self.inverse_matrix,
            (640, 480),  # Original image size
        )
