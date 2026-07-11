"""
Geometry utilities for lane and path calculations.
"""

import numpy as np
from typing import Tuple, List, Optional
import math


def distance_point_to_line(
    point: Tuple[float, float],
    line_p1: Tuple[float, float],
    line_p2: Tuple[float, float],
) -> float:
    """
    Calculate distance from point to line.
    
    Args:
        point: (x, y) point
        line_p1: First point on line
        line_p2: Second point on line
        
    Returns:
        Distance
    """
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    if denominator == 0:
        return float('inf')
    
    return numerator / denominator


def angle_between_vectors(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
) -> float:
    """
    Calculate angle between two vectors in radians.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(det, dot)
