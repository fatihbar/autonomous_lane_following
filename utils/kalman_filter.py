"""
Kalman filter for smooth state estimation.
"""

import numpy as np
from typing import Tuple


class KalmanFilter1D:
    """
    Simple 1D Kalman filter for position estimation.
    """
    
    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1,
        initial_value: float = 0.0,
    ):
        """
        Initialize Kalman filter.
        
        Args:
            process_variance: Process noise (Q)
            measurement_variance: Measurement noise (R)
            initial_value: Initial state estimate
        """
        self.q = process_variance
        self.r = measurement_variance
        self.x = initial_value  # State
        self.p = 1.0  # Estimation error
        self.k = 0.0  # Kalman gain
    
    def update(self, measurement: float) -> float:
        """
        Update with new measurement.
        
        Args:
            measurement: New measurement
            
        Returns:
            Updated state estimate
        """
        # Prediction
        self.p = self.p + self.q
        
        # Update
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        
        return self.x
