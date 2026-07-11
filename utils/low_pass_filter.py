"""
Low-pass filter for signal smoothing.
"""

from typing import Optional


class LowPassFilter:
    """
    Exponential low-pass filter.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize low-pass filter.
        
        Args:
            alpha: Smoothing factor (0-1), higher = less smoothing
        """
        self.alpha = alpha
        self.last_value: Optional[float] = None
    
    def update(self, value: float) -> float:
        """
        Update filter with new value.
        
        Args:
            value: New value
            
        Returns:
            Filtered value
        """
        if self.last_value is None:
            self.last_value = value
            return value
        
        filtered = self.alpha * value + (1 - self.alpha) * self.last_value
        self.last_value = filtered
        return filtered
