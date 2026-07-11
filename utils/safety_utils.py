"""
Safety-related utility functions.
"""

from typing import List
from ..core.data_types import SafetyStatus, SystemMode


def get_safety_color(safety_status: SafetyStatus) -> tuple:
    """
    Get color based on safety status.
    
    Args:
        safety_status: Current safety status
        
    Returns:
        BGR color tuple
    """
    if safety_status.system_mode == SystemMode.EMERGENCY_STOP:
        return (0, 0, 255)  # Red
    elif safety_status.system_mode == SystemMode.DEGRADED:
        return (0, 165, 255)  # Orange
    elif safety_status.system_mode == SystemMode.VIRTUAL_LANE:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 255, 0)  # Green


def get_mode_text(mode: SystemMode) -> str:
    """
    Get human-readable mode text.
    
    Args:
        mode: System mode
        
    Returns:
        Mode text
    """
    return mode.value
