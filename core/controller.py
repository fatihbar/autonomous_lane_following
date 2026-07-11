"""
Vehicle control - steering, throttle, and brake commands.
Includes Pure Pursuit controller.
"""

import numpy as np
from typing import Optional
from datetime import datetime
import time

from .data_types import Path, ControlCommand, SafetyStatus, SystemMode
from .logger import setup_logger

logger = setup_logger(__name__)


class VehicleController:
    """
    Vehicle control using Pure Pursuit algorithm.
    """
    
    def __init__(
        self,
        wheelbase_m: float = 2.6,
        lookahead_distance_m: float = 1.0,
        max_steering_deg: float = 35.0,
        steering_smoothing_alpha: float = 0.3,
        max_speed_mps: float = 2.0,
        min_speed_mps: float = 0.5,
        slow_speed_mps: float = 0.8,
    ):
        """
        Initialize vehicle controller.
        
        Args:
            wheelbase_m: Vehicle wheelbase in meters
            lookahead_distance_m: Pure pursuit lookahead distance
            max_steering_deg: Maximum steering angle
            steering_smoothing_alpha: Exponential smoothing for steering
            max_speed_mps: Maximum speed in m/s
            min_speed_mps: Minimum speed in m/s
            slow_speed_mps: Speed when reducing due to low confidence
        """
        self.wheelbase_m = wheelbase_m
        self.lookahead_distance_m = lookahead_distance_m
        self.max_steering_deg = max_steering_deg
        self.steering_smoothing_alpha = steering_smoothing_alpha
        self.max_speed_mps = max_speed_mps
        self.min_speed_mps = min_speed_mps
        self.slow_speed_mps = slow_speed_mps
        
        self.last_steering_deg = 0.0
    
    def control(
        self,
        path: Optional[Path],
        safety_status: SafetyStatus,
    ) -> ControlCommand:
        """
        Generate control command.
        
        Args:
            path: Planned path
            safety_status: Current safety status
            
        Returns:
            ControlCommand
        """
        start_time = time.time()
        
        # Default emergency command
        if safety_status.system_mode == SystemMode.EMERGENCY_STOP:
            return ControlCommand(
                steering_angle_deg=0.0,
                throttle_command=0.0,
                brake_command=1.0,
                target_speed_mps=0.0,
                confidence=1.0,
                is_emergency_stop=True,
            )
        
        # No path - stop
        if path is None:
            return ControlCommand(
                steering_angle_deg=0.0,
                throttle_command=0.0,
                brake_command=0.5,
                target_speed_mps=0.0,
                confidence=0.0,
                is_emergency_stop=False,
            )
        
        try:
            # Pure Pursuit steering
            steering_angle_deg = self._pure_pursuit_steering(path)
            
            # Smooth steering
            steering_angle_deg = self._smooth_steering(steering_angle_deg)
            
            # Clamp steering
            steering_angle_deg = np.clip(
                steering_angle_deg,
                -self.max_steering_deg,
                self.max_steering_deg,
            )
            
            # Speed control based on confidence
            target_speed = self._calculate_target_speed(safety_status)
            
            # Convert to throttle/brake
            throttle, brake = self._speed_to_control(target_speed)
            
            return ControlCommand(
                steering_angle_deg=steering_angle_deg,
                throttle_command=throttle,
                brake_command=brake,
                target_speed_mps=target_speed,
                confidence=safety_status.fusion_confidence,
                is_emergency_stop=False,
            )
        
        except Exception as e:
            logger.error(f"Error generating control command: {e}")
            return ControlCommand(
                steering_angle_deg=0.0,
                throttle_command=0.0,
                brake_command=0.5,
                target_speed_mps=0.0,
                confidence=0.0,
                is_emergency_stop=False,
            )
    
    def _pure_pursuit_steering(self, path: Path) -> float:
        """
        Calculate steering angle using Pure Pursuit algorithm.
        
        Args:
            path: Planned path
            
        Returns:
            Steering angle in degrees
        """
        # Lookahead point in image coordinates
        lx, ly = path.lookahead_point
        
        # Vehicle is at bottom center of image
        # Convert to vehicle frame: x is lateral, y is forward
        # Assuming image center is 640
        vehicle_x = 320  # Approximate center
        
        # Lateral error
        lateral_error = lx - vehicle_x
        
        # Pure Pursuit: steering = arctan(2 * wheelbase * lateral_error / lookahead^2)
        lookahead_pixels = path.lookahead_distance_m * 100  # Rough conversion
        
        if lookahead_pixels > 0:
            # Simplified steering calculation
            steering_rad = np.arctan2(
                2 * self.wheelbase_m * lateral_error,
                lookahead_pixels,
            )
            steering_deg = np.degrees(steering_rad)
        else:
            steering_deg = 0.0
        
        return steering_deg
    
    def _smooth_steering(self, steering_deg: float) -> float:
        """
        Apply exponential smoothing to steering command.
        
        Args:
            steering_deg: Raw steering angle
            
        Returns:
            Smoothed steering angle
        """
        smoothed = (
            self.steering_smoothing_alpha * steering_deg +
            (1 - self.steering_smoothing_alpha) * self.last_steering_deg
        )
        self.last_steering_deg = smoothed
        return smoothed
    
    def _calculate_target_speed(self, safety_status: SafetyStatus) -> float:
        """
        Calculate target speed based on confidence and safety.
        
        Args:
            safety_status: Current safety status
            
        Returns:
            Target speed in m/s
        """
        # Reduce speed if confidence is low
        if safety_status.fusion_confidence < 0.4:
            return self.slow_speed_mps
        elif safety_status.fusion_confidence < 0.6:
            return (self.max_speed_mps + self.slow_speed_mps) / 2
        else:
            return self.max_speed_mps
    
    @staticmethod
    def _speed_to_control(target_speed_mps: float) -> tuple[float, float]:
        """
        Convert target speed to throttle/brake commands.
        
        Args:
            target_speed_mps: Target speed
            
        Returns:
            (throttle, brake) values 0-1
        """
        if target_speed_mps > 0.5:
            throttle = min(1.0, target_speed_mps / 3.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = 0.5
        
        return throttle, brake
