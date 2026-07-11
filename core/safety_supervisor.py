"""
Safety supervisor - monitors system health and enforces safety constraints.
Highest priority module.
"""

import time
from typing import List, Optional
from datetime import datetime

from .data_types import (
    SafetyStatus,
    SystemMode,
    ControlCommand,
    FusionResult,
)
from .logger import setup_logger

logger = setup_logger(__name__)


class SafetySupervisor:
    """
    Monitor system safety and intervene if needed.
    """
    
    def __init__(
        self,
        min_lane_confidence: float = 0.3,
        min_fusion_confidence: float = 0.25,
        max_no_lane_time_s: float = 2.0,
        max_virtual_lane_time_s: float = 3.0,
        max_inference_latency_ms: float = 100.0,
        min_fps: float = 10.0,
        emergency_stop_distance_m: float = 0.5,
        dry_run: bool = True,
    ):
        """
        Initialize safety supervisor.
        
        Args:
            min_lane_confidence: Minimum lane confidence for autonomous
            min_fusion_confidence: Minimum fusion confidence
            max_no_lane_time_s: Max time without lane detection
            max_virtual_lane_time_s: Max time on virtual lane
            max_inference_latency_ms: Max inference latency
            min_fps: Minimum FPS
            emergency_stop_distance_m: Emergency stop LiDAR distance
            dry_run: If True, don't actually send commands to vehicle
        """
        self.min_lane_confidence = min_lane_confidence
        self.min_fusion_confidence = min_fusion_confidence
        self.max_no_lane_time_s = max_no_lane_time_s
        self.max_virtual_lane_time_s = max_virtual_lane_time_s
        self.max_inference_latency_ms = max_inference_latency_ms
        self.min_fps = min_fps
        self.emergency_stop_distance_m = emergency_stop_distance_m
        self.dry_run = dry_run
        
        # State tracking
        self.no_lane_start_time: Optional[datetime] = None
        self.system_mode = SystemMode.MANUAL
        self.warnings: List[str] = []
    
    def check_and_enforce(
        self,
        fusion_result: Optional[FusionResult],
        control_cmd: Optional[ControlCommand],
        current_fps: float,
        inference_latency_ms: float,
    ) -> tuple[SystemMode, Optional[ControlCommand], List[str]]:
        """
        Check safety constraints and potentially modify control command.
        
        Args:
            fusion_result: Fusion result
            control_cmd: Original control command
            current_fps: Current FPS
            inference_latency_ms: Inference latency
            
        Returns:
            (system_mode, modified_command, warnings)
        """
        self.warnings = []
        
        # Check basic health metrics
        if current_fps < self.min_fps:
            self.warnings.append(f"Low FPS: {current_fps:.1f} < {self.min_fps}")
        
        if inference_latency_ms > self.max_inference_latency_ms:
            self.warnings.append(
                f"High latency: {inference_latency_ms:.1f}ms > {self.max_inference_latency_ms}ms"
            )
        
        # Check fusion confidence
        if fusion_result is None:
            self.system_mode = SystemMode.EMERGENCY_STOP
            self.warnings.append("No fusion result available")
            return self.system_mode, self._emergency_stop_command(), self.warnings
        
        # Update no-lane timer
        if fusion_result.fusion_confidence < self.min_fusion_confidence:
            if self.no_lane_start_time is None:
                self.no_lane_start_time = datetime.now()
            
            no_lane_duration = (datetime.now() - self.no_lane_start_time).total_seconds()
            
            if no_lane_duration > self.max_no_lane_time_s:
                self.system_mode = SystemMode.EMERGENCY_STOP
                self.warnings.append(
                    f"No lane for {no_lane_duration:.1f}s > {self.max_no_lane_time_s}s"
                )
                return self.system_mode, self._emergency_stop_command(), self.warnings
        else:
            self.no_lane_start_time = None
        
        # Check virtual lane duration
        if (
            fusion_result.virtual_lane_active
            and fusion_result.virtual_lane_duration_s > self.max_virtual_lane_time_s
        ):
            self.system_mode = SystemMode.EMERGENCY_STOP
            self.warnings.append(
                f"Virtual lane for {fusion_result.virtual_lane_duration_s:.1f}s > {self.max_virtual_lane_time_s}s"
            )
            return self.system_mode, self._emergency_stop_command(), self.warnings
        
        # Determine operating mode
        if fusion_result.fusion_confidence >= self.min_fusion_confidence:
            if fusion_result.virtual_lane_active:
                self.system_mode = SystemMode.VIRTUAL_LANE
            else:
                self.system_mode = SystemMode.AUTONOMOUS
        else:
            self.system_mode = SystemMode.DEGRADED
        
        # Modify command if needed
        modified_cmd = control_cmd
        
        if self.system_mode == SystemMode.DEGRADED:
            # Reduce speed
            if modified_cmd is not None:
                modified_cmd.target_speed_mps *= 0.5
                modified_cmd.throttle_command *= 0.5
        
        # Apply dry-run
        if self.dry_run and modified_cmd is not None:
            modified_cmd.throttle_command = 0.0
            modified_cmd.brake_command = 0.0
        
        return self.system_mode, modified_cmd, self.warnings
    
    @staticmethod
    def _emergency_stop_command() -> ControlCommand:
        """
        Create emergency stop command.
        """
        return ControlCommand(
            steering_angle_deg=0.0,
            throttle_command=0.0,
            brake_command=1.0,
            target_speed_mps=0.0,
            confidence=1.0,
            is_emergency_stop=True,
        )
