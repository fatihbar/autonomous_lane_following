"""
Visualization module for lane detection and system status.
"""

import cv2
import numpy as np
from typing import Optional, List

from ..core.data_types import LaneGeometry, SafetyStatus, ControlCommand, FusionResult
from ..core.logger import setup_logger
from ..utils.safety_utils import get_safety_color, get_mode_text

logger = setup_logger(__name__)


class LaneVisualizer:
    """
    Visualize lane detection results.
    """
    
    def __init__(
        self,
        font_scale: float = 0.5,
        line_thickness: int = 2,
        point_radius: int = 3,
    ):
        """
        Initialize visualizer.
        
        Args:
            font_scale: Font scale for text
            line_thickness: Line thickness in pixels
            point_radius: Radius for drawing points
        """
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw(
        self,
        image: np.ndarray,
        geometry: Optional[LaneGeometry] = None,
        fusion_result: Optional[FusionResult] = None,
        safety_status: Optional[SafetyStatus] = None,
        control_cmd: Optional[ControlCommand] = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw lane detection and system status on image.
        
        Args:
            image: Input image (will be modified)
            geometry: Lane geometry
            fusion_result: Fusion result
            safety_status: Safety status
            control_cmd: Control command
            fps: Current FPS
            
        Returns:
            Image with drawings
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]
        
        # Draw lane geometry
        if geometry is not None:
            canvas = self._draw_lane_geometry(canvas, geometry)
        
        # Draw status bars
        if safety_status is not None:
            canvas = self._draw_status_bars(canvas, safety_status)
        
        # Draw system status text
        if fusion_result is not None or safety_status is not None:
            canvas = self._draw_status_text(
                canvas,
                fusion_result,
                safety_status,
                control_cmd,
            )
        
        # Draw FPS
        if fps > 0:
            cv2.putText(
                canvas,
                f"FPS: {fps:.1f}",
                (w - 100, 20),
                self.font,
                self.font_scale,
                (0, 255, 0),
                self.line_thickness,
            )
        
        return canvas
    
    def _draw_lane_geometry(
        self,
        image: np.ndarray,
        geometry: LaneGeometry,
    ) -> np.ndarray:
        """
        Draw lane boundaries and centerline.
        """
        canvas = image.copy()
        
        # Draw left boundary
        if geometry.left_boundary is not None:
            points = np.array(
                [(int(p.x), int(p.y)) for p in geometry.left_boundary.points],
                dtype=np.int32,
            )
            if len(points) > 1:
                cv2.polylines(
                    canvas,
                    [points],
                    False,
                    (255, 0, 0),  # Blue
                    self.line_thickness,
                )
        
        # Draw right boundary
        if geometry.right_boundary is not None:
            points = np.array(
                [(int(p.x), int(p.y)) for p in geometry.right_boundary.points],
                dtype=np.int32,
            )
            if len(points) > 1:
                cv2.polylines(
                    canvas,
                    [points],
                    False,
                    (0, 0, 255),  # Red
                    self.line_thickness,
                )
        
        # Draw centerline
        if geometry.centerline is not None:
            points = np.array(
                [(int(p.x), int(p.y)) for p in geometry.centerline.points],
                dtype=np.int32,
            )
            if len(points) > 1:
                cv2.polylines(
                    canvas,
                    [points],
                    False,
                    (0, 255, 0),  # Green
                    self.line_thickness,
                )
        
        return canvas
    
    def _draw_status_bars(
        self,
        image: np.ndarray,
        safety_status: SafetyStatus,
    ) -> np.ndarray:
        """
        Draw confidence and safety status bars.
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]
        
        bar_width = 150
        bar_height = 20
        x_offset = 10
        y_offset = 30
        
        # Lane confidence bar
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + bar_width, y_offset + bar_height), (200, 200, 200), 1)
        filled_width = int(bar_width * safety_status.lane_confidence)
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + filled_width, y_offset + bar_height), (0, 255, 0), -1)
        cv2.putText(canvas, f"Lane: {safety_status.lane_confidence:.2f}", (x_offset + 5, y_offset + 15), self.font, 0.35, (0, 0, 0), 1)
        
        # Fusion confidence bar
        y_offset += 25
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + bar_width, y_offset + bar_height), (200, 200, 200), 1)
        filled_width = int(bar_width * safety_status.fusion_confidence)
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + filled_width, y_offset + bar_height), (0, 165, 255), -1)
        cv2.putText(canvas, f"Fusion: {safety_status.fusion_confidence:.2f}", (x_offset + 5, y_offset + 15), self.font, 0.35, (0, 0, 0), 1)
        
        return canvas
    
    def _draw_status_text(
        self,
        image: np.ndarray,
        fusion_result: Optional[FusionResult],
        safety_status: Optional[SafetyStatus],
        control_cmd: Optional[ControlCommand],
    ) -> np.ndarray:
        """
        Draw system status text.
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]
        
        y = h - 100
        
        if safety_status is not None:
            color = get_safety_color(safety_status)
            mode_text = get_mode_text(safety_status.system_mode)
            cv2.putText(canvas, f"Mode: {mode_text}", (10, y), self.font, self.font_scale, color, self.line_thickness)
            y += 25
            cv2.putText(canvas, f"Latency: {safety_status.inference_latency_ms:.1f}ms", (10, y), self.font, self.font_scale, (0, 255, 0), 1)
        
        if fusion_result is not None:
            y = h - 50
            source_text = f"Source: {fusion_result.lane_source.value}"
            cv2.putText(canvas, source_text, (10, y), self.font, self.font_scale, (255, 255, 255), self.line_thickness)
        
        if control_cmd is not None:
            y = h - 25
            steering_text = f"Steering: {control_cmd.steering_angle_deg:.1f}°"
            cv2.putText(canvas, steering_text, (10, y), self.font, self.font_scale, (255, 255, 255), 1)
        
        return canvas
