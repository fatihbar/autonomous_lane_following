"""
YOLO-based lane detection using ultralytics.
Supports both CUDA and CPU inference.
"""

import cv2
import numpy as np
from typing import Optional
from datetime import datetime
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .data_types import YOLODetectionResult, LaneMask
from .logger import setup_logger

logger = setup_logger(__name__)


class YOLOLaneDetector:
    """
    Lane detection using YOLO segmentation model.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
        use_half: bool = True,
    ):
        """
        Initialize YOLO lane detector.
        
        Args:
            model_path: Path to .pt model file
            confidence_threshold: Model confidence threshold (0-1)
            imgsz: Input image size
            device: "cuda", "cpu", or None for auto
            use_half: Use half precision (FP16) if available
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.use_half = use_half
        
        # Try to load model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
        
        # Determine device
        if device is None:
            # Auto-detect
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
    
    def detect(
        self,
        image: np.ndarray,
        roi_top_ratio: float = 0.0,
        roi_bottom_ratio: float = 1.0,
    ) -> YOLODetectionResult:
        """
        Detect lane in image.
        
        Args:
            image: Input BGR image (H, W, 3)
            roi_top_ratio: Top of ROI as fraction of height
            roi_bottom_ratio: Bottom of ROI as fraction of height
            
        Returns:
            YOLODetectionResult with lane mask and confidence
        """
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            if len(results) == 0:
                return YOLODetectionResult(
                    lane_mask=None,
                    is_valid=False,
                    confidence=0.0,
                    mask_area=0,
                    processing_time_ms=processing_time_ms,
                    device_used=self.device,
                )
            
            result = results[0]
            
            # Check if segmentation is available
            if result.masks is None:
                logger.warning("Model returned no segmentation masks")
                return YOLODetectionResult(
                    lane_mask=None,
                    is_valid=False,
                    confidence=0.0,
                    mask_area=0,
                    processing_time_ms=processing_time_ms,
                    device_used=self.device,
                )
            
            # Get masks and confidences
            masks = result.masks.data.cpu().numpy()
            confidences = result.conf.cpu().numpy() if result.conf is not None else np.ones(len(masks))
            
            if len(masks) == 0:
                return YOLODetectionResult(
                    lane_mask=None,
                    is_valid=False,
                    confidence=0.0,
                    mask_area=0,
                    processing_time_ms=processing_time_ms,
                    device_used=self.device,
                )
            
            # Combine all masks (multiple lane detections)
            combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            max_confidence = 0.0
            
            for mask, conf in zip(masks, confidences):
                # Resize mask to image size if needed
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8))
                max_confidence = max(max_confidence, float(conf))
            
            # Apply ROI
            h = image.shape[0]
            roi_top = int(h * roi_top_ratio)
            roi_bottom = int(h * roi_bottom_ratio)
            combined_mask[:roi_top] = 0
            combined_mask[roi_bottom:] = 0
            
            mask_area = int(np.sum(combined_mask))
            
            # Calculate confidence based on model confidence and mask area
            min_area = 100  # Minimum pixels for valid detection
            if mask_area < min_area:
                final_confidence = 0.0
            else:
                final_confidence = max_confidence
            
            lane_mask = LaneMask(
                mask=combined_mask,
                confidence=final_confidence,
                processing_time_ms=processing_time_ms,
            )
            
            return YOLODetectionResult(
                lane_mask=lane_mask,
                is_valid=final_confidence > 0.3,
                confidence=final_confidence,
                mask_area=mask_area,
                processing_time_ms=processing_time_ms,
                device_used=self.device,
            )
        
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return YOLODetectionResult(
                lane_mask=None,
                is_valid=False,
                confidence=0.0,
                mask_area=0,
                processing_time_ms=processing_time_ms,
                device_used=self.device,
            )
