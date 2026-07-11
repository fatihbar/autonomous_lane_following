"""
Video and camera input source adapter.
Supports video files and live camera feeds.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from ..core.logger import setup_logger

logger = setup_logger(__name__)


class VideoSource:
    """
    Read frames from video file or camera.
    """
    
    def __init__(self, source: str):
        """
        Initialize video source.
        
        Args:
            source: Path to video file or camera index (e.g., "0" for default camera)
        """
        self.source = source
        self.cap = None
        self.fps = 30
        self.width = 640
        self.height = 480
        self.frame_count = 0
        self.current_frame_idx = 0
        
        self._open_source()
    
    def _open_source(self):
        """
        Open video source.
        """
        try:
            # Try to open as camera index first
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
                logger.info(f"Opened camera {self.source}")
            else:
                # Try as video file
                if not Path(self.source).exists():
                    raise FileNotFoundError(f"Video file not found: {self.source}")
                
                self.cap = cv2.VideoCapture(self.source)
                logger.info(f"Opened video file: {self.source}")
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open source: {self.source}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(
                f"Video: {self.width}x{self.height} @ {self.fps}fps, {self.frame_count} frames"
            )
        
        except Exception as e:
            logger.error(f"Error opening source {self.source}: {e}")
            raise
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            (success, frame) - frame is BGR image or None if failed
        """
        if self.cap is None:
            return False, None
        
        success, frame = self.cap.read()
        
        if success:
            self.current_frame_idx += 1
        
        return success, frame
    
    def seek(self, frame_idx: int):
        """
        Seek to specific frame.
        
        Args:
            frame_idx: Frame index to seek to
        """
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame_idx = frame_idx
    
    def release(self):
        """
        Release video source.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()
