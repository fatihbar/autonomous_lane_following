"""
Main standalone application for testing lane detection.
"""

import argparse
import time
import yaml
import cv2
from pathlib import Path

from core.yolo_lane_detector import YOLOLaneDetector
from core.hough_lane_detector import HoughLaneDetector
from core.lane_mask_processor import LaneMaskProcessor
from core.lane_geometry import LaneGeometryBuilder
from core.virtual_lane_generator import VirtualLaneGenerator
from core.sensor_fusion import SensorFusion
from core.path_planner import PathPlanner
from core.controller import VehicleController
from core.safety_supervisor import SafetySupervisor
from core.ai_diagnostics import AIDiagnostics
from core.data_types import SafetyStatus, SystemMode, SystemDiagnostics
from adapters.video_source import VideoSource
from visualization.lane_visualizer import LaneVisualizer
from core.logger import setup_logger

logger = setup_logger(__name__)


class AutonomousLaneFollower:
    """
    Main autonomous lane following system.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize system from config files.
        
        Args:
            config_dir: Configuration directory
        """
        self.config_dir = Path(config_dir)
        
        # Load configs
        self.perception_cfg = self._load_config("perception.yaml")
        self.hough_cfg = self._load_config("hough_fallback.yaml")
        self.fusion_cfg = self._load_config("fusion.yaml")
        self.safety_cfg = self._load_config("safety.yaml")
        self.controller_cfg = self._load_config("controller.yaml")
        self.viz_cfg = self._load_config("visualization.yaml")
        
        # Initialize components
        self.yolo_detector = None
        self.hough_detector = None
        self.mask_processor = None
        self.geometry_builder = None
        self.virtual_lane_gen = None
        self.sensor_fusion = None
        self.path_planner = None
        self.controller = None
        self.safety_supervisor = None
        self.visualizer = None
        
        # State
        self.frame_count = 0
        self.fps = 0.0
        self.frame_times = []
    
    def _load_config(self, filename: str) -> dict:
        """
        Load YAML config file.
        """
        config_path = self.config_dir / filename
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            return {}
    
    def initialize(self, model_path: str):
        """
        Initialize all components with model.
        
        Args:
            model_path: Path to YOLO model
        """
        logger.info("Initializing autonomous lane follower...")
        
        # YOLO
        try:
            self.yolo_detector = YOLOLaneDetector(
                model_path=model_path,
                confidence_threshold=self.perception_cfg.get("model", {}).get("confidence_threshold", 0.5),
                imgsz=self.perception_cfg.get("model", {}).get("imgsz", 640),
            )
            logger.info("YOLO detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            raise
        
        # Hough fallback
        self.hough_detector = HoughLaneDetector(
            canny_low=self.hough_cfg.get("canny", {}).get("low_threshold", 50),
            canny_high=self.hough_cfg.get("canny", {}).get("high_threshold", 150),
        )
        logger.info("Hough detector initialized")
        
        # Processors and builders
        self.mask_processor = LaneMaskProcessor()
        self.geometry_builder = LaneGeometryBuilder()
        self.virtual_lane_gen = VirtualLaneGenerator()
        self.sensor_fusion = SensorFusion()
        self.path_planner = PathPlanner()
        self.controller = VehicleController()
        self.safety_supervisor = SafetySupervisor(
            dry_run=self.safety_cfg.get("vehicle_control", {}).get("dry_run", True),
        )
        self.visualizer = LaneVisualizer()
        
        logger.info("All components initialized")
    
    def process_frame(self, image) -> dict:
        """
        Process a single frame.
        
        Args:
            image: Input BGR image
            
        Returns:
            Result dictionary with all detections and commands
        """
        frame_start = time.time()
        h, w = image.shape[:2]
        
        # YOLO detection
        yolo_result = self.yolo_detector.detect(image)
        yolo_geometry = None
        if yolo_result.lane_mask is not None:
            yolo_geometry = self.mask_processor.process(yolo_result.lane_mask, (h, w))
        
        # Hough fallback
        hough_result = self.hough_detector.detect(image)
        hough_geometry = self.geometry_builder.from_hough_result(
            hough_result, h, w
        )
        
        # Virtual lane
        virtual_geometry, virtual_duration, virtual_active = self.virtual_lane_gen.update(
            yolo_geometry if yolo_geometry is not None else hough_geometry
        )
        
        # Fusion
        fusion_result = self.sensor_fusion.fuse(
            yolo_geometry=yolo_geometry,
            hough_geometry=hough_geometry,
            virtual_geometry=virtual_geometry,
            lidar_result=None,
            virtual_duration_s=virtual_duration,
        )
        
        # Planning
        path = self.path_planner.plan(fusion_result.final_lane, h)
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Safety status
        safety_status = SafetyStatus(
            system_mode=SystemMode.AUTONOMOUS,
            emergency_stop_active=False,
            lane_confidence=fusion_result.final_lane.confidence if fusion_result.final_lane else 0.0,
            fusion_confidence=fusion_result.fusion_confidence,
            inference_latency_ms=frame_time * 1000,
            current_fps=self.fps,
            virtual_lane_duration_s=virtual_duration,
            no_lane_duration_s=0.0,
        )
        
        # Control
        control_cmd = self.controller.control(path, safety_status)
        
        # Safety check
        final_mode, final_cmd, warnings = self.safety_supervisor.check_and_enforce(
            fusion_result,
            control_cmd,
            self.fps,
            frame_time * 1000,
        )
        
        safety_status.system_mode = final_mode
        if warnings:
            safety_status.last_warning = warnings[0]
        
        self.frame_count += 1
        
        return {
            "yolo_geometry": yolo_geometry,
            "hough_geometry": hough_geometry,
            "virtual_geometry": virtual_geometry,
            "fusion_result": fusion_result,
            "path": path,
            "control_cmd": final_cmd,
            "safety_status": safety_status,
            "fps": self.fps,
            "frame_time": frame_time,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Lane Following System"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/lane_segmentation.pt",
        help="Path to YOLO model",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Video file path or camera index (default: 0 for webcam)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Configuration directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_videos/output.mp4",
        help="Output video file path",
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = AutonomousLaneFollower(config_dir=args.config)
    system.initialize(args.model)
    
    # Open video source
    video_source = VideoSource(args.video)
    
    # Output video writer
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        video_source.fps,
        (video_source.width, video_source.height),
    )
    
    logger.info("Starting processing...")
    
    try:
        while True:
            success, frame = video_source.read()
            if not success:
                logger.info("Video ended")
                break
            
            # Process frame
            result = system.process_frame(frame)
            
            # Visualize
            viz_frame = system.visualizer.draw(
                frame,
                geometry=result["fusion_result"].final_lane,
                fusion_result=result["fusion_result"],
                safety_status=result["safety_status"],
                control_cmd=result["control_cmd"],
                fps=result["fps"],
            )
            
            # Write output
            out.write(viz_frame)
            
            # Display
            cv2.imshow("Lane Following", viz_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    finally:
        video_source.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    main()
