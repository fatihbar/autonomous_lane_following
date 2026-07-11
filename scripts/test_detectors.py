"""
Test script for lane detection components.
"""

import argparse
import cv2
from pathlib import Path

from core.yolo_lane_detector import YOLOLaneDetector
from core.hough_lane_detector import HoughLaneDetector
from core.lane_mask_processor import LaneMaskProcessor
from core.logger import setup_logger

logger = setup_logger(__name__)


def test_yolo(model_path: str, image_path: str):
    """
    Test YOLO detector.
    """
    logger.info(f"Testing YOLO with {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Initialize detector
    detector = YOLOLaneDetector(model_path)
    
    # Detect
    result = detector.detect(image)
    logger.info(f"YOLO Result: valid={result.is_valid}, confidence={result.confidence:.2f}, mask_area={result.mask_area}")
    
    # Process mask
    if result.lane_mask is not None:
        processor = LaneMaskProcessor()
        geometry = processor.process(result.lane_mask, image.shape[:2])
        logger.info(f"Lane Geometry: source={geometry.source}, confidence={geometry.confidence:.2f}")


def test_hough(image_path: str):
    """
    Test Hough detector.
    """
    logger.info(f"Testing Hough with {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Initialize detector
    detector = HoughLaneDetector()
    
    # Detect
    result = detector.detect(image)
    logger.info(
        f"Hough Result: valid={result.is_valid}, confidence={result.confidence:.2f}, "
        f"left_points={len(result.left_line_points)}, right_points={len(result.right_line_points)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Test lane detection components")
    parser.add_argument(
        "--model",
        type=str,
        default="models/lane_segmentation.pt",
        help="Path to YOLO model",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Test image path",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["yolo", "hough", "all"],
        default="all",
        help="Which detector to test",
    )
    
    args = parser.parse_args()
    
    if args.test in ["yolo", "all"]:
        test_yolo(args.model, args.image)
    
    if args.test in ["hough", "all"]:
        test_hough(args.image)


if __name__ == "__main__":
    main()
