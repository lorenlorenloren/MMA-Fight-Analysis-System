"""Cage/octagon detection and normalization."""

import cv2
import numpy as np
from typing import Tuple
from loguru import logger


class CageDetector:
    """Detect UFC octagon and compute homography for spatial normalization."""
    
    def __init__(self, cage_diameter: float = 9.1):
        """Initialize cage detector.
        
        Args:
            cage_diameter: UFC octagon diameter in meters (default 9.1m)
        """
        self.cage_diameter = cage_diameter
        self.radius = cage_diameter / 2
        logger.info(f"CageDetector initialized with {cage_diameter}m diameter")
    
    def detect_octagon(self, frame: np.ndarray, use_morphology: bool = True) -> np.ndarray:
        """Detect octagon corners in frame.
        
        Args:
            frame: Input frame (RGB or BGR)
            use_morphology: Apply morphological operations
            
        Returns:
            Array of 8 corner points (N, 2) or None if detection fails
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            if use_morphology:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Hough lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
            
            if lines is None:
                logger.warning("No lines detected for octagon")
                return None
            
            logger.info(f"Detected {len(lines)} line segments")
            return self._fit_octagon(lines, frame.shape)
        except Exception as e:
            logger.error(f"Cage detection error: {e}")
            return None
    
    def _fit_octagon(self, lines, frame_shape: Tuple) -> np.ndarray:
        """Fit octagon model to detected lines."""
        # Placeholder implementation
        return np.array([[100, 100], [300, 50], [500, 100], [550, 300],
                        [500, 500], [300, 550], [100, 500], [50, 300]], dtype=np.float32)
    
    def compute_homography(self, octagon_corners: np.ndarray) -> np.ndarray:
        """Compute homography from image plane to world plane.
        
        Args:
            octagon_corners: 8 corner points of detected octagon
            
        Returns:
            3x3 homography matrix
        """
        if octagon_corners is None:
            logger.warning("Cannot compute homography without octagon corners")
            return None
        
        # Canonical octagon in world coordinates (meters)
        angles = np.linspace(0, 2*np.pi, 9)[:-1]
        world_corners = np.array([
            [self.radius * np.cos(a), self.radius * np.sin(a)] for a in angles
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(octagon_corners, world_corners)
        logger.info("Homography computed successfully")
        return H
    
    def transform_point(self, point: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """Transform point from image to world coordinates.
        
        Args:
            point: Point in image coordinates (x, y)
            homography: Homography matrix
            
        Returns:
            Point in world coordinates
        """
        if homography is None:
            return point
        
        point_h = np.array([point[0], point[1], 1.0])
        world_point_h = homography @ point_h
        world_point = world_point_h[:2] / world_point_h[2]
        return world_point
