"""Video loading and preprocessing."""

import cv2
import numpy as np
from typing import Generator, Tuple
from pathlib import Path
from loguru import logger


class VideoProcessor:
    """Handles video frame extraction and preprocessing."""
    
    def __init__(self, target_fps: int = 30, target_resolution: Tuple[int, int] = (1920, 1080)):
        """Initialize video processor.
        
        Args:
            target_fps: Target frames per second for output
            target_resolution: (width, height) tuple for resize
        """
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        logger.info(f"VideoProcessor initialized: {target_fps} FPS, {target_resolution} resolution")
    
    def process_video(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """Process video file frame-by-frame.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Preprocessed frames (RGB format)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(source_fps / self.target_fps) if source_fps > 0 else 1
        
        logger.info(f"Processing video: {video_path.name} ({source_fps} FPS) -> {self.target_fps} FPS")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip == 0:
                # Resize and convert BGR to RGB
                frame = cv2.resize(frame, self.target_resolution)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Video processing complete. Total frames yielded: {frame_idx // frame_skip}")
    
    def save_video(self, frames: list, output_path: str, fps: int = 30):
        """Save frames to video file.
        
        Args:
            frames: List of frames (numpy arrays)
            output_path: Output video file path
            fps: Frames per second for output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not frames:
            logger.warning("No frames to save")
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in frames:
            # Convert RGB back to BGR for writing
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        logger.info(f"Video saved: {output_path}")
