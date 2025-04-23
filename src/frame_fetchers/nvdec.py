from typing import Any
from pathlib import Path

import torch
import cv2


from src.frame_fetchers.abstract import AbstractFrameFetcher


class OpenCVFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self._cap = cv2.VideoCapture(str(self.video_path))
        
        # Get video properties
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self._current_index = 0
        
        # Set up CUDA device if available and requested
        self.use_cuda = torch.cuda.is_available() and gpu_id >= 0
        self.device = torch.device(f'cuda:{gpu_id}' if self.use_cuda else 'cpu')

    def _next_decode(self) -> Any:
        """Decode the next frame from the video"""
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def _seek_and_decode(self, index: int) -> Any:
        """Seek to a specific frame index and decode it"""
        # OpenCV's frame indexing starts from 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self._next_decode()

    def _convert(self, frame: Any) -> torch.Tensor:
        """Convert the decoded frame to a grayscale PyTorch tensor"""
        if frame is None:
            return None
            
        # Convert to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to tensor and move to appropriate device
        frame_tensor = torch.from_numpy(grayscale_frame).to(self.device)
        
        return frame_tensor
    
    def __del__(self):
        """Clean up resources when the object is deleted"""
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()
