"""Singleton client for applying segmentation masks to camera frames."""

import logging
import os
import numpy as np
import cv2
from typing import Any
from numpy.typing import NDArray
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

logger = logging.getLogger(__name__)


class SegmentationClient:
    """
    Manages a single InferenceHTTPClient connection shared across all cameras
    to apply segmentation masks to frames. All cameras send requests to the same inference server.
    """
    
    _instance: "SegmentationClient | None" = None
    
    def __init__(self, api_url: str, model_id: str, confidence_threshold: float):
        """Initialize the segmentation client.
        
        Args:
            api_url: url of the inference server (e.g., "http://localhost:9001")
            model_id: roboflow model id (e.g., "switches-wymit/2")
            confidence_threshold: prediction confidence threshold (between 0 and 1)
        """
        self.api_url = api_url
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        
        load_dotenv()
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY environment variable is not set. "
                "Please set it in the .env file or export it with: export ROBOFLOW_API_KEY=your_key_here"
            )
        
        # Initialize client
        self.client = InferenceHTTPClient(
            api_url=api_url,
            api_key=api_key
        )
        
        # Configure inference settings
        config = InferenceConfiguration(
            confidence_threshold=confidence_threshold,
            iou_threshold=0.5
        )
        self.client.configure(config)
        self.client.select_model(model_id)
        
        logger.info(
            f"Initialized SegmentationClient: api_url={api_url}, "
            f"model_id={model_id}, confidence={confidence_threshold}"
        )
    
    @classmethod
    def get_instance(
        cls,
        api_url: str | None = None,
        model_id: str | None = None,
        confidence_threshold: float | None = None
    ) -> "SegmentationClient | None":
        """
        Args:
            api_url: url of the inference server (required on first call)
            model_id: roboflow model id (required on first call)
            confidence_threshold: prediction confidence threshold (between 0 and 1)
            
        Returns:
            The SegmentationClient instance or None
        """
        if cls._instance is None:
            if api_url is None or model_id is None or confidence_threshold is None:
                return None
            cls._instance = cls(api_url, model_id, confidence_threshold)
        return cls._instance
    
    def check_server_health(self) -> bool:
        """Check if the inference server is running and responsive.
        
        Returns:
            True if server is healthy, false otherwise
        """
        try:
            # Create a small test frame
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.client.infer(test_frame)
            logger.info(f"Segmentation server at {self.api_url} is healthy")
            return True
        except Exception as e:
            logger.error(f"Segmentation server health check failed: {e}")
            return False
    
    def apply_mask(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Apply segmentation mask to a frame.
            
        Returns: frame with segmentation mask applied
        """
        try:
            # Convert RGB to BGR color format: lerobot uses RGB internally but model expects BGR
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Run inference
            result = self.client.infer(bgr_frame)
            
            # Apply segmentation masks
            display_frame = frame.copy()
            if result and 'predictions' in result:
                for pred in result['predictions']:
                    if 'points' in pred:
                        points = np.array(
                            [[p['x'], p['y']] for p in pred['points']], 
                            dtype=np.int32
                        )
                        
                        # Polygon mask: green highlight/overlay
                        overlay = display_frame.copy()
                        cv2.fillPoly(overlay, [points], (0, 255, 0))  # Green mask
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                        
                        # Polygon outline: border around the mask
                        cv2.polylines(display_frame, [points], True, (0, 255, 0), 2)
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise RuntimeError(f"Failed to apply segmentation mask: {e}") from e
