"""Feature extraction module for motion analysis."""

from .optical_flow import OpticalFlowExtractor
from .motion_history import MotionHistoryExtractor
from .extractor import FeatureExtractor

__all__ = ["OpticalFlowExtractor", "MotionHistoryExtractor", "FeatureExtractor"]
