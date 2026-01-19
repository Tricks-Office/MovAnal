"""Unified feature extractor combining multiple feature extraction methods."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from .optical_flow import OpticalFlowExtractor, OpticalFlowResult
from .motion_history import MotionHistoryExtractor, MotionHistoryResult


@dataclass
class FeatureResult:
    """Combined result from all feature extractors."""

    optical_flow: Optional[OpticalFlowResult] = None
    motion_history: Optional[MotionHistoryResult] = None
    features: Dict[str, Any] = field(default_factory=dict)

    def get_feature_vector(self) -> np.ndarray:
        """Get concatenated feature vector for model input.

        Returns:
            1D numpy array of all numeric features.
        """
        features = []

        if self.optical_flow is not None:
            features.extend([
                self.optical_flow.mean_magnitude,
                self.optical_flow.max_magnitude,
                self.optical_flow.motion_energy,
            ])

        if self.motion_history is not None:
            features.extend([
                self.motion_history.motion_ratio,
                self.motion_history.recent_motion_ratio,
            ])

        for key, value in self.features.items():
            if isinstance(value, (int, float)):
                features.append(float(value))

        return np.array(features, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary with all feature values.
        """
        result = dict(self.features)

        if self.optical_flow is not None:
            result["of_mean_magnitude"] = self.optical_flow.mean_magnitude
            result["of_max_magnitude"] = self.optical_flow.max_magnitude
            result["of_motion_energy"] = self.optical_flow.motion_energy

        if self.motion_history is not None:
            result["mh_motion_ratio"] = self.motion_history.motion_ratio
            result["mh_recent_motion_ratio"] = self.motion_history.recent_motion_ratio

        return result


class FeatureExtractor:
    """Unified feature extractor combining multiple extraction methods.

    Manages multiple feature extraction methods and combines their outputs
    into a unified result.

    Args:
        enable_optical_flow: Enable optical flow extraction.
        enable_motion_history: Enable motion history extraction.
        optical_flow_config: Configuration for optical flow extractor.
        motion_history_config: Configuration for motion history extractor.

    Example:
        >>> extractor = FeatureExtractor(
        ...     enable_optical_flow=True,
        ...     enable_motion_history=True
        ... )
        >>> result = extractor.extract(frame)
        >>> features = result.get_feature_vector()
        >>> print(f"Feature vector shape: {features.shape}")
    """

    def __init__(
        self,
        enable_optical_flow: bool = True,
        enable_motion_history: bool = True,
        optical_flow_config: Optional[Dict] = None,
        motion_history_config: Optional[Dict] = None,
    ):
        self._enable_optical_flow = enable_optical_flow
        self._enable_motion_history = enable_motion_history

        # Initialize optical flow extractor
        self._optical_flow: Optional[OpticalFlowExtractor] = None
        if enable_optical_flow:
            of_config = optical_flow_config or {}
            self._optical_flow = OpticalFlowExtractor(
                pyr_scale=of_config.get("pyr_scale", 0.5),
                levels=of_config.get("levels", 3),
                winsize=of_config.get("winsize", 15),
                iterations=of_config.get("iterations", 3),
                poly_n=of_config.get("poly_n", 5),
                poly_sigma=of_config.get("poly_sigma", 1.2),
            )

        # Initialize motion history extractor
        self._motion_history: Optional[MotionHistoryExtractor] = None
        if enable_motion_history:
            mh_config = motion_history_config or {}
            self._motion_history = MotionHistoryExtractor(
                duration=mh_config.get("duration", 0.5),
                threshold=mh_config.get("threshold", 32),
                fps=mh_config.get("fps", 30.0),
            )

        self._frame_count = 0

    def extract(self, frame: np.ndarray) -> FeatureResult:
        """Extract features from a single frame.

        Uses incremental processing for temporal features.

        Args:
            frame: Input frame (BGR format).

        Returns:
            FeatureResult with all extracted features.
        """
        result = FeatureResult()

        # Extract optical flow
        if self._optical_flow is not None:
            of_result = self._optical_flow.compute_incremental(frame)
            if of_result is not None:
                result.optical_flow = of_result
                stats = self._optical_flow.get_motion_statistics(of_result)
                result.features.update({f"of_{k}": v for k, v in stats.items()})

        # Extract motion history
        if self._motion_history is not None:
            mh_result = self._motion_history.update(frame)
            if mh_result is not None:
                result.motion_history = mh_result
                stats = self._motion_history.get_statistics(mh_result)
                result.features.update({f"mh_{k}": v for k, v in stats.items()})

        self._frame_count += 1
        result.features["frame_number"] = self._frame_count

        return result

    def extract_batch(self, frames: List[np.ndarray]) -> List[FeatureResult]:
        """Extract features from a batch of frames.

        Args:
            frames: List of input frames.

        Returns:
            List of FeatureResult objects.
        """
        return [self.extract(frame) for frame in frames]

    def extract_with_prev(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> FeatureResult:
        """Extract features using explicit previous and current frames.

        Unlike extract(), this doesn't rely on internal state.

        Args:
            prev_frame: Previous frame.
            curr_frame: Current frame.

        Returns:
            FeatureResult with extracted features.
        """
        result = FeatureResult()

        # Extract optical flow
        if self._optical_flow is not None:
            of_result = self._optical_flow.compute(prev_frame, curr_frame)
            result.optical_flow = of_result
            stats = self._optical_flow.get_motion_statistics(of_result)
            result.features.update({f"of_{k}": v for k, v in stats.items()})

        return result

    def reset(self) -> None:
        """Reset all internal states for fresh processing."""
        if self._optical_flow is not None:
            self._optical_flow.reset()
        if self._motion_history is not None:
            self._motion_history.reset()
        self._frame_count = 0

    def get_feature_names(self) -> List[str]:
        """Get names of all features produced by this extractor.

        Returns:
            List of feature names.
        """
        names = []

        if self._enable_optical_flow:
            names.extend([
                "of_mean_magnitude",
                "of_max_magnitude",
                "of_std_magnitude",
                "of_motion_area_ratio",
                "of_motion_energy",
                "of_dominant_direction",
            ])

        if self._enable_motion_history:
            names.extend([
                "mh_motion_ratio",
                "mh_mean_recency",
                "mh_motion_area",
                "mh_max_recency",
            ])

        names.append("frame_number")

        return names

    def get_visualization(self, result: FeatureResult) -> Dict[str, np.ndarray]:
        """Get visualizations of extracted features.

        Args:
            result: FeatureResult from extract().

        Returns:
            Dictionary mapping visualization names to images.
        """
        visualizations = {}

        if result.optical_flow is not None and result.optical_flow.visualization is not None:
            visualizations["optical_flow"] = result.optical_flow.visualization

        if result.motion_history is not None and self._motion_history is not None:
            visualizations["motion_history"] = self._motion_history.visualize(
                result.motion_history
            )

        return visualizations

    @classmethod
    def from_config(cls, config: Dict) -> "FeatureExtractor":
        """Create extractor from configuration dictionary.

        Args:
            config: Full configuration dictionary.

        Returns:
            Configured FeatureExtractor.
        """
        features_config = config.get("features", {})
        video_config = config.get("video", {})

        # Merge video fps into motion history config
        mh_config = features_config.get("motion_history", {}).copy()
        if "fps" not in mh_config:
            mh_config["fps"] = video_config.get("fps", 30.0)

        return cls(
            enable_optical_flow="optical_flow" in features_config,
            enable_motion_history="motion_history" in features_config,
            optical_flow_config=features_config.get("optical_flow"),
            motion_history_config=mh_config,
        )

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor(optical_flow={self._enable_optical_flow}, "
            f"motion_history={self._enable_motion_history})"
        )
