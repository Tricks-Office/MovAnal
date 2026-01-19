"""Optical flow extraction for motion analysis."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class OpticalFlowResult:
    """Result container for optical flow computation."""

    flow: np.ndarray  # (H, W, 2) flow vectors (dx, dy)
    magnitude: np.ndarray  # (H, W) flow magnitude
    angle: np.ndarray  # (H, W) flow direction in radians
    visualization: Optional[np.ndarray] = None  # (H, W, 3) HSV visualization

    @property
    def mean_magnitude(self) -> float:
        """Get mean flow magnitude."""
        return float(np.mean(self.magnitude))

    @property
    def max_magnitude(self) -> float:
        """Get maximum flow magnitude."""
        return float(np.max(self.magnitude))

    @property
    def motion_energy(self) -> float:
        """Get total motion energy (sum of squared magnitudes)."""
        return float(np.sum(self.magnitude ** 2))


class OpticalFlowExtractor:
    """Extract optical flow features from video frames.

    Uses Farneback algorithm for dense optical flow computation.

    Args:
        pyr_scale: Pyramid scale factor.
        levels: Number of pyramid levels.
        winsize: Averaging window size.
        iterations: Number of iterations at each pyramid level.
        poly_n: Size of pixel neighborhood for polynomial expansion.
        poly_sigma: Standard deviation for polynomial expansion.
        visualize: If True, generate HSV visualization.

    Example:
        >>> extractor = OpticalFlowExtractor()
        >>> result = extractor.compute(prev_frame, curr_frame)
        >>> print(f"Mean motion: {result.mean_magnitude:.2f}")
        >>> cv2.imshow("Flow", result.visualization)
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        visualize: bool = True,
    ):
        self._pyr_scale = pyr_scale
        self._levels = levels
        self._winsize = winsize
        self._iterations = iterations
        self._poly_n = poly_n
        self._poly_sigma = poly_sigma
        self._visualize = visualize

        self._prev_gray: Optional[np.ndarray] = None

    def compute(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> OpticalFlowResult:
        """Compute optical flow between two frames.

        Args:
            prev_frame: Previous frame (BGR or grayscale).
            curr_frame: Current frame (BGR or grayscale).

        Returns:
            OpticalFlowResult with flow vectors and analysis.
        """
        # Convert to grayscale if needed
        prev_gray = self._to_gray(prev_frame)
        curr_gray = self._to_gray(curr_frame)

        # Compute dense optical flow using Farneback algorithm
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=self._pyr_scale,
            levels=self._levels,
            winsize=self._winsize,
            iterations=self._iterations,
            poly_n=self._poly_n,
            poly_sigma=self._poly_sigma,
            flags=0,
        )

        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Generate visualization if requested
        visualization = None
        if self._visualize:
            visualization = self._visualize_flow(magnitude, angle)

        return OpticalFlowResult(
            flow=flow,
            magnitude=magnitude,
            angle=angle,
            visualization=visualization,
        )

    def compute_incremental(self, frame: np.ndarray) -> Optional[OpticalFlowResult]:
        """Compute optical flow incrementally from the previous frame.

        Maintains internal state of the previous frame for continuous
        processing of video streams.

        Args:
            frame: Current frame (BGR or grayscale).

        Returns:
            OpticalFlowResult, or None if this is the first frame.
        """
        curr_gray = self._to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = curr_gray
            return None

        result = self.compute(self._prev_gray, curr_gray)
        self._prev_gray = curr_gray

        return result

    def reset(self) -> None:
        """Reset the previous frame state for incremental processing."""
        self._prev_gray = None

    def compute_sparse(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        points: Optional[np.ndarray] = None,
        max_corners: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute sparse optical flow using Lucas-Kanade method.

        Args:
            prev_frame: Previous frame.
            curr_frame: Current frame.
            points: Optional feature points to track. If None, detect corners.
            max_corners: Maximum number of corners to detect.

        Returns:
            Tuple of (prev_points, curr_points, status) where status indicates
            which points were successfully tracked.
        """
        prev_gray = self._to_gray(prev_frame)
        curr_gray = self._to_gray(curr_frame)

        # Detect corners if no points provided
        if points is None:
            points = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=max_corners,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7,
            )

        if points is None or len(points) == 0:
            return np.array([]), np.array([]), np.array([])

        # Calculate optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None
        )

        return points, curr_points, status.ravel()

    def get_motion_regions(
        self,
        result: OpticalFlowResult,
        threshold: float = 2.0,
    ) -> np.ndarray:
        """Get binary mask of regions with significant motion.

        Args:
            result: OpticalFlowResult from compute().
            threshold: Minimum magnitude to consider as motion.

        Returns:
            Binary mask where 1 indicates motion.
        """
        return (result.magnitude > threshold).astype(np.uint8) * 255

    def get_motion_statistics(self, result: OpticalFlowResult) -> Dict[str, float]:
        """Get statistical summary of motion in the frame.

        Args:
            result: OpticalFlowResult from compute().

        Returns:
            Dictionary with motion statistics.
        """
        mag = result.magnitude

        # Calculate dominant direction
        mean_angle = np.arctan2(
            np.mean(np.sin(result.angle) * mag),
            np.mean(np.cos(result.angle) * mag),
        )

        return {
            "mean_magnitude": float(np.mean(mag)),
            "max_magnitude": float(np.max(mag)),
            "std_magnitude": float(np.std(mag)),
            "motion_area_ratio": float(np.mean(mag > 2.0)),
            "motion_energy": float(np.sum(mag ** 2)),
            "dominant_direction": float(np.degrees(mean_angle)),
        }

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if needed."""
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _visualize_flow(
        self,
        magnitude: np.ndarray,
        angle: np.ndarray,
    ) -> np.ndarray:
        """Create HSV visualization of optical flow.

        Hue represents direction, saturation is constant,
        value represents magnitude.

        Args:
            magnitude: Flow magnitude array.
            angle: Flow angle array (radians).

        Returns:
            BGR visualization image.
        """
        hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)

        # Hue: direction (0-180 for OpenCV)
        hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)

        # Saturation: constant
        hsv[..., 1] = 255

        # Value: normalized magnitude
        max_mag = np.max(magnitude) + 1e-5
        hsv[..., 2] = np.clip(magnitude / max_mag * 255, 0, 255).astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @classmethod
    def from_config(cls, config: Dict) -> "OpticalFlowExtractor":
        """Create extractor from configuration dictionary.

        Args:
            config: Feature extraction configuration.

        Returns:
            Configured OpticalFlowExtractor.
        """
        flow_config = config.get("optical_flow", {})
        return cls(
            pyr_scale=flow_config.get("pyr_scale", 0.5),
            levels=flow_config.get("levels", 3),
            winsize=flow_config.get("winsize", 15),
            iterations=flow_config.get("iterations", 3),
            poly_n=flow_config.get("poly_n", 5),
            poly_sigma=flow_config.get("poly_sigma", 1.2),
        )

    def __repr__(self) -> str:
        return (
            f"OpticalFlowExtractor(levels={self._levels}, "
            f"winsize={self._winsize}, iterations={self._iterations})"
        )
