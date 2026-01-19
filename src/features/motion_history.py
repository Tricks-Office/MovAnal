"""Motion history extraction for temporal motion analysis."""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionHistoryResult:
    """Result container for motion history computation."""

    mhi: np.ndarray  # Motion History Image
    mei: np.ndarray  # Motion Energy Image (binary)
    silhouette: np.ndarray  # Current motion silhouette

    @property
    def motion_ratio(self) -> float:
        """Get ratio of frame with motion."""
        return float(np.mean(self.mei > 0))

    @property
    def recent_motion_ratio(self) -> float:
        """Get ratio of frame with recent motion (from MHI)."""
        return float(np.mean(self.mhi > 0))


class MotionHistoryExtractor:
    """Extract motion history features from video frames.

    Computes Motion History Image (MHI) and Motion Energy Image (MEI)
    for temporal motion analysis.

    MHI encodes how recently motion occurred at each pixel,
    MEI is a binary image showing where motion has occurred.

    Args:
        duration: Time duration for motion history in seconds.
        threshold: Pixel difference threshold for motion detection.
        fps: Frame rate for timestamp calculation.

    Example:
        >>> extractor = MotionHistoryExtractor(duration=1.0, threshold=32)
        >>> result = extractor.update(frame)
        >>> if result:
        ...     cv2.imshow("MHI", (result.mhi * 255).astype(np.uint8))
    """

    def __init__(
        self,
        duration: float = 0.5,
        threshold: int = 32,
        fps: float = 30.0,
    ):
        self._duration = duration
        self._threshold = threshold
        self._fps = fps

        self._prev_gray: Optional[np.ndarray] = None
        self._mhi: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._frame_count: int = 0

    @property
    def duration(self) -> float:
        """Get the motion history duration."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the motion history duration."""
        self._duration = value

    @property
    def threshold(self) -> int:
        """Get the motion detection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: int) -> None:
        """Set the motion detection threshold."""
        self._threshold = value

    def update(self, frame: np.ndarray) -> Optional[MotionHistoryResult]:
        """Update motion history with a new frame.

        Args:
            frame: Input frame (BGR or grayscale).

        Returns:
            MotionHistoryResult, or None if this is the first frame.
        """
        # Convert to grayscale
        gray = self._to_gray(frame)

        # Initialize on first frame
        if self._prev_gray is None:
            self._prev_gray = gray
            self._mhi = np.zeros(gray.shape, dtype=np.float32)
            return None

        # Calculate frame difference
        diff = cv2.absdiff(self._prev_gray, gray)

        # Create motion silhouette (binary)
        _, silhouette = cv2.threshold(
            diff, self._threshold, 1, cv2.THRESH_BINARY
        )

        # Update timestamp
        self._frame_count += 1
        self._timestamp = self._frame_count / self._fps

        # Update motion history image
        cv2.motempl.updateMotionHistory(
            silhouette,
            self._mhi,
            self._timestamp,
            self._duration,
        )

        # Normalize MHI for output
        mhi_normalized = self._mhi.copy()
        if self._timestamp > 0:
            # Scale to 0-1 range based on duration
            min_timestamp = max(0, self._timestamp - self._duration)
            mask = mhi_normalized > 0
            mhi_normalized[mask] = (
                (mhi_normalized[mask] - min_timestamp) / self._duration
            )
            mhi_normalized = np.clip(mhi_normalized, 0, 1)

        # Create Motion Energy Image (binary)
        mei = (self._mhi > 0).astype(np.uint8)

        # Update previous frame
        self._prev_gray = gray

        return MotionHistoryResult(
            mhi=mhi_normalized,
            mei=mei,
            silhouette=silhouette,
        )

    def compute_gradient(
        self,
        result: MotionHistoryResult,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Compute global motion orientation from MHI gradient.

        Args:
            result: MotionHistoryResult from update().
            mask: Optional mask to limit computation area.

        Returns:
            Tuple of (orientation_angle, magnitude) in degrees.
        """
        if mask is None:
            mask = result.mei

        # Calculate orientation
        orientation, _ = cv2.motempl.calcGlobalOrientation(
            result.mhi * self._duration,  # Un-normalize for computation
            mask,
            self._mhi,
            self._timestamp,
            self._duration,
        )

        # Estimate magnitude from MHI values
        magnitude = float(np.mean(result.mhi[mask > 0])) if np.any(mask > 0) else 0.0

        return orientation, magnitude

    def segment_motion(
        self,
        result: MotionHistoryResult,
        min_area: int = 100,
    ) -> list:
        """Segment motion regions in the MHI.

        Args:
            result: MotionHistoryResult from update().
            min_area: Minimum area for a motion region.

        Returns:
            List of (bounding_rect, orientation, area) tuples.
        """
        # Find contours in MEI
        contours, _ = cv2.findContours(
            result.mei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get bounding rectangle
            rect = cv2.boundingRect(contour)

            # Create mask for this region
            mask = np.zeros_like(result.mei)
            cv2.drawContours(mask, [contour], -1, 1, -1)

            # Calculate orientation for this region
            orientation, _ = self.compute_gradient(result, mask)

            regions.append((rect, orientation, area))

        return regions

    def get_statistics(self, result: MotionHistoryResult) -> Dict[str, float]:
        """Get motion statistics from the result.

        Args:
            result: MotionHistoryResult from update().

        Returns:
            Dictionary with motion statistics.
        """
        mhi = result.mhi
        mei = result.mei

        # Calculate basic statistics
        motion_area = np.sum(mei > 0)
        total_area = mei.size

        return {
            "motion_ratio": motion_area / total_area,
            "mean_recency": float(np.mean(mhi[mei > 0])) if motion_area > 0 else 0.0,
            "motion_area": motion_area,
            "max_recency": float(np.max(mhi)) if motion_area > 0 else 0.0,
        }

    def visualize(
        self,
        result: MotionHistoryResult,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Create colored visualization of motion history.

        Args:
            result: MotionHistoryResult from update().
            colormap: OpenCV colormap to use.

        Returns:
            BGR visualization image.
        """
        # Convert MHI to uint8
        mhi_vis = (result.mhi * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(mhi_vis, colormap)

        # Make background black where no motion
        colored[result.mei == 0] = [0, 0, 0]

        return colored

    def reset(self) -> None:
        """Reset the motion history state."""
        self._prev_gray = None
        self._mhi = None
        self._timestamp = 0.0
        self._frame_count = 0

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if needed."""
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @classmethod
    def from_config(cls, config: Dict) -> "MotionHistoryExtractor":
        """Create extractor from configuration dictionary.

        Args:
            config: Feature extraction configuration.

        Returns:
            Configured MotionHistoryExtractor.
        """
        mh_config = config.get("motion_history", {})
        video_config = config.get("video", {})

        return cls(
            duration=mh_config.get("duration", 0.5),
            threshold=mh_config.get("threshold", 32),
            fps=video_config.get("fps", 30.0),
        )

    def __repr__(self) -> str:
        return (
            f"MotionHistoryExtractor(duration={self._duration}, "
            f"threshold={self._threshold})"
        )
