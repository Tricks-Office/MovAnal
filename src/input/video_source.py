"""Abstract base class for video sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class VideoInfo:
    """Video metadata information."""

    width: int
    height: int
    fps: float
    frame_count: int  # -1 for live streams
    fourcc: str = ""

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return resolution as (width, height) tuple."""
        return (self.width, self.height)

    @property
    def is_live(self) -> bool:
        """Check if this is a live stream (unknown frame count)."""
        return self.frame_count < 0


class VideoSource(ABC):
    """Abstract base class for video input sources.

    This class defines the interface for all video input implementations,
    including file readers, camera inputs, and network streams.

    Usage:
        with FileVideoSource("video.mp4") as source:
            for frame in source:
                process(frame)
    """

    def __init__(self):
        self._is_open = False
        self._current_frame = 0

    @abstractmethod
    def open(self) -> bool:
        """Open the video source.

        Returns:
            True if successfully opened, False otherwise.
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the source.

        Returns:
            Tuple of (success, frame) where:
                - success: True if frame was read successfully
                - frame: numpy array of shape (H, W, C) in BGR format, or None if failed
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the video source and release resources."""
        pass

    @abstractmethod
    def get_info(self) -> VideoInfo:
        """Get video metadata information.

        Returns:
            VideoInfo object containing video properties.
        """
        pass

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame number.

        Args:
            frame_number: Target frame number (0-indexed).

        Returns:
            True if seek was successful, False otherwise.

        Note:
            Not all sources support seeking (e.g., live streams).
        """
        return False

    def is_open(self) -> bool:
        """Check if the source is currently open.

        Returns:
            True if the source is open and ready to read.
        """
        return self._is_open

    @property
    def current_frame(self) -> int:
        """Get the current frame position.

        Returns:
            Current frame number (0-indexed).
        """
        return self._current_frame

    def __enter__(self) -> "VideoSource":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames in the video source.

        Yields:
            Frames as numpy arrays in BGR format.
        """
        if not self._is_open:
            self.open()

        while True:
            success, frame = self.read()
            if not success or frame is None:
                break
            yield frame

    def __next__(self) -> np.ndarray:
        """Get the next frame.

        Returns:
            Next frame as numpy array.

        Raises:
            StopIteration: When no more frames are available.
        """
        success, frame = self.read()
        if not success or frame is None:
            raise StopIteration
        return frame
