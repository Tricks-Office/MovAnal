"""Thread-safe frame buffer for video processing."""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class TimestampedFrame:
    """Frame with associated timestamp and metadata."""

    frame: np.ndarray
    timestamp: float  # Unix timestamp in seconds
    frame_number: int

    @property
    def age(self) -> float:
        """Get the age of this frame in seconds."""
        return time.time() - self.timestamp


class FrameBuffer:
    """Thread-safe circular buffer for video frames.

    Maintains a fixed-size buffer of the most recent frames,
    useful for temporal analysis and motion detection.

    Args:
        max_size: Maximum number of frames to store.

    Example:
        >>> buffer = FrameBuffer(max_size=30)
        >>> buffer.push(frame1, frame_number=0)
        >>> buffer.push(frame2, frame_number=1)
        >>> recent_frames = buffer.get_recent(5)
        >>> latest = buffer.get_latest()
    """

    def __init__(self, max_size: int = 30):
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._max_size = max_size
        self._buffer: Deque[TimestampedFrame] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._frame_count = 0

    @property
    def max_size(self) -> int:
        """Get the maximum buffer size."""
        return self._max_size

    @property
    def size(self) -> int:
        """Get the current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        with self._lock:
            return len(self._buffer) >= self._max_size

    def push(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add a frame to the buffer.

        If the buffer is full, the oldest frame is removed.

        Args:
            frame: Frame to add (numpy array).
            frame_number: Optional frame number. Auto-increments if not provided.
            timestamp: Optional timestamp. Uses current time if not provided.
        """
        if timestamp is None:
            timestamp = time.time()

        if frame_number is None:
            frame_number = self._frame_count

        timestamped_frame = TimestampedFrame(
            frame=frame.copy(),  # Store a copy to avoid external modifications
            timestamp=timestamp,
            frame_number=frame_number,
        )

        with self._lock:
            self._buffer.append(timestamped_frame)
            self._frame_count += 1

    def get_latest(self) -> Optional[TimestampedFrame]:
        """Get the most recent frame.

        Returns:
            Most recent TimestampedFrame, or None if buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def get_oldest(self) -> Optional[TimestampedFrame]:
        """Get the oldest frame in the buffer.

        Returns:
            Oldest TimestampedFrame, or None if buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[0]

    def get_recent(self, n: int) -> List[TimestampedFrame]:
        """Get the n most recent frames.

        Args:
            n: Number of frames to retrieve.

        Returns:
            List of TimestampedFrames, most recent last.
        """
        with self._lock:
            n = min(n, len(self._buffer))
            if n == 0:
                return []
            return list(self._buffer)[-n:]

    def get_all(self) -> List[TimestampedFrame]:
        """Get all frames in the buffer.

        Returns:
            List of all TimestampedFrames, oldest first.
        """
        with self._lock:
            return list(self._buffer)

    def get_frames_only(self, n: Optional[int] = None) -> List[np.ndarray]:
        """Get just the frame arrays without metadata.

        Args:
            n: Number of frames to retrieve. None for all frames.

        Returns:
            List of frame arrays.
        """
        with self._lock:
            if n is None:
                return [tf.frame for tf in self._buffer]
            n = min(n, len(self._buffer))
            return [tf.frame for tf in list(self._buffer)[-n:]]

    def get_frame_at(self, index: int) -> Optional[TimestampedFrame]:
        """Get frame at a specific buffer index.

        Args:
            index: Buffer index (0 = oldest, -1 = newest).

        Returns:
            TimestampedFrame at the index, or None if invalid index.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            try:
                return self._buffer[index]
            except IndexError:
                return None

    def get_by_frame_number(self, frame_number: int) -> Optional[TimestampedFrame]:
        """Get frame by its frame number.

        Args:
            frame_number: The frame number to find.

        Returns:
            TimestampedFrame with matching frame number, or None if not found.
        """
        with self._lock:
            for tf in self._buffer:
                if tf.frame_number == frame_number:
                    return tf
            return None

    def clear(self) -> None:
        """Remove all frames from the buffer."""
        with self._lock:
            self._buffer.clear()

    def pop(self) -> Optional[TimestampedFrame]:
        """Remove and return the oldest frame.

        Returns:
            Oldest TimestampedFrame, or None if buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer.popleft()

    def get_time_range(self) -> Optional[tuple[float, float]]:
        """Get the time range of frames in the buffer.

        Returns:
            Tuple of (oldest_timestamp, newest_timestamp), or None if empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return (self._buffer[0].timestamp, self._buffer[-1].timestamp)

    def get_duration(self) -> float:
        """Get the duration covered by frames in the buffer.

        Returns:
            Duration in seconds, or 0 if buffer has less than 2 frames.
        """
        time_range = self.get_time_range()
        if time_range is None:
            return 0.0
        return time_range[1] - time_range[0]

    def __len__(self) -> int:
        """Return the current buffer size."""
        return self.size

    def __iter__(self):
        """Iterate over frames in the buffer (oldest to newest)."""
        with self._lock:
            return iter(list(self._buffer))

    def __repr__(self) -> str:
        return f"FrameBuffer(size={self.size}, max_size={self._max_size})"
