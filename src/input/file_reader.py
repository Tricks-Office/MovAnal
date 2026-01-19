"""File-based video source implementation."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .video_source import VideoInfo, VideoSource

logger = logging.getLogger(__name__)


class FileVideoSource(VideoSource):
    """Video source for reading from video files.

    Supports common video formats: MP4, AVI, MKV, MOV, etc.

    Args:
        file_path: Path to the video file.
        loop: If True, loop the video when it ends.

    Example:
        >>> with FileVideoSource("video.mp4") as source:
        ...     print(f"Video info: {source.get_info()}")
        ...     for frame in source:
        ...         cv2.imshow("Frame", frame)
        ...         if cv2.waitKey(1) & 0xFF == ord('q'):
        ...             break
    """

    SUPPORTED_FORMATS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

    def __init__(self, file_path: Union[str, Path], loop: bool = False):
        super().__init__()
        self._file_path = Path(file_path)
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None

    @property
    def file_path(self) -> Path:
        """Get the video file path."""
        return self._file_path

    def open(self) -> bool:
        """Open the video file.

        Returns:
            True if successfully opened, False otherwise.

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            ValueError: If the file format is not supported.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"Video file not found: {self._file_path}")

        suffix = self._file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        self._cap = cv2.VideoCapture(str(self._file_path))

        if not self._cap.isOpened():
            logger.error(f"Failed to open video file: {self._file_path}")
            return False

        self._is_open = True
        self._current_frame = 0
        self._info = self._read_video_info()

        logger.info(
            f"Opened video: {self._file_path} "
            f"({self._info.width}x{self._info.height} @ {self._info.fps:.1f}fps, "
            f"{self._info.frame_count} frames)"
        )

        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the video file.

        Returns:
            Tuple of (success, frame).
        """
        if not self._is_open or self._cap is None:
            return False, None

        success, frame = self._cap.read()

        if not success:
            if self._loop:
                self.seek(0)
                success, frame = self._cap.read()
                if not success:
                    return False, None
            else:
                return False, None

        self._current_frame += 1
        return True, frame

    def close(self) -> None:
        """Close the video file and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._is_open = False
        self._current_frame = 0
        logger.debug(f"Closed video: {self._file_path}")

    def get_info(self) -> VideoInfo:
        """Get video metadata information.

        Returns:
            VideoInfo object with video properties.

        Raises:
            RuntimeError: If the video is not open.
        """
        if self._info is None:
            if not self._is_open:
                raise RuntimeError("Video is not open. Call open() first.")
            self._info = self._read_video_info()

        return self._info

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame number.

        Args:
            frame_number: Target frame number (0-indexed).

        Returns:
            True if seek was successful, False otherwise.
        """
        if not self._is_open or self._cap is None:
            return False

        if frame_number < 0:
            frame_number = 0

        info = self.get_info()
        if info.frame_count > 0 and frame_number >= info.frame_count:
            frame_number = info.frame_count - 1

        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if success:
            self._current_frame = frame_number

        return success

    def seek_time(self, time_ms: float) -> bool:
        """Seek to a specific time position.

        Args:
            time_ms: Target time in milliseconds.

        Returns:
            True if seek was successful, False otherwise.
        """
        if not self._is_open or self._cap is None:
            return False

        success = self._cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        if success:
            self._current_frame = int(
                self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            )

        return success

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame without changing the current position.

        Args:
            frame_number: Frame number to retrieve.

        Returns:
            Frame as numpy array, or None if failed.
        """
        if not self._is_open:
            return None

        current_pos = self._current_frame
        if self.seek(frame_number):
            success, frame = self.read()
            self.seek(current_pos)
            return frame if success else None

        return None

    def read_range(
        self, start_frame: int, end_frame: int
    ) -> list[np.ndarray]:
        """Read a range of frames.

        Args:
            start_frame: Starting frame number (inclusive).
            end_frame: Ending frame number (exclusive).

        Returns:
            List of frames as numpy arrays.
        """
        frames = []
        if not self.seek(start_frame):
            return frames

        for _ in range(end_frame - start_frame):
            success, frame = self.read()
            if not success or frame is None:
                break
            frames.append(frame)

        return frames

    def _read_video_info(self) -> VideoInfo:
        """Read video information from the capture object."""
        if self._cap is None:
            raise RuntimeError("Video capture is not initialized")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        return VideoInfo(
            width=width,
            height=height,
            fps=fps if fps > 0 else 30.0,
            frame_count=frame_count,
            fourcc=fourcc,
        )

    def __len__(self) -> int:
        """Return the total number of frames."""
        if self._info is None and self._is_open:
            self._info = self._read_video_info()
        return self._info.frame_count if self._info else 0

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return f"FileVideoSource('{self._file_path}', {status})"
