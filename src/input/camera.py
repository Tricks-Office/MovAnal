"""Camera and RTSP stream video source implementation."""

import logging
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .video_source import VideoInfo, VideoSource

logger = logging.getLogger(__name__)


class CameraSource(VideoSource):
    """Video source for USB cameras and RTSP streams.

    Supports:
    - USB cameras via device index
    - RTSP streams via URL
    - HTTP streams

    Args:
        source: Camera device index (int) or stream URL (str).
        width: Desired frame width (only for USB cameras).
        height: Desired frame height (only for USB cameras).
        fps: Desired frame rate (only for USB cameras).
        buffer_size: OpenCV capture buffer size.
        reconnect_attempts: Number of reconnection attempts on failure.
        reconnect_delay: Delay between reconnection attempts in seconds.

    Example:
        >>> # USB camera
        >>> with CameraSource(0) as cam:
        ...     for frame in cam:
        ...         cv2.imshow("Camera", frame)
        ...         if cv2.waitKey(1) & 0xFF == ord('q'):
        ...             break

        >>> # RTSP stream
        >>> with CameraSource("rtsp://192.168.1.100:554/stream") as cam:
        ...     for frame in cam:
        ...         process(frame)
    """

    def __init__(
        self,
        source: Union[int, str],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 1,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ):
        super().__init__()
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._buffer_size = buffer_size
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10

    @property
    def source(self) -> Union[int, str]:
        """Get the camera source (device index or URL)."""
        return self._source

    @property
    def is_rtsp(self) -> bool:
        """Check if this is an RTSP stream."""
        return isinstance(self._source, str) and self._source.startswith("rtsp://")

    @property
    def is_http(self) -> bool:
        """Check if this is an HTTP stream."""
        return isinstance(self._source, str) and (
            self._source.startswith("http://") or self._source.startswith("https://")
        )

    @property
    def is_usb(self) -> bool:
        """Check if this is a USB camera."""
        return isinstance(self._source, int)

    def open(self) -> bool:
        """Open the camera or stream.

        Returns:
            True if successfully opened, False otherwise.
        """
        for attempt in range(self._reconnect_attempts):
            try:
                if self.is_rtsp:
                    self._cap = cv2.VideoCapture(
                        self._source, cv2.CAP_FFMPEG
                    )
                else:
                    self._cap = cv2.VideoCapture(self._source)

                if self._cap is None or not self._cap.isOpened():
                    logger.warning(
                        f"Failed to open camera (attempt {attempt + 1}/"
                        f"{self._reconnect_attempts}): {self._source}"
                    )
                    time.sleep(self._reconnect_delay)
                    continue

                self._configure_capture()
                self._is_open = True
                self._current_frame = 0
                self._consecutive_failures = 0
                self._info = self._read_video_info()

                logger.info(
                    f"Opened camera: {self._source} "
                    f"({self._info.width}x{self._info.height} @ {self._info.fps:.1f}fps)"
                )

                return True

            except Exception as e:
                logger.error(f"Error opening camera: {e}")
                time.sleep(self._reconnect_delay)

        logger.error(
            f"Failed to open camera after {self._reconnect_attempts} attempts"
        )
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the camera.

        Returns:
            Tuple of (success, frame).
        """
        if not self._is_open or self._cap is None:
            return False, None

        success, frame = self._cap.read()

        if not success:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                logger.warning("Too many consecutive failures, attempting reconnect")
                if self._reconnect():
                    success, frame = self._cap.read()
                    if success:
                        self._consecutive_failures = 0
                else:
                    return False, None
            else:
                return False, None
        else:
            self._consecutive_failures = 0
            self._current_frame += 1

        return success, frame

    def close(self) -> None:
        """Close the camera and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._is_open = False
        self._current_frame = 0
        logger.debug(f"Closed camera: {self._source}")

    def get_info(self) -> VideoInfo:
        """Get camera information.

        Returns:
            VideoInfo object with camera properties.

        Raises:
            RuntimeError: If the camera is not open.
        """
        if self._info is None:
            if not self._is_open:
                raise RuntimeError("Camera is not open. Call open() first.")
            self._info = self._read_video_info()

        return self._info

    def grab_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab the latest frame, discarding buffered frames.

        Useful for real-time applications where latency matters more
        than getting every frame.

        Returns:
            Tuple of (success, frame).
        """
        if not self._is_open or self._cap is None:
            return False, None

        # Grab and discard frames to get to the latest
        for _ in range(self._buffer_size):
            self._cap.grab()

        return self._cap.retrieve()

    def set_resolution(self, width: int, height: int) -> bool:
        """Set the camera resolution.

        Args:
            width: Desired frame width.
            height: Desired frame height.

        Returns:
            True if resolution was set successfully.
        """
        if not self._is_open or self._cap is None:
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._width = actual_width
        self._height = actual_height
        self._info = None  # Reset info cache

        return actual_width == width and actual_height == height

    def set_fps(self, fps: int) -> bool:
        """Set the camera frame rate.

        Args:
            fps: Desired frame rate.

        Returns:
            True if fps was set successfully.
        """
        if not self._is_open or self._cap is None:
            return False

        self._cap.set(cv2.CAP_PROP_FPS, fps)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = int(actual_fps)
        self._info = None

        return abs(actual_fps - fps) < 1

    def _configure_capture(self) -> None:
        """Configure capture settings."""
        if self._cap is None:
            return

        # Set buffer size (reduces latency for real-time applications)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        # Set resolution and fps for USB cameras
        if self.is_usb:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def _read_video_info(self) -> VideoInfo:
        """Read video information from the capture object."""
        if self._cap is None:
            raise RuntimeError("Camera capture is not initialized")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        return VideoInfo(
            width=width,
            height=height,
            fps=fps if fps > 0 else self._fps,
            frame_count=-1,  # Live stream has unknown frame count
            fourcc="",
        )

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the camera.

        Returns:
            True if reconnection was successful.
        """
        logger.info(f"Attempting to reconnect to camera: {self._source}")
        self.close()
        return self.open()

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        source_type = "USB" if self.is_usb else "RTSP" if self.is_rtsp else "HTTP"
        return f"CameraSource({source_type}, {self._source}, {status})"
