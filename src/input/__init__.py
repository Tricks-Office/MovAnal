"""Video input module for handling various video sources."""

from .video_source import VideoSource
from .file_reader import FileVideoSource
from .camera import CameraSource
from .frame_buffer import FrameBuffer

__all__ = ["VideoSource", "FileVideoSource", "CameraSource", "FrameBuffer"]
