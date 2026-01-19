"""Unit tests for FileVideoSource."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.input.file_reader import FileVideoSource
from src.input.video_source import VideoInfo


class TestFileVideoSource:
    """Test cases for FileVideoSource class."""

    def test_init(self):
        """Test FileVideoSource initialization."""
        source = FileVideoSource("test.mp4")
        assert source.file_path == Path("test.mp4")
        assert not source.is_open()

    def test_unsupported_format(self):
        """Test that unsupported formats raise ValueError."""
        source = FileVideoSource("test.txt")
        with pytest.raises(ValueError, match="Unsupported video format"):
            source.open()

    def test_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        source = FileVideoSource("nonexistent.mp4")
        with pytest.raises(FileNotFoundError):
            source.open()

    @patch("cv2.VideoCapture")
    def test_open_success(self, mock_capture):
        """Test successful file opening."""
        # Setup mock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 640,   # CAP_PROP_FRAME_WIDTH
            1: 480,   # CAP_PROP_FRAME_HEIGHT
            5: 30.0,  # CAP_PROP_FPS
            7: 100,   # CAP_PROP_FRAME_COUNT
            6: 0,     # CAP_PROP_FOURCC
        }.get(prop, 0)
        mock_capture.return_value = mock_cap

        # Create temp file to pass exists() check
        with patch.object(Path, "exists", return_value=True):
            source = FileVideoSource("test.mp4")
            result = source.open()

        assert result is True
        assert source.is_open()

    @patch("cv2.VideoCapture")
    def test_read_frame(self, mock_capture):
        """Test reading frames."""
        # Setup mock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {
            0: 640, 1: 480, 5: 30.0, 7: 100, 6: 0
        }.get(prop, 0)
        mock_capture.return_value = mock_cap

        with patch.object(Path, "exists", return_value=True):
            source = FileVideoSource("test.mp4")
            source.open()
            success, frame = source.read()

        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    @patch("cv2.VideoCapture")
    def test_get_info(self, mock_capture):
        """Test getting video info."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 1920, 1: 1080, 5: 60.0, 7: 1000, 6: 0
        }.get(prop, 0)
        mock_capture.return_value = mock_cap

        with patch.object(Path, "exists", return_value=True):
            source = FileVideoSource("test.mp4")
            source.open()
            info = source.get_info()

        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 60.0
        assert info.frame_count == 1000

    @patch("cv2.VideoCapture")
    def test_seek(self, mock_capture):
        """Test seeking to frame."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.set.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 640, 1: 480, 5: 30.0, 7: 100, 6: 0
        }.get(prop, 0)
        mock_capture.return_value = mock_cap

        with patch.object(Path, "exists", return_value=True):
            source = FileVideoSource("test.mp4")
            source.open()
            result = source.seek(50)

        assert result is True
        mock_cap.set.assert_called()

    @patch("cv2.VideoCapture")
    def test_close(self, mock_capture):
        """Test closing video source."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 640, 1: 480, 5: 30.0, 7: 100, 6: 0
        }.get(prop, 0)
        mock_capture.return_value = mock_cap

        with patch.object(Path, "exists", return_value=True):
            source = FileVideoSource("test.mp4")
            source.open()
            source.close()

        assert not source.is_open()
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_context_manager(self, mock_capture):
        """Test context manager protocol."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 640, 1: 480, 5: 30.0, 7: 100, 6: 0
        }.get(prop, 0)
        mock_capture.return_value = mock_cap

        with patch.object(Path, "exists", return_value=True):
            with FileVideoSource("test.mp4") as source:
                assert source.is_open()
            assert not source.is_open()

    @patch("cv2.VideoCapture")
    def test_iterator(self, mock_capture):
        """Test iterator protocol."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 640, 1: 480, 5: 30.0, 7: 3, 6: 0
        }.get(prop, 0)

        # Return 3 frames then stop
        frames = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8)),
            (True, np.full((480, 640, 3), 128, dtype=np.uint8)),
            (False, None),
        ]
        mock_cap.read.side_effect = frames
        mock_capture.return_value = mock_cap

        with patch.object(Path, "exists", return_value=True):
            source = FileVideoSource("test.mp4")
            source.open()
            frame_list = list(source)

        assert len(frame_list) == 3


class TestVideoInfo:
    """Test cases for VideoInfo dataclass."""

    def test_resolution_property(self):
        """Test resolution tuple property."""
        info = VideoInfo(width=1920, height=1080, fps=30.0, frame_count=100)
        assert info.resolution == (1920, 1080)

    def test_is_live_property(self):
        """Test is_live property for live streams."""
        live_info = VideoInfo(width=640, height=480, fps=30.0, frame_count=-1)
        file_info = VideoInfo(width=640, height=480, fps=30.0, frame_count=100)

        assert live_info.is_live is True
        assert file_info.is_live is False
