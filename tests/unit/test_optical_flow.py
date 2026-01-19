"""Unit tests for OpticalFlowExtractor."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.optical_flow import OpticalFlowExtractor, OpticalFlowResult


class TestOpticalFlowExtractor:
    """Test cases for OpticalFlowExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create an OpticalFlowExtractor instance."""
        return OpticalFlowExtractor()

    @pytest.fixture
    def sample_frames(self):
        """Create sample frames for testing."""
        # Create two slightly different frames
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.zeros((240, 320, 3), dtype=np.uint8)

        # Add a moving object
        frame1[100:150, 100:150] = 255
        frame2[100:150, 110:160] = 255  # Shifted 10 pixels right

        return frame1, frame2

    @pytest.fixture
    def static_frames(self):
        """Create identical frames (no motion)."""
        frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        return frame.copy(), frame.copy()

    def test_init_default(self):
        """Test default initialization."""
        extractor = OpticalFlowExtractor()
        assert extractor._pyr_scale == 0.5
        assert extractor._levels == 3
        assert extractor._winsize == 15

    def test_compute_returns_result(self, extractor, sample_frames):
        """Test that compute returns OpticalFlowResult."""
        frame1, frame2 = sample_frames
        result = extractor.compute(frame1, frame2)

        assert isinstance(result, OpticalFlowResult)
        assert result.flow is not None
        assert result.magnitude is not None
        assert result.angle is not None

    def test_flow_shape(self, extractor, sample_frames):
        """Test output shapes."""
        frame1, frame2 = sample_frames
        result = extractor.compute(frame1, frame2)

        h, w = frame1.shape[:2]
        assert result.flow.shape == (h, w, 2)
        assert result.magnitude.shape == (h, w)
        assert result.angle.shape == (h, w)

    def test_visualization_shape(self, extractor, sample_frames):
        """Test visualization output shape."""
        frame1, frame2 = sample_frames
        result = extractor.compute(frame1, frame2)

        assert result.visualization is not None
        h, w = frame1.shape[:2]
        assert result.visualization.shape == (h, w, 3)

    def test_motion_detected(self, extractor, sample_frames):
        """Test that motion is detected in moving frames."""
        frame1, frame2 = sample_frames
        result = extractor.compute(frame1, frame2)

        assert result.mean_magnitude > 0
        assert result.max_magnitude > 0

    def test_no_motion_static(self, extractor, static_frames):
        """Test minimal motion for static frames."""
        frame1, frame2 = static_frames
        result = extractor.compute(frame1, frame2)

        # Static frames should have very low motion
        assert result.mean_magnitude < 0.1

    def test_incremental_first_frame(self, extractor, sample_frames):
        """Test incremental processing returns None on first frame."""
        frame1, _ = sample_frames
        result = extractor.compute_incremental(frame1)

        assert result is None

    def test_incremental_second_frame(self, extractor, sample_frames):
        """Test incremental processing returns result on second frame."""
        frame1, frame2 = sample_frames
        extractor.compute_incremental(frame1)
        result = extractor.compute_incremental(frame2)

        assert result is not None
        assert isinstance(result, OpticalFlowResult)

    def test_reset(self, extractor, sample_frames):
        """Test reset clears internal state."""
        frame1, _ = sample_frames
        extractor.compute_incremental(frame1)
        extractor.reset()
        result = extractor.compute_incremental(frame1)

        assert result is None  # Should act like first frame

    def test_get_motion_regions(self, extractor, sample_frames):
        """Test motion region extraction."""
        frame1, frame2 = sample_frames
        result = extractor.compute(frame1, frame2)
        mask = extractor.get_motion_regions(result, threshold=1.0)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_get_motion_statistics(self, extractor, sample_frames):
        """Test motion statistics computation."""
        frame1, frame2 = sample_frames
        result = extractor.compute(frame1, frame2)
        stats = extractor.get_motion_statistics(result)

        assert "mean_magnitude" in stats
        assert "max_magnitude" in stats
        assert "std_magnitude" in stats
        assert "motion_area_ratio" in stats
        assert "motion_energy" in stats
        assert "dominant_direction" in stats

    def test_grayscale_input(self, extractor):
        """Test with grayscale input frames."""
        frame1 = np.zeros((240, 320), dtype=np.uint8)
        frame2 = np.zeros((240, 320), dtype=np.uint8)
        frame1[100:150, 100:150] = 255
        frame2[100:150, 110:160] = 255

        result = extractor.compute(frame1, frame2)

        assert result is not None
        assert result.flow.shape == (240, 320, 2)

    def test_from_config(self):
        """Test creating extractor from config."""
        config = {
            "optical_flow": {
                "pyr_scale": 0.4,
                "levels": 5,
                "winsize": 21,
                "iterations": 5,
            }
        }
        extractor = OpticalFlowExtractor.from_config(config)

        assert extractor._pyr_scale == 0.4
        assert extractor._levels == 5
        assert extractor._winsize == 21
        assert extractor._iterations == 5


class TestOpticalFlowResult:
    """Test cases for OpticalFlowResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result."""
        flow = np.random.randn(100, 100, 2).astype(np.float32)
        magnitude = np.abs(flow[..., 0])
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        return OpticalFlowResult(flow=flow, magnitude=magnitude, angle=angle)

    def test_mean_magnitude(self, sample_result):
        """Test mean_magnitude property."""
        expected = float(np.mean(sample_result.magnitude))
        assert abs(sample_result.mean_magnitude - expected) < 1e-6

    def test_max_magnitude(self, sample_result):
        """Test max_magnitude property."""
        expected = float(np.max(sample_result.magnitude))
        assert abs(sample_result.max_magnitude - expected) < 1e-6

    def test_motion_energy(self, sample_result):
        """Test motion_energy property."""
        expected = float(np.sum(sample_result.magnitude ** 2))
        assert abs(sample_result.motion_energy - expected) < 1e-3
