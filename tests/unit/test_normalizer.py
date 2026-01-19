"""Unit tests for FrameNormalizer."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.normalizer import FrameNormalizer


class TestFrameNormalizer:
    """Test cases for FrameNormalizer class."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample BGR frame for testing."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def grayscale_frame(self):
        """Create a sample grayscale frame for testing."""
        return np.random.randint(0, 256, (480, 640), dtype=np.uint8)

    def test_init_default(self):
        """Test default initialization."""
        normalizer = FrameNormalizer()
        assert normalizer._target_size is None
        assert normalizer._normalize_method == "minmax"
        assert normalizer._grayscale is False

    def test_resize(self, sample_frame):
        """Test frame resizing."""
        normalizer = FrameNormalizer(target_size=(256, 256))
        result = normalizer.resize(sample_frame, (256, 256))

        assert result.shape == (256, 256, 3)

    def test_normalize_minmax(self, sample_frame):
        """Test min-max normalization."""
        normalizer = FrameNormalizer(normalize_method="minmax")
        result = normalizer.normalize_pixels(sample_frame, "minmax")

        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_standard(self, sample_frame):
        """Test standard normalization."""
        normalizer = FrameNormalizer(normalize_method="standard")
        result = normalizer.normalize_pixels(sample_frame, "standard")

        assert result.dtype == np.float32
        # Standard normalization can produce values outside 0-1

    def test_denormalize_minmax(self, sample_frame):
        """Test denormalization reverses normalization."""
        normalizer = FrameNormalizer(normalize_method="minmax")
        normalized = normalizer.normalize_pixels(sample_frame, "minmax")
        denormalized = normalizer.denormalize(normalized)

        assert denormalized.dtype == np.uint8
        # Allow small differences due to float conversion
        np.testing.assert_allclose(denormalized, sample_frame, atol=1)

    def test_to_grayscale_bgr(self, sample_frame):
        """Test BGR to grayscale conversion."""
        normalizer = FrameNormalizer(grayscale=True)
        result = normalizer.to_grayscale(sample_frame)

        assert result.ndim == 2
        assert result.shape == (480, 640)

    def test_to_grayscale_already_gray(self, grayscale_frame):
        """Test grayscale conversion when already grayscale."""
        normalizer = FrameNormalizer(grayscale=True)
        result = normalizer.to_grayscale(grayscale_frame)

        assert result.ndim == 2
        np.testing.assert_array_equal(result, grayscale_frame)

    def test_apply_clahe_bgr(self, sample_frame):
        """Test CLAHE on BGR frame."""
        normalizer = FrameNormalizer(clahe_enabled=True)
        result = normalizer.apply_clahe(sample_frame)

        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype

    def test_apply_clahe_grayscale(self, grayscale_frame):
        """Test CLAHE on grayscale frame."""
        normalizer = FrameNormalizer(clahe_enabled=True)
        result = normalizer.apply_clahe(grayscale_frame)

        assert result.shape == grayscale_frame.shape

    def test_full_pipeline(self, sample_frame):
        """Test full normalization pipeline."""
        normalizer = FrameNormalizer(
            target_size=(256, 256),
            normalize_method="minmax",
            clahe_enabled=True,
            grayscale=False,
        )
        result = normalizer.normalize(sample_frame)

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_from_config(self):
        """Test creating normalizer from config dict."""
        config = {
            "resize": {"width": 320, "height": 240},
            "normalize": {"method": "standard"},
            "grayscale": True,
            "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid_size": [16, 16]},
        }
        normalizer = FrameNormalizer.from_config(config)

        assert normalizer._target_size == (320, 240)
        assert normalizer._normalize_method == "standard"
        assert normalizer._grayscale is True
        assert normalizer._clahe_enabled is True

    def test_update_clahe_params(self, sample_frame):
        """Test updating CLAHE parameters."""
        normalizer = FrameNormalizer(clahe_enabled=True)
        normalizer.update_clahe_params(clip_limit=4.0, tile_size=(16, 16))

        # Verify it works after update
        result = normalizer.apply_clahe(sample_frame)
        assert result.shape == sample_frame.shape

    def test_invalid_normalize_method(self, sample_frame):
        """Test invalid normalization method raises error."""
        normalizer = FrameNormalizer()
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalizer.normalize_pixels(sample_frame, "invalid")
