"""Pytest configuration and fixtures."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_bgr_frame():
    """Create a sample BGR frame (480x640)."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_frame():
    """Create a sample grayscale frame (480x640)."""
    return np.random.randint(0, 256, (480, 640), dtype=np.uint8)


@pytest.fixture
def small_bgr_frame():
    """Create a small BGR frame for faster tests (120x160)."""
    return np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)


@pytest.fixture
def moving_frames():
    """Create two frames with simulated motion."""
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame2 = np.zeros((240, 320, 3), dtype=np.uint8)

    # Add a moving rectangle
    frame1[50:100, 50:100] = [255, 255, 255]
    frame2[50:100, 60:110] = [255, 255, 255]  # Moved 10px right

    return frame1, frame2


@pytest.fixture
def default_config():
    """Return default configuration dictionary."""
    return {
        "video": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "buffer_size": 30,
        },
        "preprocessing": {
            "resize": {"width": 256, "height": 256},
            "normalize": {"method": "minmax"},
            "grayscale": False,
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        },
        "features": {
            "optical_flow": {
                "pyr_scale": 0.5,
                "levels": 3,
                "winsize": 15,
                "iterations": 3,
            },
            "motion_history": {
                "duration": 0.5,
                "threshold": 32,
            },
        },
    }
