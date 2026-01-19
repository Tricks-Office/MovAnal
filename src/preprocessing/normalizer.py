"""Frame normalization utilities for preprocessing."""

from typing import Optional, Tuple, Union

import cv2
import numpy as np


class FrameNormalizer:
    """Frame normalization for preprocessing pipeline.

    Handles resolution adjustment, pixel value normalization,
    grayscale conversion, and CLAHE enhancement.

    Args:
        target_size: Target resolution as (width, height). None to keep original.
        normalize_method: "minmax" for 0-1, "standard" for -1 to 1, None to skip.
        mean: Mean values for standard normalization (per channel).
        std: Standard deviation for standard normalization (per channel).
        grayscale: If True, convert to grayscale.
        clahe_enabled: If True, apply CLAHE enhancement.
        clahe_clip_limit: CLAHE clip limit parameter.
        clahe_tile_size: CLAHE tile grid size as (width, height).

    Example:
        >>> normalizer = FrameNormalizer(
        ...     target_size=(256, 256),
        ...     normalize_method="minmax",
        ...     clahe_enabled=True
        ... )
        >>> normalized = normalizer.normalize(frame)
        >>> original = normalizer.denormalize(normalized)
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize_method: Optional[str] = "minmax",
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        grayscale: bool = False,
        clahe_enabled: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
    ):
        self._target_size = target_size
        self._normalize_method = normalize_method
        self._mean = np.array(mean) if mean else np.array([0.485, 0.456, 0.406])
        self._std = np.array(std) if std else np.array([0.229, 0.224, 0.225])
        self._grayscale = grayscale
        self._clahe_enabled = clahe_enabled
        self._clahe_clip_limit = clahe_clip_limit
        self._clahe_tile_size = clahe_tile_size

        # Create CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size,
        )

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """Apply all normalization steps to a frame.

        Args:
            frame: Input frame (BGR, uint8).

        Returns:
            Normalized frame.
        """
        result = frame.copy()

        # Resize if target size specified
        if self._target_size is not None:
            result = self.resize(result, self._target_size)

        # Apply CLAHE if enabled
        if self._clahe_enabled:
            result = self.apply_clahe(result)

        # Convert to grayscale if requested
        if self._grayscale:
            result = self.to_grayscale(result)

        # Normalize pixel values
        if self._normalize_method is not None:
            result = self.normalize_pixels(result, self._normalize_method)

        return result

    def denormalize(self, frame: np.ndarray) -> np.ndarray:
        """Reverse pixel normalization.

        Args:
            frame: Normalized frame.

        Returns:
            Denormalized frame (uint8).
        """
        result = frame.copy()

        if self._normalize_method == "minmax":
            # Convert from 0-1 to 0-255
            result = (result * 255).astype(np.uint8)
        elif self._normalize_method == "standard":
            # Reverse standard normalization
            if result.ndim == 3:
                result = result * self._std + self._mean
            result = (result * 255).clip(0, 255).astype(np.uint8)

        return result

    def resize(
        self,
        frame: np.ndarray,
        size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Resize frame to target size.

        Args:
            frame: Input frame.
            size: Target size as (width, height).
            interpolation: OpenCV interpolation method.

        Returns:
            Resized frame.
        """
        return cv2.resize(frame, size, interpolation=interpolation)

    def normalize_pixels(
        self,
        frame: np.ndarray,
        method: str = "minmax",
    ) -> np.ndarray:
        """Normalize pixel values.

        Args:
            frame: Input frame (uint8).
            method: "minmax" for 0-1 range, "standard" for standardization.

        Returns:
            Normalized frame (float32).
        """
        frame = frame.astype(np.float32)

        if method == "minmax":
            # Scale to 0-1
            return frame / 255.0
        elif method == "standard":
            # Standardize using mean and std
            frame = frame / 255.0
            if frame.ndim == 3:
                frame = (frame - self._mean) / self._std
            return frame
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale.

        Args:
            frame: Input frame (BGR).

        Returns:
            Grayscale frame.
        """
        if frame.ndim == 2:
            return frame

        if frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            return frame

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Enhances local contrast while limiting noise amplification.
        Useful for handling varying lighting conditions.

        Args:
            frame: Input frame (BGR or grayscale).

        Returns:
            Enhanced frame.
        """
        if frame.ndim == 2:
            # Grayscale
            return self._clahe.apply(frame)
        else:
            # Convert to LAB color space, apply CLAHE to L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def update_clahe_params(
        self,
        clip_limit: Optional[float] = None,
        tile_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update CLAHE parameters.

        Args:
            clip_limit: New clip limit value.
            tile_size: New tile grid size.
        """
        if clip_limit is not None:
            self._clahe_clip_limit = clip_limit
        if tile_size is not None:
            self._clahe_tile_size = tile_size

        self._clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip_limit,
            tileGridSize=self._clahe_tile_size,
        )

    @classmethod
    def from_config(cls, config: dict) -> "FrameNormalizer":
        """Create FrameNormalizer from configuration dictionary.

        Args:
            config: Preprocessing configuration dictionary.

        Returns:
            Configured FrameNormalizer instance.
        """
        resize_config = config.get("resize", {})
        normalize_config = config.get("normalize", {})
        clahe_config = config.get("clahe", {})

        target_size = None
        if resize_config.get("width") and resize_config.get("height"):
            target_size = (resize_config["width"], resize_config["height"])

        return cls(
            target_size=target_size,
            normalize_method=normalize_config.get("method", "minmax"),
            mean=tuple(normalize_config.get("mean", [0.485, 0.456, 0.406])),
            std=tuple(normalize_config.get("std", [0.229, 0.224, 0.225])),
            grayscale=config.get("grayscale", False),
            clahe_enabled=clahe_config.get("enabled", False),
            clahe_clip_limit=clahe_config.get("clip_limit", 2.0),
            clahe_tile_size=tuple(clahe_config.get("tile_grid_size", [8, 8])),
        )

    def __repr__(self) -> str:
        return (
            f"FrameNormalizer(target_size={self._target_size}, "
            f"method={self._normalize_method}, grayscale={self._grayscale}, "
            f"clahe={self._clahe_enabled})"
        )
