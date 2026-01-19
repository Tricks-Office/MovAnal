"""Preprocessing pipeline for chaining multiple preprocessing steps."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply the preprocessing step to a frame."""
        pass

    @property
    def name(self) -> str:
        """Get the name of this step."""
        return self.__class__.__name__


class Resize(PreprocessingStep):
    """Resize frame to target dimensions."""

    def __init__(
        self,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self._width = width
        self._height = height
        self._interpolation = interpolation

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(
            frame, (self._width, self._height), interpolation=self._interpolation
        )


class CLAHE(PreprocessingStep):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return self._clahe.apply(frame)
        else:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class Normalize(PreprocessingStep):
    """Normalize pixel values."""

    def __init__(
        self,
        method: str = "minmax",
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
    ):
        self._method = method
        self._mean = np.array(mean) if mean else np.array([0.485, 0.456, 0.406])
        self._std = np.array(std) if std else np.array([0.229, 0.224, 0.225])

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.float32)
        if self._method == "minmax":
            return frame / 255.0
        elif self._method == "standard":
            frame = frame / 255.0
            if frame.ndim == 3:
                frame = (frame - self._mean) / self._std
            return frame
        else:
            raise ValueError(f"Unknown normalization method: {self._method}")


class Grayscale(PreprocessingStep):
    """Convert frame to grayscale."""

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class ROICrop(PreprocessingStep):
    """Crop frame to ROI region."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return frame[
            self._y : self._y + self._height, self._x : self._x + self._width
        ].copy()


class GaussianBlur(PreprocessingStep):
    """Apply Gaussian blur."""

    def __init__(self, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 0):
        self._kernel_size = kernel_size
        self._sigma = sigma

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, self._kernel_size, self._sigma)


class MedianBlur(PreprocessingStep):
    """Apply median blur."""

    def __init__(self, kernel_size: int = 5):
        self._kernel_size = kernel_size

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(frame, self._kernel_size)


class ChannelConvert(PreprocessingStep):
    """Convert color channels."""

    def __init__(self, conversion: int):
        """
        Args:
            conversion: OpenCV color conversion code (e.g., cv2.COLOR_BGR2RGB)
        """
        self._conversion = conversion

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, self._conversion)


class Lambda(PreprocessingStep):
    """Apply a custom function as a preprocessing step."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name: str = "Lambda"):
        self._func = func
        self._name = name

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self._func(frame)

    @property
    def name(self) -> str:
        return self._name


class PreprocessingPipeline:
    """Pipeline for chaining multiple preprocessing steps.

    Applies a sequence of preprocessing steps to video frames.
    Steps are executed in order.

    Args:
        steps: List of PreprocessingStep objects or callable functions.

    Example:
        >>> pipeline = PreprocessingPipeline([
        ...     Resize(256, 256),
        ...     CLAHE(),
        ...     Normalize(),
        ... ])
        >>> processed = pipeline(frame)
        >>> # Or process batch
        >>> processed_batch = pipeline.process_batch(frames)
    """

    def __init__(self, steps: Optional[List[Union[PreprocessingStep, Callable]]] = None):
        self._steps: List[PreprocessingStep] = []
        if steps:
            for step in steps:
                self.add_step(step)

    def add_step(self, step: Union[PreprocessingStep, Callable]) -> "PreprocessingPipeline":
        """Add a preprocessing step to the pipeline.

        Args:
            step: PreprocessingStep or callable function.

        Returns:
            Self for method chaining.
        """
        if isinstance(step, PreprocessingStep):
            self._steps.append(step)
        elif callable(step):
            self._steps.append(Lambda(step))
        else:
            raise TypeError(f"Step must be PreprocessingStep or callable, got {type(step)}")

        return self

    def insert_step(
        self, index: int, step: Union[PreprocessingStep, Callable]
    ) -> "PreprocessingPipeline":
        """Insert a step at a specific position.

        Args:
            index: Position to insert at.
            step: PreprocessingStep or callable.

        Returns:
            Self for method chaining.
        """
        if isinstance(step, PreprocessingStep):
            self._steps.insert(index, step)
        elif callable(step):
            self._steps.insert(index, Lambda(step))
        else:
            raise TypeError(f"Step must be PreprocessingStep or callable")

        return self

    def remove_step(self, index: int) -> PreprocessingStep:
        """Remove a step at a specific position.

        Args:
            index: Position to remove from.

        Returns:
            The removed step.
        """
        return self._steps.pop(index)

    def clear(self) -> None:
        """Remove all steps from the pipeline."""
        self._steps.clear()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps to a frame.

        Args:
            frame: Input frame.

        Returns:
            Processed frame.
        """
        result = frame
        for step in self._steps:
            result = step(result)
        return result

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply preprocessing to a batch of frames.

        Args:
            frames: List of input frames.

        Returns:
            List of processed frames.
        """
        return [self(frame) for frame in frames]

    def get_step_names(self) -> List[str]:
        """Get names of all steps in the pipeline."""
        return [step.name for step in self._steps]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PreprocessingPipeline":
        """Create pipeline from configuration dictionary.

        Args:
            config: Preprocessing configuration.

        Returns:
            Configured PreprocessingPipeline.
        """
        pipeline = cls()

        # Add resize step if configured
        resize_config = config.get("resize", {})
        if resize_config.get("width") and resize_config.get("height"):
            pipeline.add_step(
                Resize(resize_config["width"], resize_config["height"])
            )

        # Add CLAHE if enabled
        clahe_config = config.get("clahe", {})
        if clahe_config.get("enabled", False):
            pipeline.add_step(
                CLAHE(
                    clip_limit=clahe_config.get("clip_limit", 2.0),
                    tile_grid_size=tuple(clahe_config.get("tile_grid_size", [8, 8])),
                )
            )

        # Add grayscale conversion if enabled
        if config.get("grayscale", False):
            pipeline.add_step(Grayscale())

        # Add normalization
        normalize_config = config.get("normalize", {})
        if normalize_config.get("method"):
            pipeline.add_step(
                Normalize(
                    method=normalize_config["method"],
                    mean=tuple(normalize_config.get("mean", [0.485, 0.456, 0.406])),
                    std=tuple(normalize_config.get("std", [0.229, 0.224, 0.225])),
                )
            )

        return pipeline

    def __len__(self) -> int:
        """Return number of steps in the pipeline."""
        return len(self._steps)

    def __iter__(self):
        """Iterate over steps in the pipeline."""
        return iter(self._steps)

    def __repr__(self) -> str:
        step_names = ", ".join(self.get_step_names())
        return f"PreprocessingPipeline([{step_names}])"
