"""Region of Interest (ROI) management for video processing."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml


@dataclass
class ROI:
    """Region of Interest definition."""

    x: int
    y: int
    width: int
    height: int
    name: str = ""

    @property
    def x2(self) -> int:
        """Get the right x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Get the bottom y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the ROI."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Get the area of the ROI."""
        return self.width * self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return ROI as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def as_slice(self) -> Tuple[slice, slice]:
        """Return ROI as numpy slices for frame indexing."""
        return (slice(self.y, self.y2), slice(self.x, self.x2))

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the ROI."""
        px, py = point
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def intersects(self, other: "ROI") -> bool:
        """Check if this ROI intersects with another."""
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )

    def to_dict(self) -> Dict:
        """Convert ROI to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ROI":
        """Create ROI from dictionary."""
        return cls(**data)


class ROIManager:
    """Manager for multiple Regions of Interest.

    Supports defining, saving, loading, and applying ROIs to video frames.

    Args:
        frame_size: Frame size as (width, height) for validation.

    Example:
        >>> manager = ROIManager(frame_size=(640, 480))
        >>> manager.add_roi(100, 100, 200, 200, name="equipment1")
        >>> manager.add_roi(300, 100, 200, 200, name="equipment2")
        >>> cropped = manager.extract("equipment1", frame)
        >>> manager.save("rois.yaml")
    """

    def __init__(self, frame_size: Optional[Tuple[int, int]] = None):
        self._rois: Dict[str, ROI] = {}
        self._frame_size = frame_size
        self._roi_counter = 0

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        """Get the frame size used for validation."""
        return self._frame_size

    @frame_size.setter
    def frame_size(self, size: Tuple[int, int]) -> None:
        """Set the frame size."""
        self._frame_size = size

    @property
    def roi_names(self) -> List[str]:
        """Get list of all ROI names."""
        return list(self._rois.keys())

    @property
    def count(self) -> int:
        """Get the number of ROIs."""
        return len(self._rois)

    def add_roi(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        name: Optional[str] = None,
    ) -> str:
        """Add a new ROI.

        Args:
            x: Left x coordinate.
            y: Top y coordinate.
            width: ROI width.
            height: ROI height.
            name: Optional name for the ROI. Auto-generated if not provided.

        Returns:
            Name of the added ROI.

        Raises:
            ValueError: If ROI is invalid or out of frame bounds.
        """
        if name is None:
            name = f"roi_{self._roi_counter}"
            self._roi_counter += 1

        roi = ROI(x=x, y=y, width=width, height=height, name=name)
        self._validate_roi(roi)
        self._rois[name] = roi

        return name

    def add_roi_from_points(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        name: Optional[str] = None,
    ) -> str:
        """Add ROI from two corner points.

        Args:
            pt1: First corner point (x, y).
            pt2: Opposite corner point (x, y).
            name: Optional ROI name.

        Returns:
            Name of the added ROI.
        """
        x = min(pt1[0], pt2[0])
        y = min(pt1[1], pt2[1])
        width = abs(pt2[0] - pt1[0])
        height = abs(pt2[1] - pt1[1])

        return self.add_roi(x, y, width, height, name)

    def remove_roi(self, name: str) -> bool:
        """Remove an ROI by name.

        Args:
            name: Name of the ROI to remove.

        Returns:
            True if ROI was removed, False if not found.
        """
        if name in self._rois:
            del self._rois[name]
            return True
        return False

    def get_roi(self, name: str) -> Optional[ROI]:
        """Get an ROI by name.

        Args:
            name: Name of the ROI.

        Returns:
            ROI object, or None if not found.
        """
        return self._rois.get(name)

    def get_all_rois(self) -> Dict[str, ROI]:
        """Get all ROIs as a dictionary."""
        return self._rois.copy()

    def extract(self, name: str, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract ROI region from a frame.

        Args:
            name: Name of the ROI.
            frame: Input frame.

        Returns:
            Cropped frame region, or None if ROI not found.
        """
        roi = self._rois.get(name)
        if roi is None:
            return None

        y_slice, x_slice = roi.as_slice()
        return frame[y_slice, x_slice].copy()

    def extract_all(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all ROI regions from a frame.

        Args:
            frame: Input frame.

        Returns:
            Dictionary mapping ROI names to cropped regions.
        """
        return {name: self.extract(name, frame) for name in self._rois}

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_names: bool = True,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Draw ROI rectangles on a frame.

        Args:
            frame: Input frame.
            color: Rectangle color (BGR).
            thickness: Line thickness.
            show_names: If True, show ROI names.
            font_scale: Font scale for names.

        Returns:
            Frame with ROI rectangles drawn.
        """
        result = frame.copy()

        for name, roi in self._rois.items():
            cv2.rectangle(
                result,
                (roi.x, roi.y),
                (roi.x2, roi.y2),
                color,
                thickness,
            )

            if show_names:
                cv2.putText(
                    result,
                    name,
                    (roi.x, roi.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                )

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Save ROIs to a file.

        Args:
            path: Output file path (supports .yaml, .json).
        """
        path = Path(path)
        data = {"rois": [roi.to_dict() for roi in self._rois.values()]}

        if self._frame_size:
            data["frame_size"] = {"width": self._frame_size[0], "height": self._frame_size[1]}

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() in {".yaml", ".yml"}:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        elif path.suffix.lower() == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def load(self, path: Union[str, Path]) -> None:
        """Load ROIs from a file.

        Args:
            path: Input file path (supports .yaml, .json).
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"ROI file not found: {path}")

        if path.suffix.lower() in {".yaml", ".yml"}:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self._rois.clear()

        if "frame_size" in data:
            fs = data["frame_size"]
            self._frame_size = (fs["width"], fs["height"])

        for roi_data in data.get("rois", []):
            roi = ROI.from_dict(roi_data)
            self._rois[roi.name] = roi

    def clear(self) -> None:
        """Remove all ROIs."""
        self._rois.clear()

    def _validate_roi(self, roi: ROI) -> None:
        """Validate ROI dimensions and bounds.

        Args:
            roi: ROI to validate.

        Raises:
            ValueError: If ROI is invalid.
        """
        if roi.width <= 0 or roi.height <= 0:
            raise ValueError("ROI width and height must be positive")

        if roi.x < 0 or roi.y < 0:
            raise ValueError("ROI coordinates must be non-negative")

        if self._frame_size is not None:
            frame_w, frame_h = self._frame_size
            if roi.x2 > frame_w or roi.y2 > frame_h:
                raise ValueError(
                    f"ROI exceeds frame bounds. "
                    f"ROI: ({roi.x}, {roi.y}, {roi.x2}, {roi.y2}), "
                    f"Frame: ({frame_w}, {frame_h})"
                )

    def __len__(self) -> int:
        return len(self._rois)

    def __contains__(self, name: str) -> bool:
        return name in self._rois

    def __iter__(self):
        return iter(self._rois.values())

    def __repr__(self) -> str:
        return f"ROIManager(count={len(self._rois)}, frame_size={self._frame_size})"
