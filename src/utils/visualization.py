"""Real-time visualization utilities for video processing."""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..preprocessing.roi_manager import ROIManager


class FPSCounter:
    """Simple FPS counter for performance monitoring."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._timestamps: List[float] = []

    def tick(self) -> float:
        """Record a frame timestamp and return current FPS."""
        now = time.time()
        self._timestamps.append(now)

        # Keep only recent timestamps
        if len(self._timestamps) > self._window_size:
            self._timestamps = self._timestamps[-self._window_size:]

        # Calculate FPS
        if len(self._timestamps) < 2:
            return 0.0

        duration = self._timestamps[-1] - self._timestamps[0]
        if duration > 0:
            return (len(self._timestamps) - 1) / duration
        return 0.0

    def reset(self) -> None:
        """Reset the FPS counter."""
        self._timestamps.clear()


class RealtimeVisualizer:
    """Real-time visualization for video processing pipeline.

    Displays original frames with overlays for optical flow, ROIs,
    and performance metrics.

    Args:
        window_name: Name of the OpenCV window.
        show_fps: If True, display FPS counter.
        show_flow: If True, overlay optical flow visualization.
        show_roi: If True, draw ROI regions.
        layout: Display layout ("single", "grid", "horizontal").

    Example:
        >>> visualizer = RealtimeVisualizer()
        >>> visualizer.start()
        >>> for frame in video:
        ...     processed = process(frame)
        ...     if not visualizer.show(frame, flow_vis=flow_image):
        ...         break
        >>> visualizer.stop()
    """

    def __init__(
        self,
        window_name: str = "MovAnal",
        show_fps: bool = True,
        show_flow: bool = True,
        show_roi: bool = True,
        layout: str = "horizontal",
    ):
        self._window_name = window_name
        self._show_fps = show_fps
        self._show_flow = show_flow
        self._show_roi = show_roi
        self._layout = layout

        self._fps_counter = FPSCounter()
        self._roi_manager: Optional[ROIManager] = None
        self._running = False

    def set_roi_manager(self, roi_manager: ROIManager) -> None:
        """Set the ROI manager for ROI visualization."""
        self._roi_manager = roi_manager

    def start(self) -> None:
        """Initialize the visualization window."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        self._running = True
        self._fps_counter.reset()

    def stop(self) -> None:
        """Close the visualization window."""
        cv2.destroyWindow(self._window_name)
        self._running = False

    def show(
        self,
        frame: np.ndarray,
        flow_vis: Optional[np.ndarray] = None,
        motion_history_vis: Optional[np.ndarray] = None,
        extra_info: Optional[Dict[str, str]] = None,
        wait_ms: int = 1,
    ) -> bool:
        """Display frame with overlays.

        Args:
            frame: Main frame to display (BGR).
            flow_vis: Optional optical flow visualization.
            motion_history_vis: Optional motion history visualization.
            extra_info: Optional dictionary of text to display.
            wait_ms: Milliseconds to wait for key press.

        Returns:
            False if 'q' was pressed or window was closed, True otherwise.
        """
        if not self._running:
            return False

        # Update FPS
        fps = self._fps_counter.tick()

        # Prepare main frame
        display_frame = frame.copy()

        # Draw ROIs if enabled
        if self._show_roi and self._roi_manager is not None:
            display_frame = self._roi_manager.draw(display_frame)

        # Add FPS text
        if self._show_fps:
            self._draw_text(display_frame, f"FPS: {fps:.1f}", (10, 30))

        # Add extra info
        if extra_info:
            y_offset = 60
            for key, value in extra_info.items():
                self._draw_text(display_frame, f"{key}: {value}", (10, y_offset))
                y_offset += 30

        # Build final display based on layout
        if self._layout == "single" or (flow_vis is None and motion_history_vis is None):
            final_display = display_frame
        elif self._layout == "horizontal":
            final_display = self._build_horizontal_layout(
                display_frame, flow_vis, motion_history_vis
            )
        elif self._layout == "grid":
            final_display = self._build_grid_layout(
                display_frame, flow_vis, motion_history_vis
            )
        else:
            final_display = display_frame

        # Show the frame
        cv2.imshow(self._window_name, final_display)

        # Handle key press
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        return True

    def _build_horizontal_layout(
        self,
        main_frame: np.ndarray,
        flow_vis: Optional[np.ndarray],
        mh_vis: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build horizontal layout with multiple panels."""
        panels = [main_frame]

        if flow_vis is not None and self._show_flow:
            flow_resized = self._resize_to_match(flow_vis, main_frame)
            self._draw_text(flow_resized, "Optical Flow", (10, 30))
            panels.append(flow_resized)

        if mh_vis is not None:
            mh_resized = self._resize_to_match(mh_vis, main_frame)
            self._draw_text(mh_resized, "Motion History", (10, 30))
            panels.append(mh_resized)

        return np.hstack(panels)

    def _build_grid_layout(
        self,
        main_frame: np.ndarray,
        flow_vis: Optional[np.ndarray],
        mh_vis: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build 2x2 grid layout."""
        h, w = main_frame.shape[:2]
        grid = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Top-left: main frame (scaled to half)
        main_small = cv2.resize(main_frame, (w, h))
        grid[:h, :w] = main_small

        # Top-right: optical flow
        if flow_vis is not None and self._show_flow:
            flow_small = cv2.resize(flow_vis, (w, h))
            if flow_small.ndim == 2:
                flow_small = cv2.cvtColor(flow_small, cv2.COLOR_GRAY2BGR)
            self._draw_text(flow_small, "Optical Flow", (5, 20), font_scale=0.5)
            grid[:h, w:] = flow_small

        return grid

    def _resize_to_match(
        self, image: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Resize image to match reference dimensions."""
        h, w = reference.shape[:2]
        resized = cv2.resize(image, (w, h))

        # Convert grayscale to BGR if needed
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        return resized

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.7,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        """Draw text with background on frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = position

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 2, y - text_h - 5),
            (x + text_w + 2, y + baseline),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def create_comparison_view(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        error_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create side-by-side comparison of original and reconstructed frames.

        Useful for autoencoder visualization in Phase 2.

        Args:
            original: Original input frame.
            reconstructed: Reconstructed frame from autoencoder.
            error_map: Optional pixel-wise error map.

        Returns:
            Combined visualization image.
        """
        # Ensure same size
        h, w = original.shape[:2]
        reconstructed = cv2.resize(reconstructed, (w, h))

        # Add labels
        orig_labeled = original.copy()
        recon_labeled = reconstructed.copy()
        self._draw_text(orig_labeled, "Original", (10, 30))
        self._draw_text(recon_labeled, "Reconstructed", (10, 30))

        panels = [orig_labeled, recon_labeled]

        if error_map is not None:
            # Normalize and colorize error map
            error_normalized = cv2.normalize(
                error_map, None, 0, 255, cv2.NORM_MINMAX
            )
            error_colored = cv2.applyColorMap(
                error_normalized.astype(np.uint8), cv2.COLORMAP_JET
            )
            error_colored = cv2.resize(error_colored, (w, h))
            self._draw_text(error_colored, "Error Map", (10, 30))
            panels.append(error_colored)

        return np.hstack(panels)

    @property
    def is_running(self) -> bool:
        """Check if visualizer is running."""
        return self._running

    def __enter__(self) -> "RealtimeVisualizer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        return (
            f"RealtimeVisualizer(window={self._window_name}, "
            f"fps={self._show_fps}, flow={self._show_flow}, roi={self._show_roi})"
        )
