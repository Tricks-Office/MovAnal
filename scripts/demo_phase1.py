"""Phase 1 Demo Script

Demonstrates the complete Phase 1 pipeline:
- Video input (file or camera)
- Preprocessing (resize, CLAHE, normalization)
- Feature extraction (Optical Flow, Motion History)
- Real-time visualization

Usage:
    python scripts/demo_phase1.py --input video.mp4
    python scripts/demo_phase1.py --camera 0
    python scripts/demo_phase1.py --input video.mp4 --config configs/custom.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from src.input import FileVideoSource, CameraSource, FrameBuffer
from src.preprocessing import PreprocessingPipeline, ROIManager
from src.preprocessing.pipeline import Resize, CLAHE, Normalize
from src.features import FeatureExtractor
from src.utils.config import load_config_with_defaults
from src.utils.visualization import RealtimeVisualizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MovAnal Phase 1 Demo - Video Processing Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        help="Camera device index (0 for default webcam)",
    )
    parser.add_argument(
        "--rtsp",
        type=str,
        help="RTSP stream URL",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no-flow",
        action="store_true",
        help="Disable optical flow visualization",
    )
    parser.add_argument(
        "--no-motion-history",
        action="store_true",
        help="Disable motion history visualization",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        help="Path to save output video",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop video file playback",
    )

    args = parser.parse_args()

    # Validate input source
    if not any([args.input, args.camera is not None, args.rtsp]):
        parser.error("Must specify --input, --camera, or --rtsp")

    return args


def create_video_source(args):
    """Create appropriate video source based on arguments."""
    if args.input:
        logger.info(f"Opening video file: {args.input}")
        return FileVideoSource(args.input, loop=args.loop)
    elif args.camera is not None:
        logger.info(f"Opening camera device: {args.camera}")
        return CameraSource(args.camera)
    elif args.rtsp:
        logger.info(f"Opening RTSP stream: {args.rtsp}")
        return CameraSource(args.rtsp)


def main():
    """Main demo function."""
    args = parse_args()

    # Load configuration
    config_path = args.config if args.config else None
    config = load_config_with_defaults(config_path)
    logger.info("Configuration loaded")

    # Create video source
    video_source = create_video_source(args)

    # Create preprocessing pipeline
    preproc_config = config.get("preprocessing", {})
    pipeline = PreprocessingPipeline()

    resize_cfg = preproc_config.get("resize", {})
    if resize_cfg.get("width") and resize_cfg.get("height"):
        pipeline.add_step(Resize(resize_cfg["width"], resize_cfg["height"]))

    clahe_cfg = preproc_config.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        pipeline.add_step(CLAHE(
            clip_limit=clahe_cfg.get("clip_limit", 2.0),
            tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
        ))

    logger.info(f"Preprocessing pipeline: {pipeline.get_step_names()}")

    # Create feature extractor
    feature_extractor = FeatureExtractor(
        enable_optical_flow=not args.no_flow,
        enable_motion_history=not args.no_motion_history,
        optical_flow_config=config.get("features", {}).get("optical_flow", {}),
        motion_history_config=config.get("features", {}).get("motion_history", {}),
    )
    logger.info(f"Feature extractor: {feature_extractor}")

    # Create frame buffer
    buffer_size = config.get("video", {}).get("buffer_size", 30)
    frame_buffer = FrameBuffer(max_size=buffer_size)

    # Create visualizer
    vis_config = config.get("visualization", {})
    visualizer = RealtimeVisualizer(
        window_name=vis_config.get("window_name", "MovAnal Phase 1 Demo"),
        show_fps=vis_config.get("show_fps", True),
        show_flow=vis_config.get("show_flow", True) and not args.no_flow,
        show_roi=vis_config.get("show_roi", True),
        layout="horizontal",
    )

    # Optional: Create ROI manager
    roi_manager = ROIManager()
    visualizer.set_roi_manager(roi_manager)

    # Video writer for saving output
    video_writer = None
    if args.save_output:
        logger.info(f"Output will be saved to: {args.save_output}")

    # Processing statistics
    frame_count = 0
    total_time = 0
    feature_stats = []

    try:
        with video_source as source:
            video_info = source.get_info()
            logger.info(
                f"Video info: {video_info.width}x{video_info.height} "
                f"@ {video_info.fps:.1f}fps"
            )

            # Update ROI manager with frame size
            roi_manager.frame_size = (video_info.width, video_info.height)

            # Add sample ROI (can be removed or customized)
            # roi_manager.add_roi(100, 100, 200, 200, name="sample_roi")

            visualizer.start()

            logger.info("Starting video processing... Press 'q' to quit.")

            for frame in source:
                start_time = time.time()

                # Preprocess frame
                processed = pipeline(frame)

                # Extract features
                feature_result = feature_extractor.extract(frame)

                # Store in buffer
                frame_buffer.push(processed, frame_number=frame_count)

                # Get visualizations
                visualizations = feature_extractor.get_visualization(feature_result)

                # Prepare display info
                extra_info = {}
                if feature_result.optical_flow is not None:
                    extra_info["Motion"] = f"{feature_result.optical_flow.mean_magnitude:.2f}"

                if feature_result.motion_history is not None:
                    extra_info["MH Ratio"] = f"{feature_result.motion_history.motion_ratio:.2%}"

                extra_info["Frame"] = str(frame_count)
                extra_info["Buffer"] = f"{frame_buffer.size}/{frame_buffer.max_size}"

                # Show visualization
                flow_vis = visualizations.get("optical_flow")
                mh_vis = visualizations.get("motion_history")

                if not visualizer.show(
                    frame,
                    flow_vis=flow_vis,
                    motion_history_vis=mh_vis,
                    extra_info=extra_info,
                ):
                    break

                # Save output if requested
                if args.save_output:
                    if video_writer is None:
                        display_frame = frame  # You might want to capture the actual display
                        h, w = display_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            args.save_output, fourcc, video_info.fps, (w, h)
                        )
                    video_writer.write(frame)

                # Collect statistics
                process_time = time.time() - start_time
                total_time += process_time
                frame_count += 1

                if feature_result.optical_flow is not None:
                    feature_stats.append(feature_result.to_dict())

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        visualizer.stop()
        if video_writer is not None:
            video_writer.release()

    # Print summary statistics
    logger.info("=" * 50)
    logger.info("Processing Summary")
    logger.info("=" * 50)
    logger.info(f"Total frames processed: {frame_count}")

    if frame_count > 0:
        avg_time = total_time / frame_count
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        logger.info(f"Average processing time: {avg_time * 1000:.2f} ms/frame")
        logger.info(f"Average processing FPS: {avg_fps:.1f}")

    if feature_stats:
        of_magnitudes = [s.get("of_mean_magnitude", 0) for s in feature_stats]
        logger.info(f"Average optical flow magnitude: {np.mean(of_magnitudes):.3f}")
        logger.info(f"Max optical flow magnitude: {np.max(of_magnitudes):.3f}")

    logger.info("Demo completed.")


if __name__ == "__main__":
    main()
