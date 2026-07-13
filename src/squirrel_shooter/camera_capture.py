"""Headless USB-camera recording command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import monotonic

import cv2

from .camera_common import CameraOpenError, FrameRateMeter, annotate_frame, open_camera
from .config import ConfigError, DEFAULT_CONFIG_PATH, load_config
from .files import timestamped_output_path


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be greater than zero")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record an annotated camera test video")
    parser.add_argument(
        "--seconds",
        type=positive_seconds,
        default=10.0,
        help="recording length in seconds (default: 10)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path (default: config/default.yaml)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        app_config = load_config(args.config)
        capture = open_camera(app_config.camera)
    except (ConfigError, CameraOpenError) as exc:
        print(f"Camera setup error: {exc}", file=sys.stderr)
        return 1

    output_directory = app_config.camera.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = timestamped_output_path(
        output_directory,
        "camera-test",
        ".avi",
    )

    writer: cv2.VideoWriter | None = None
    frames_written = 0
    meter = FrameRateMeter()

    try:
        ok, first_frame = capture.read()
        if not ok:
            print("Camera opened, but no frame could be read.", file=sys.stderr)
            return 1

        height, width = first_frame.shape[:2]
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        output_fps = reported_fps if reported_fps > 0 else app_config.camera.requested_fps
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            print(
                "OpenCV could not create the MJPG/AVI output file. "
                "Check that the output directory is writable.",
                file=sys.stderr,
            )
            return 1

        print(
            f"Recording {args.seconds:g} seconds at {width}x{height} "
            f"to {output_path.resolve()}"
        )
        deadline = monotonic() + args.seconds
        frame = first_frame

        while monotonic() < deadline:
            writer.write(annotate_frame(frame, meter.update()))
            frames_written += 1
            ok, frame = capture.read()
            if not ok:
                print("Camera stopped returning frames before the test ended.", file=sys.stderr)
                break
    except KeyboardInterrupt:
        print("Recording stopped early.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames_written == 0:
        print("No frames were written.", file=sys.stderr)
        return 1

    print(f"Saved {frames_written} frames to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
