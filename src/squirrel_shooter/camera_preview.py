"""Interactive USB-camera preview command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

from .camera_common import CameraOpenError, FrameRateMeter, annotate_frame, open_camera
from .config import ConfigError, DEFAULT_CONFIG_PATH, load_config
from .files import timestamped_output_path


WINDOW_TITLE = "Squirrel Shooter - USB camera (q quit, s save still)"


def display_available() -> bool:
    """Return whether the current environment appears to offer a GUI display."""

    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview and verify the USB camera")
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
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if not display_available():
        print(
            "No desktop display was detected. Use "
            "'python -m squirrel_shooter.camera_capture --seconds 10' "
            "for a headless test.",
            file=sys.stderr,
        )
        return 2

    try:
        capture = open_camera(app_config.camera)
    except CameraOpenError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    app_config.camera.output_directory.mkdir(parents=True, exist_ok=True)
    meter = FrameRateMeter()
    print("Preview started. Press q to quit or s to save an annotated still image.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print(
                    "Camera opened, but OpenCV could not read a frame. "
                    "Try the diagnostic command and another device index.",
                    file=sys.stderr,
                )
                return 1

            annotated = annotate_frame(frame, meter.update())
            cv2.imshow(WINDOW_TITLE, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return 0
            if key == ord("s"):
                output_path = timestamped_output_path(
                    app_config.camera.output_directory,
                    "camera-still",
                    ".jpg",
                )
                if cv2.imwrite(str(output_path), annotated):
                    print(f"Saved still image: {output_path.resolve()}")
                else:
                    print(f"Could not save still image: {output_path}", file=sys.stderr)
    except cv2.error as exc:
        print(f"OpenCV preview window failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
