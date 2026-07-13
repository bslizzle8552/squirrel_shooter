"""Report Linux video devices and perform a one-frame OpenCV camera check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .camera_common import CameraOpenError, capture_dimensions, open_camera
from .config import ConfigError, DEFAULT_CONFIG_PATH, load_config


def linux_video_devices() -> list[tuple[Path, str | None]]:
    """Return Linux /dev/video* nodes with kernel-reported names when available."""

    devices: list[tuple[Path, str | None]] = []
    dev_directory = Path("/dev")
    if not dev_directory.is_dir():
        return devices

    for device in sorted(dev_directory.glob("video*")):
        name_file = Path("/sys/class/video4linux") / device.name / "name"
        try:
            name = name_file.read_text(encoding="utf-8").strip()
        except OSError:
            name = None
        devices.append((device, name))
    return devices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List camera devices and test one frame")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path (default: config/default.yaml)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    devices = linux_video_devices()

    if devices:
        print("Linux video devices:")
        for device, name in devices:
            description = f" - {name}" if name else ""
            print(f"  {device}{description}")
    elif sys.platform.startswith("linux"):
        print("No /dev/video* devices were found.")
    else:
        print("/dev/video* discovery is only available on Linux/Raspberry Pi OS.")

    try:
        app_config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    settings = app_config.camera
    print(
        f"Trying configured device index {settings.device_index} "
        f"at {settings.requested_width}x{settings.requested_height} "
        f"and {settings.requested_fps:g} FPS..."
    )

    try:
        capture = open_camera(settings)
    except CameraOpenError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    try:
        ok, frame = capture.read()
        width, height, fps = capture_dimensions(capture)
        if not ok or frame is None:
            print("FAILED: camera opened, but a test frame could not be read.", file=sys.stderr)
            return 1
        print(f"SUCCESS: read one frame. Camera reports {width}x{height} at {fps:.1f} FPS.")
        if (width, height) != (settings.requested_width, settings.requested_height):
            print("Note: the camera negotiated a different resolution than requested.")
        return 0
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
