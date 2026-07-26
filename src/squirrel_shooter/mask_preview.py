"""Save one frame annotated with the configured motion inclusion zone."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .camera_common import CameraOpenError, open_camera
from .config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_config
from .watch_detection import draw_inclusion_zone, inclusion_zone_mask


DEFAULT_OUTPUT_PATH = Path("debug/active-mask.jpg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save one frame with the active motion mask overlaid"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--image",
        type=Path,
        help="annotate an existing image instead of reading one camera frame",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def _read_frame(config: AppConfig, image_path: Path | None) -> np.ndarray:
    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise OSError(f"Could not read image: {image_path}")
        return frame

    capture = open_camera(config.camera)
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraOpenError("The camera opened but did not return a frame")
        return frame
    finally:
        capture.release()


def save_mask_preview(
    config: AppConfig,
    output_path: Path,
    *,
    image_path: Path | None = None,
) -> Path:
    """Write an annotated mask preview from an image file or the configured camera."""

    frame = _read_frame(config, image_path)
    zone_mask = inclusion_zone_mask(config.motion, frame.shape[:2])
    annotated = draw_inclusion_zone(frame, zone_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Could not write mask preview: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = save_mask_preview(
            load_config(args.config),
            args.output,
            image_path=args.image,
        )
    except (CameraOpenError, ConfigError, OSError) as exc:
        print(f"Mask preview error: {exc}")
        return 1
    print(f"Saved active mask preview: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
