"""Supervised command-line utility for the pan/tilt bench."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from .config import DEFAULT_CONFIG_PATH, load_config
from .pan_tilt import PanTiltController, PanTiltPosition


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely exercise the PCA9685 pan/tilt servos")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML configuration path")
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--hold", action="store_true", help="keep PWM active after the command")
    behavior.add_argument("--release-after", action="store_true", help="release PWM after the command")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("center", help="move smoothly to the calibrated center")
    subparsers.add_parser("park", help="move smoothly to the configured park position")
    subparsers.add_parser("release", help="release PWM without moving")
    subparsers.add_parser("position", help="print this process's software-tracked position")

    move = subparsers.add_parser("move", help="move smoothly to a pan and tilt target")
    move.add_argument("pan", type=float)
    move.add_argument("tilt", type=float)
    pan = subparsers.add_parser("pan", help="command the pan axis only")
    pan.add_argument("angle", type=float)
    tilt = subparsers.add_parser("tilt", help="command the tilt axis only")
    tilt.add_argument("angle", type=float)

    subparsers.add_parser("range-test", help="slowly exercise each calibrated axis range")
    subparsers.add_parser("tracking-demo", help="run a coordinated smooth tracking pattern")
    subparsers.add_parser("fast-demo", help="run a controlled fast-acquisition pattern")
    return parser


def _release_override(args: argparse.Namespace) -> bool | None:
    if args.hold:
        return False
    if args.release_after:
        return True
    return None


def _print_position(controller: PanTiltController) -> None:
    position = controller.position
    if position is None:
        print(
            "Tracked position: uncommanded (servos have no feedback; this process has not commanded both axes)."
        )
        return
    print(f"Tracked position: pan={position.pan:.2f}, tilt={position.tilt:.2f}")


def _run_sequence(
    controller: PanTiltController,
    points: tuple[PanTiltPosition, ...],
    move: Callable[..., PanTiltPosition],
    *,
    release: bool | None,
) -> None:
    for point in points:
        move(point.pan, point.tilt, release=False)
    if release is True or (release is None and controller.config.release_pwm_after_movement):
        controller.release_pwm()
    _print_position(controller)


def run_command(controller: PanTiltController, args: argparse.Namespace) -> None:
    config = controller.config
    release = _release_override(args)
    if args.command == "center":
        controller.center(release=release)
    elif args.command == "park":
        controller.park(release=release)
    elif args.command == "release":
        controller.release_pwm()
    elif args.command == "move":
        controller.move_to_smooth(args.pan, args.tilt, release=release)
    elif args.command == "pan":
        controller.move_pan(args.angle, release=release)
    elif args.command == "tilt":
        controller.move_tilt(args.angle, release=release)
    elif args.command == "range-test":
        points = (
            PanTiltPosition(config.pan_center, config.tilt_center),
            PanTiltPosition(config.pan_min, config.tilt_center),
            PanTiltPosition(config.pan_max, config.tilt_center),
            PanTiltPosition(config.pan_center, config.tilt_min),
            PanTiltPosition(config.pan_center, config.tilt_max),
            PanTiltPosition(config.pan_center, config.tilt_center),
        )
        _run_sequence(controller, points, controller.move_to_smooth, release=release)
        return
    elif args.command == "tracking-demo":
        pan_span = config.pan_max - config.pan_min
        tilt_span = config.tilt_max - config.tilt_min
        points = (
            PanTiltPosition(config.pan_center, config.tilt_center),
            PanTiltPosition(config.pan_min + pan_span * 0.25, config.tilt_min + tilt_span * 0.25),
            PanTiltPosition(config.pan_max - pan_span * 0.25, config.tilt_min + tilt_span * 0.25),
            PanTiltPosition(config.pan_max - pan_span * 0.25, config.tilt_max - tilt_span * 0.25),
            PanTiltPosition(config.pan_min + pan_span * 0.25, config.tilt_max - tilt_span * 0.25),
            PanTiltPosition(config.pan_center, config.tilt_center),
        )
        _run_sequence(controller, points, controller.move_to_smooth, release=release)
        return
    elif args.command == "fast-demo":
        points = (
            PanTiltPosition(config.pan_center, config.tilt_center),
            PanTiltPosition(config.pan_min, config.tilt_center),
            PanTiltPosition(config.pan_max, config.tilt_center),
            PanTiltPosition(config.pan_center, config.tilt_max),
            PanTiltPosition(config.pan_center, config.tilt_center),
        )
        _run_sequence(controller, points, controller.move_to_fast, release=release)
        return
    elif args.command != "position":
        raise ValueError(f"Unsupported bench command: {args.command}")
    _print_position(controller)


def _safe_cleanup(controller: PanTiltController) -> None:
    try:
        controller.cleanup()
    except Exception:
        LOGGER.exception("Pan/tilt cleanup also failed")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    controller: PanTiltController | None = None
    try:
        controller = PanTiltController(load_config(args.config).pan_tilt)
        run_command(controller, args)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; attempting configured safe park and cleanup")
        if controller is not None:
            _safe_cleanup(controller)
        return 130
    except Exception as exc:
        LOGGER.exception("Bench command failed: %s", exc)
        if controller is not None:
            _safe_cleanup(controller)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
