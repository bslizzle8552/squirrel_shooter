"""Unattended, vision-only garden motion watcher."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import cv2

from .camera_common import CameraOpenError, FrameRateMeter, capture_dimensions, open_camera
from .camera_preview import display_available
from .config import ConfigError, DEFAULT_CONFIG_PATH, load_config
from .diagnostics import cleanup_oldest
from .event_report import generate_reports
from .event_storage import EventLogWriter, EventRecorder, RollingFrameBuffer, SessionLog, enforce_retention, recover_incomplete_events
from .files import timestamped_output_path
from .watch_detection import MotionWatcherDetector, annotate_watch_frame


WINDOW_TITLE = "Squirrel Squirter watcher (q quit, s still, e test event, r report)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vision-only garden motion watcher")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--headless", action="store_true", help="do not open a preview window; stop with Ctrl+C")
    return parser


def _camera_metadata(config: object, actual_width: int, actual_height: int, camera_reported_fps: float) -> dict[str, object]:
    camera = config.camera  # type: ignore[attr-defined]
    return {
        "requested_width": camera.requested_width,
        "requested_height": camera.requested_height,
        "requested_fps": camera.requested_fps,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "camera_reported_fps": camera_reported_fps,
        "measured_camera_fps": 0.0,
        "camera_mode_if_known": camera.camera_mode_if_known,
        "ir_mode_if_explicitly_detected_or_configured": camera.ir_mode_if_explicitly_detected_or_configured,
        "low_fps_observed": False,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    requested = {
        "requested_width": config.camera.requested_width,
        "requested_height": config.camera.requested_height,
        "requested_fps": config.camera.requested_fps,
        "camera_mode_if_known": config.camera.camera_mode_if_known,
        "ir_mode_if_explicitly_detected_or_configured": config.camera.ir_mode_if_explicitly_detected_or_configured,
    }
    logs = EventLogWriter(config)
    session = SessionLog(config, requested)
    events_root = config.camera.output_directory / "events"
    recovered = recover_incomplete_events(events_root)
    if recovered:
        session.data["recovered_incomplete_events"] = [str(path) for path in recovered]
        session.save()
    try:
        capture = open_camera(config.camera)
    except CameraOpenError as exc:
        session.data["camera_open_result"] = "failed"
        session.finish(clean=False, exception=str(exc))
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    actual_width, actual_height, reported_fps = capture_dimensions(capture)
    camera_metadata = _camera_metadata(config, actual_width, actual_height, reported_fps)
    session.data["camera_open_result"] = "success"
    session.data["actual_camera_mode"] = {
        "actual_width": actual_width,
        "actual_height": actual_height,
        "camera_reported_fps": reported_fps,
        "camera_mode_if_known": config.camera.camera_mode_if_known,
        "ir_mode_if_explicitly_detected_or_configured": config.camera.ir_mode_if_explicitly_detected_or_configured,
    }
    session.save()
    detector = MotionWatcherDetector(config.motion)
    recorder = EventRecorder(config, logs, camera_metadata)
    prebuffer = RollingFrameBuffer(config.motion.event_lifecycle.pre_event_seconds)
    meter = FrameRateMeter(smoothing=0.12)
    show_preview = not args.headless and display_available()
    failed_reads = 0
    last_rejection: str | None = None
    last_session_save = monotonic()
    clean = False
    caught_exception: str | None = None
    print(
        f"Watcher started at {actual_width}x{actual_height}; camera reports {reported_fps:.2f} FPS. "
        + ("Use q/s/e/r in the preview window." if show_preview else "Headless mode: use Ctrl+C to stop safely.")
    )

    try:
        while True:
            ok, frame = capture.read()
            now = monotonic()
            if not ok or frame is None:
                failed_reads += 1
                session.increment("dropped_or_failed_frame_reads")
                session.increment("camera_read_errors")
                if failed_reads < config.camera.reopen_after_failed_reads:
                    continue
                capture.release()
                print("Camera read failures reached the reopen threshold; attempting recovery.", file=sys.stderr)
                sleep(config.camera.reopen_delay_seconds)
                try:
                    capture = open_camera(config.camera)
                    failed_reads = 0
                    detector.clear_candidates()
                    session.data.setdefault("camera_reopens", 0)
                    session.data["camera_reopens"] += 1
                except CameraOpenError as exc:
                    session.data["exception_details"].append(str(exc))
                    session.save()
                continue
            failed_reads = 0
            measured_fps = meter.update(now)
            session.sample_fps(measured_fps)
            camera_metadata["measured_camera_fps"] = measured_fps
            camera_metadata["low_fps_observed"] = 0 < measured_fps < config.camera.low_fps_threshold
            result = detector.process(frame, now=now)
            session.increment("raw_contours", result.raw_contour_count)
            session.increment("grouped_candidates", len(result.groups))
            annotated = annotate_watch_frame(frame, result, measured_fps=measured_fps)

            if result.global_motion.reason and result.state.value == "GLOBAL_RECOVERY":
                reason = result.global_motion.reason
                if last_rejection != reason:
                    rejection = {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "reason": reason,
                        **result.global_motion.as_dict(),
                        "measured_camera_fps": measured_fps,
                        "low_fps_observed": camera_metadata["low_fps_observed"],
                        "ir_mode_if_explicitly_detected_or_configured": camera_metadata["ir_mode_if_explicitly_detected_or_configured"],
                    }
                    if config.motion.global_rejection.log_rejected_global_events:
                        logs.append_rejection(rejection)
                    session.reject(reason)
                    last_rejection = reason
                    if config.motion.global_rejection.save_debug_snapshot:
                        directory = config.camera.output_directory / "rejections"
                        directory.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(timestamped_output_path(directory, reason, "jpg")), annotated)
                        cleanup_oldest(directory, "*.jpg", config.storage.max_debug_images, logging.getLogger(__name__))
            elif result.state.value == "READY":
                last_rejection = None

            groups_by_track = {group.track_id: group for group in result.groups}
            for group in result.groups:
                if group.newly_confirmed and group.track_id not in recorder.active:
                    recorder.begin(group.track_id, group, frame, annotated, prebuffer.frames(), now=now, measured_fps=measured_fps)
                    session.increment("confirmed_events")
                elif group.track_id in recorder.active:
                    recorder.update(group.track_id, group, annotated, now=now)
            for track_id, event in list(recorder.active.items()):
                if track_id not in groups_by_track:
                    recorder.update(track_id, None, annotated, now=now)
                if recorder.should_finish(event, now):
                    recorder.finish(track_id, now=now)
                    active_directories = {item.directory for item in recorder.active.values()}
                    actions = enforce_retention(events_root, config.retention, active_directories=active_directories)
                    session.data["retention_actions"].extend(actions)

            key = -1
            if show_preview:
                cv2.imshow(WINDOW_TITLE, annotated)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                clean = True
                break
            if key == ord("s"):
                manual_directory = config.camera.output_directory / "manual"
                manual_directory.mkdir(parents=True, exist_ok=True)
                path = timestamped_output_path(manual_directory, "manual-still", "jpg")
                if cv2.imwrite(str(path), annotated):
                    cleanup_oldest(manual_directory, "*.jpg", config.storage.max_event_captures, logging.getLogger(__name__))
                    print(f"Saved manual still: {path.resolve()}")
            if key == ord("e"):
                available = [group for group in result.groups if group.track_id not in recorder.active]
                if available:
                    group = max(available, key=lambda item: item.foreground_pixels)
                    recorder.begin(group.track_id, group, frame, annotated, prebuffer.frames(), now=now, measured_fps=measured_fps)
                    session.increment("confirmed_events")
                    print(f"Forced test event for track {group.track_id}.")
                else:
                    print("No current motion candidate is available to force.")
            if key == ord("r"):
                paths = generate_reports(config)
                print("Rebuilt report: " + ", ".join(str(path.resolve()) for path in paths))

            prebuffer.append(now, annotated)
            if now - last_session_save >= 10.0:
                session.save()
                last_session_save = now
    except KeyboardInterrupt:
        clean = True
        print("Stopping watcher cleanly.")
    except Exception as exc:
        caught_exception = f"{type(exc).__name__}: {exc}"
        session.data["exception_details"].append(caught_exception)
        print(f"Watcher error: {caught_exception}", file=sys.stderr)
    finally:
        now = monotonic()
        recorder.finish_all(now=now)
        capture.release()
        if show_preview:
            cv2.destroyAllWindows()
        actions = enforce_retention(events_root, config.retention)
        session.data["retention_actions"].extend(actions)
        if config.reporting.rebuild_on_clean_shutdown and clean:
            try:
                generate_reports(config)
            except Exception as exc:
                session.data["exception_details"].append(f"Report generation failed: {exc}")
        session.finish(clean=clean)
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
