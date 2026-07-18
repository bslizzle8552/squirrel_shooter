"""Local preview harness for the Squirrel Squirter dashboard on any host.

Builds the real Flask app with duck-typed fake camera/vision services, a temp
output directory full of synthetic event snapshots/clips, and a generated MJPEG
feed — no Raspberry Pi, camera, or real config/data directories required.

    python scripts/preview_dashboard.py [--host 127.0.0.1] [--port 5000]
    npm run dev -- --port 5001
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ["SQUIRREL_DEMO"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from squirrel_shooter.camera_service import CameraStatus  # noqa: E402
from squirrel_shooter.config import load_config  # noqa: E402
from squirrel_shooter.vision_service import VisionStatus  # noqa: E402
from squirrel_shooter.web_dashboard import create_app  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic garden imagery
# ---------------------------------------------------------------------------

def _garden_background(width: int, height: int, seed: int = 7) -> np.ndarray:
    """Static garden backdrop: sky band, hedge, lawn with noise texture."""

    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * 0.26)
    # sky
    frame[:horizon] = (120, 105, 70)  # BGR: soft warm daylight haze
    # hedge row
    hedge = np.zeros((int(height * 0.24), width, 3), dtype=np.uint8)
    hedge[:] = (38, 88, 40)
    hedge_noise = rng.integers(-16, 16, hedge.shape, dtype=np.int16)
    hedge = np.clip(hedge.astype(np.int16) + hedge_noise, 0, 255).astype(np.uint8)
    frame[horizon : horizon + hedge.shape[0]] = hedge
    # lawn
    lawn_top = horizon + hedge.shape[0]
    lawn = np.zeros((height - lawn_top, width, 3), dtype=np.uint8)
    lawn[:] = (30, 105, 42)
    lawn_noise = rng.integers(-14, 14, lawn.shape, dtype=np.int16)
    lawn = np.clip(lawn.astype(np.int16) + lawn_noise, 0, 255).astype(np.uint8)
    frame[lawn_top:] = lawn
    # a few darker bush blobs on the lawn
    for _ in range(5):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(lawn_top + 10, height - 10))
        radius = int(rng.integers(18, 46))
        cv2.ellipse(frame, (cx, cy), (radius, int(radius * 0.55)), 0, 0, 360, (26, 82, 34), -1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _draw_detection(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int] = (36, 165, 245),  # BGR amber, matches the UI accent
) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _stamp(frame: np.ndarray, text: str) -> None:
    cv2.putText(
        frame, text, (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 239, 243), 1, cv2.LINE_AA,
    )


def _snapshot_image(event_id: str, when: datetime, seed: int) -> np.ndarray:
    frame = _garden_background(640, 360, seed=seed)
    rng = np.random.default_rng(seed * 31 + 5)
    box = (
        int(rng.integers(60, 420)),
        int(rng.integers(140, 260)),
        int(rng.integers(70, 130)),
        int(rng.integers(50, 90)),
    )
    _draw_detection(frame, box, "small animal candidate")
    _stamp(frame, f"{when.strftime('%Y-%m-%d %H:%M:%S')}  {event_id}")
    return frame


def _classifier_input(event_id: str, when: datetime, seed: int) -> np.ndarray:
    frame = _garden_background(384, 384, seed=seed * 13 + 3)
    _draw_detection(frame, (110, 150, 150, 110), "classifier input")
    _stamp(frame, when.strftime("%H:%M:%S"))
    return frame


def _write_clip(path: Path, when: datetime, seed: int) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (320, 180))
    if not writer.isOpened():
        return False
    base = _garden_background(320, 180, seed=seed)
    for index in range(16):
        frame = base.copy()
        x = 40 + index * 12
        _draw_detection(frame, (x, 95, 56, 40), "motion")
        _stamp(frame, when.strftime("%H:%M:%S"))
        writer.write(frame)
    writer.release()
    return path.is_file()


# ---------------------------------------------------------------------------
# Demo data generation (all inside a temp directory; real config is untouched)
# ---------------------------------------------------------------------------

REVIEW_SPECS = [
    # (classification_status, display_label, suggestion label/conf, top label/conf, error)
    ("review", "Unknown", ("car", 0.52), ("car", 0.52), None),
    ("review", "Unknown", ("person", 0.44), ("person", 0.44), None),
    ("review", "Unknown", ("car", 0.38), ("car", 0.38), None),
    ("review", "Unknown", ("person", 0.31), ("person", 0.31), None),
    ("known", "Car", None, ("car", 0.91), None),
    ("known", "Person", None, ("person", 0.87), None),
    ("unknown", "Unknown", None, ("bicycle", 0.22), None),
    ("unknown", "Unknown", None, None, None),
    ("unclassified", "Classification unavailable", None, None, "Model inference timed out after 3000 ms"),
    ("false_positive", "False Positive", None, ("person", 0.19), None),
]


def build_demo_tree(root: Path) -> list[dict[str, Any]]:
    """Create captures/, events/<date>/<id>/ and loose capture images."""

    output = root / "captures"
    events_root = output / "events" / "2026-07-16"
    events_root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "classifier.jsonl").write_text("", encoding="utf-8")
    (root / "reports" / "latest-report.html").write_text(
        "<h1>Squirrel Squirter demo report</h1><p>Synthetic preview data.</p>", encoding="utf-8"
    )

    now = datetime.now().astimezone().replace(microsecond=0)
    events: list[dict[str, Any]] = []
    for index, (status, display, suggestion, top, error) in enumerate(REVIEW_SPECS):
        event_id = f"event-{index + 1:03d}"
        when = now - timedelta(minutes=9 * (len(REVIEW_SPECS) - index))
        directory = events_root / event_id
        directory.mkdir(parents=True, exist_ok=True)
        snapshot_path = directory / "snapshot.jpg"
        clip_path = directory / "clip.avi"
        cv2.imwrite(str(snapshot_path), _snapshot_image(event_id, when, seed=index + 11))
        if not _write_clip(clip_path, when, seed=index + 3):
            clip_path = None  # UI hides clip affordances when no clip exists
        cv2.imwrite(str(directory / "classifier-input.jpg"), _classifier_input(event_id, when, seed=index + 7))

        event_payload = {
            "status": "complete",
            "event_id": event_id,
            "start_timestamp": when.isoformat(),
            "duration": round(4.0 + index * 0.7, 1),
            "provisional_category": "small_animal_candidate",
            "movement_attributes": ["coherent_travel"],
            "snapshot_path": str(snapshot_path),
            "clip_path": str(clip_path) if clip_path else None,
        }
        (directory / "event.json").write_text(json.dumps(event_payload, indent=2), encoding="utf-8")

        record = {
            "schema_version": 2,
            "item_id": event_id,
            "event_id": event_id,
            "classifier_timestamp": (when + timedelta(seconds=6)).isoformat(),
            "submitted_at": when.isoformat(),
            "frame_number": 1,
            "model": "mobilenet-ssd (demo)",
            "detections": [],
            "top_label": top[0] if top else None,
            "top_confidence": top[1] if top else None,
            "review_suggestion_label": suggestion[0] if suggestion else None,
            "review_suggestion_confidence": suggestion[1] if suggestion else None,
            "classification_status": status,
            "display_label": display,
            "label_source": "automatic" if status == "known" else ("human" if status == "false_positive" else None),
            "latency_ms": round(38.0 + index * 6.5, 1),
            "error": error,
            "reviewed_at": None,
        }
        (directory / "classification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        events.append(event_payload)

    # loose standalone captures for the /captures archive
    for index in range(6):
        when = now - timedelta(minutes=25 * (index + 1))
        frame = _garden_background(640, 360, seed=100 + index)
        _stamp(frame, when.strftime("%Y-%m-%d %H:%M:%S"))
        cv2.imwrite(str(output / f"event-{when.strftime('%Y%m%d-%H%M%S')}-demo{index}.jpg"), frame)

    events.reverse()  # newest first, matching VisionService.recent_events()
    return events


def write_demo_config(root: Path) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["camera"]["output_directory"] = (root / "captures").as_posix()
    raw["motion"]["debug_outputs"]["directory"] = (root / "debug").as_posix()
    raw["logging"]["directory"] = (root / "logs").as_posix()
    raw["reporting"]["directory"] = (root / "reports").as_posix()
    raw["classifier"]["evidence_directory"] = (root / "captures" / "classifier").as_posix()
    path = root / "demo-config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Duck-typed fake services (same shapes as CameraService / VisionService)
# ---------------------------------------------------------------------------

class PreviewCameraService:
    def __init__(self) -> None:
        self._started = time.monotonic()

    def start(self) -> None:  # pragma: no cover - parity no-op
        pass

    def stop(self, timeout: float = 3.0) -> None:  # pragma: no cover - parity no-op
        del timeout

    def status(self) -> CameraStatus:
        now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        fps = 9.6 + math.sin(time.monotonic() / 3.0) * 0.4
        return CameraStatus(
            online=True,
            width=1280,
            height=720,
            fps=fps,
            error=None,
            last_frame_at=now_iso,
            last_frame_age_seconds=0.12,
            frames_received=int((time.monotonic() - self._started) * 10),
            thread_alive=True,
            reported_fps=10.0,
            read_failures=0,
            reconnects=0,
            camera_open_count=1,
            annotated_frames=int((time.monotonic() - self._started) * 5),
            last_annotated_at=now_iso,
            annotated_frame_age_seconds=0.2,
        )


class PreviewVisionService:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self._started = time.monotonic()
        self._frames = 0
        self._background = _garden_background(640, 360, seed=42)
        self._lock = threading.Lock()

    def start(self) -> None:  # pragma: no cover - parity no-op
        pass

    def stop(self, timeout: float = 3.0) -> None:  # pragma: no cover - parity no-op
        del timeout

    def status(self) -> VisionStatus:
        now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        last_event = self._events[0]["start_timestamp"] if self._events else None
        return VisionStatus(
            state="READY",
            enabled=True,
            processing_fps=4.8 + math.sin(time.monotonic() / 2.5) * 0.5,
            blob_count=1,
            persistence_count=3,
            frames_processed=self._frames,
            candidates_seen=self._frames // 40,
            accepted_events=len(self._events),
            rejected_events=2,
            snapshots_saved=len(self._events),
            last_detector_update=now_iso,
            last_detector_age_seconds=0.18,
            last_event=last_event,
            last_snapshot=last_event,
            last_error=None,
            thread_alive=True,
            capture_directory_writable=True,
        )

    def recent_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def mjpeg_frames(self) -> Iterator[bytes]:
        while True:
            t = time.monotonic()
            frame = self._background.copy()
            # one detection box drifting back and forth across the lawn
            cx = int(320 + 200 * math.sin(t * 0.9))
            cy = int(235 + 30 * math.sin(t * 1.7))
            _draw_detection(frame, (cx - 55, cy - 35, 110, 70), "small animal candidate")
            _stamp(frame, datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S") + "  DEMO")
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                with self._lock:
                    self._frames += 1
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.2)  # ~5 FPS, easy on the host CPU


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview the Squirrel Squirter dashboard with synthetic demo data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    demo_root = Path(tempfile.mkdtemp(prefix="squirrel-preview-"))
    events = build_demo_tree(demo_root)
    config_path = write_demo_config(demo_root)
    config = load_config(config_path)

    app = create_app(
        config_path,
        camera_service=PreviewCameraService(),  # type: ignore[arg-type]
        vision_service=PreviewVisionService(events),  # type: ignore[arg-type]
        temperature_reader=lambda: 46.8 + math.sin(time.monotonic() / 8.0) * 0.6,
        start_camera=False,
        start_vision=False,
    )
    del config  # config object is held by the app extensions

    print(f"Squirrel Squirter demo preview")
    print(f"  URL:      http://{args.host}:{args.port}/")
    print(f"  Demo dir: {demo_root} (synthetic data only; real config untouched)")
    print(f"  Stop:     Ctrl+C")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
