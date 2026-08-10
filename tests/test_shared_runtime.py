from __future__ import annotations

import inspect
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from conftest import write_test_config
from squirrel_shooter.app import ApplicationRuntime, DashboardServer, _apply_overrides, build_parser
from squirrel_shooter.camera_service import CameraService, FramePacket
from squirrel_shooter.config import CameraConfig, SharedCameraConfig, load_config
from squirrel_shooter.motion_runtime import MotionProcessingService
from squirrel_shooter.web_dashboard import create_app
import squirrel_shooter.web_dashboard as web_dashboard_module


def wait_until(predicate: object, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:  # type: ignore[operator]
        time.sleep(0.01)


class ContinuousCapture:
    def __init__(self, frame: np.ndarray, released: threading.Event) -> None:
        self.frame = frame
        self.released = released

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frame.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frame.shape[0])
        if prop == cv2.CAP_PROP_FPS:
            return 10.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.released.wait(0.01):
            return False, None
        return True, self.frame.copy()

    def release(self) -> None:
        self.released.set()


def runtime_config(tmp_path: Path):
    config = load_config(
        write_test_config(
            tmp_path,
            motion__processing_width=64,
            motion__startup_warmup__seconds=0,
            motion__startup_warmup__minimum_frames=1,
            motion__global_rejection__enabled=False,
            reporting__directory=(tmp_path / "captures" / "reports").as_posix(),
        )
    )
    return replace(
        config,
        shared_camera=replace(config.shared_camera, reconnect_delay_seconds=0.01, maximum_consecutive_read_failures=1),
        runtime=replace(config.runtime, shutdown_timeout_seconds=2),
    )


def test_shared_runtime_publishes_raw_and_annotated_frames(tmp_path: Path) -> None:
    released = threading.Event()
    raw = np.full((36, 64, 3), 40, dtype=np.uint8)
    service = CameraService(
        CameraConfig(0, 64, 36, 30, tmp_path),
        capture_factory=lambda _: ContinuousCapture(raw, released),
        platform_checker=lambda: True,
    )
    service.start()
    try:
        wait_until(lambda: service.status().frames_received > 0)
        packet = service.wait_for_frame(-1)
        assert packet is not None and np.array_equal(packet.frame, raw)
        annotated = raw.copy()
        cv2.rectangle(annotated, (8, 8), (24, 24), (0, 255, 0), -1)
        assert service.publish_annotated(packet.sequence, annotated)
        assert np.array_equal(service.latest_annotated_frame(), annotated)
        stream_frame = next(service.mjpeg_frames(maximum_fps=10, annotated_only=True))
        assert stream_frame.startswith(b"--frame\r\nContent-Type: image/jpeg")
        jpeg = stream_frame.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n")
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded[16, 16, 1] > 200 and decoded[16, 16, 1] > decoded[16, 16, 0] * 3
        assert service.status().annotated_frames == 1
    finally:
        service.stop()
    assert released.is_set()


def test_shared_camera_keeps_raw_manual_fire_pre_roll(tmp_path: Path) -> None:
    released = threading.Event()
    raw = np.full((36, 64, 3), 80, dtype=np.uint8)
    service = CameraService(
        CameraConfig(0, 64, 36, 30, tmp_path),
        capture_factory=lambda _: ContinuousCapture(raw, released),
        platform_checker=lambda: True,
        encode_jpeg=False,
        frame_buffer_seconds=0.2,
        frame_buffer_fps=20.0,
    )
    service.start()
    try:
        wait_until(lambda: service.status().frames_received >= 3)
        frames = list(service.buffered_frames(0.0))
        metadata = service.buffered_frame_metadata(0.0)
        assert frames
        assert [item.sequence for item in metadata] == [item.sequence for item in frames]
        assert all(packet.frame.shape == raw.shape for packet in frames)
        assert [packet.sequence for packet in frames] == sorted(packet.sequence for packet in frames)
    finally:
        service.stop()


def test_shared_camera_does_not_encode_dashboard_jpegs_without_viewers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released = threading.Event()
    raw = np.full((36, 64, 3), 40, dtype=np.uint8)
    service = CameraService(
        CameraConfig(0, 64, 36, 30, tmp_path),
        capture_factory=lambda _: ContinuousCapture(raw, released),
        platform_checker=lambda: True,
        frame_buffer_seconds=0,
    )
    real_encode = cv2.imencode
    encode_calls = 0

    def counting_encode(*args: object, **kwargs: object):
        nonlocal encode_calls
        encode_calls += 1
        return real_encode(*args, **kwargs)

    monkeypatch.setattr("squirrel_shooter.camera_service.cv2.imencode", counting_encode)
    service.start()
    stream = None
    try:
        wait_until(lambda: service.status().frames_received >= 3)
        packet = service.wait_for_frame(-1)
        assert packet is not None
        service.publish_annotated(packet.sequence, packet.frame)
        time.sleep(0.05)
        assert encode_calls == 0
        assert service.status().dashboard_viewers == 0

        stream = service.mjpeg_frames(maximum_fps=8, annotated_only=True)
        assert next(stream).startswith(b"--frame")
        assert encode_calls == 1
        assert service.status().dashboard_viewers == 1
    finally:
        if stream is not None:
            stream.close()
        service.stop()
    assert service.status().dashboard_viewers == 0


def test_dashboard_viewers_share_one_encoded_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    released = threading.Event()
    raw = np.full((36, 64, 3), 40, dtype=np.uint8)
    service = CameraService(
        CameraConfig(0, 64, 36, 30, tmp_path),
        capture_factory=lambda _: ContinuousCapture(raw, released),
        platform_checker=lambda: True,
        frame_buffer_seconds=0,
    )
    real_encode = cv2.imencode
    encode_calls = 0

    def counting_encode(*args: object, **kwargs: object):
        nonlocal encode_calls
        encode_calls += 1
        return real_encode(*args, **kwargs)

    monkeypatch.setattr("squirrel_shooter.camera_service.cv2.imencode", counting_encode)
    service.start()
    first = second = None
    try:
        wait_until(lambda: service.status().frames_received > 0)
        packet = service.wait_for_frame(-1)
        assert packet is not None
        service.publish_annotated(packet.sequence, packet.frame)
        first = service.mjpeg_frames(maximum_fps=8, annotated_only=True)
        first_frame = next(first)
        second = service.mjpeg_frames(maximum_fps=8, annotated_only=True)
        second_frame = next(second)
        assert first_frame == second_frame
        assert encode_calls == 1
        assert service.status().dashboard_viewers == 2
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
        service.stop()


def test_manual_pre_roll_copies_at_target_rate_without_jpeg_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released = threading.Event()
    service = CameraService(
        CameraConfig(0, 64, 36, 30, tmp_path),
        capture_factory=lambda _: ContinuousCapture(np.zeros((36, 64, 3), dtype=np.uint8), released),
        platform_checker=lambda: True,
        encode_jpeg=False,
        frame_buffer_seconds=1.0,
        frame_buffer_fps=10.0,
    )
    encode_calls = 0

    def unexpected_encode(*_args: object, **_kwargs: object):
        nonlocal encode_calls
        encode_calls += 1
        raise AssertionError("idle pre-roll must not JPEG-encode")

    monkeypatch.setattr("squirrel_shooter.camera_service.cv2.imencode", unexpected_encode)
    service.start()
    try:
        wait_until(lambda: service.status().frames_received >= 25)
        status = service.status()
        assert status.pre_roll_frames_encoded == 0
        assert status.pre_roll_frames_copied < status.frames_received / 3
        assert 1 <= status.pre_roll_frames_buffered <= 5
        assert status.pre_roll_target_fps == 10.0
        assert encode_calls == 0
    finally:
        service.stop()


def test_motion_processing_samples_fast_camera_at_configured_rate(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)

    class FastCamera:
        def __init__(self) -> None:
            self.sequence = 0

        def wait_for_frame(self, _after_sequence: int):
            self.sequence += 1
            return SimpleNamespace(sequence=self.sequence)

        def status(self):
            return SimpleNamespace(read_failures=0)

    motion = MotionProcessingService(FastCamera(), config)  # type: ignore[arg-type]
    processed: list[float] = []
    motion._process_packet = lambda _packet: processed.append(time.monotonic())  # type: ignore[method-assign]
    motion._finalize = lambda *, clean: None  # type: ignore[method-assign]
    thread = threading.Thread(target=motion._run)
    thread.start()
    time.sleep(0.36)
    motion._stop_event.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert 3 <= len(processed) <= 5
    assert all(later - earlier >= 0.08 for earlier, later in zip(processed, processed[1:]))


def test_headless_motion_skips_idle_annotation_until_a_viewer_connects(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    config = replace(config, runtime=replace(config.runtime, headless=True))

    class Camera:
        has_dashboard_viewers = False

        def __init__(self) -> None:
            self.published = 0

        def status(self) -> SimpleNamespace:
            return SimpleNamespace(fps=10.0, reported_fps=15.0, width=64, height=36, read_failures=0)

        def publish_annotated(self, _sequence: int, _frame: np.ndarray) -> None:
            self.published += 1

    camera = Camera()
    motion = MotionProcessingService(camera, config)  # type: ignore[arg-type]
    frame = np.zeros((36, 64, 3), dtype=np.uint8)

    motion._process_packet(FramePacket(1, frame, "stamp", 1.0))

    assert motion.status().annotations_rendered == 0
    assert motion.status().idle_annotations_skipped == 1
    assert camera.published == 0

    camera.has_dashboard_viewers = True
    motion._process_packet(FramePacket(2, frame, "stamp", 1.1))

    assert motion.status().annotations_rendered == 1
    assert camera.published == 1


def test_application_runtime_retains_only_the_pre_roll_window_in_memory(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    runtime = ApplicationRuntime(config)

    assert runtime.camera.frame_buffer_seconds == pytest.approx(
        config.manual_control.recording.pre_roll_seconds
        + config.manual_control.fire_pulse_seconds
        + max(1.0, config.shared_camera.consumer_wait_timeout_seconds)
    )
    assert runtime.camera.status().pre_roll_target_fps == config.manual_control.recording.target_fps


def test_live_stream_holds_last_seen_box_during_tracker_gap(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    motion = MotionProcessingService(SimpleNamespace(), config)  # type: ignore[arg-type]
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    group = SimpleNamespace(
        track_id=7,
        bounding_box=(20, 20, 30, 25),
        provisional_category="small_animal_candidate",
    )

    motion._add_live_box_holds(frame, (group,), 1.0)
    held = motion._add_live_box_holds(frame, (), 1.1)
    expired = motion._add_live_box_holds(frame, (), 2.0)

    assert held[20, 20].any()
    assert not expired[20, 20].any()


def test_night_mode_finishes_active_clip_and_pauses_until_color_returns(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)

    class Classifier:
        def __init__(self) -> None:
            self.pause_states: list[bool] = []

        def set_paused(self, paused: bool) -> None:
            self.pause_states.append(paused)

    class Recorder:
        def __init__(self) -> None:
            self.active = {7: SimpleNamespace(event_id="day-event")}
            self.finished = 0
            self.begin_calls = 0

        def finish_all(self, *, now: float, notes: str) -> list[dict[str, object]]:
            del now
            self.finished += 1
            self.active.clear()
            return [{"event_id": "day-event", "end_timestamp": "now", "notes": notes}]

        def begin(self, *_args: object, **_kwargs: object) -> object:
            self.begin_calls += 1
            raise AssertionError("night mode must not begin an event")

    classifier = Classifier()
    recorder = Recorder()
    motion = MotionProcessingService(
        SimpleNamespace(),
        config,
        classifier_service=classifier,  # type: ignore[arg-type]
    )
    motion._recorder = recorder  # type: ignore[assignment]
    motion._active_events = 1
    dummy_result = SimpleNamespace()
    motion._prebuffer.append(0.0, np.zeros((20, 20, 3), dtype=np.uint8), dummy_result, 10.0)  # type: ignore[arg-type]
    night_result = SimpleNamespace(
        global_motion=SimpleNamespace(reason="probable_ir_mode_switch", colorfulness=0.0),
        groups=(SimpleNamespace(track_id=9),),
    )

    motion._update_night_mode(night_result, 1.0)  # type: ignore[arg-type]
    motion._handle_events(SimpleNamespace(), night_result, lambda: np.zeros((20, 20, 3), dtype=np.uint8), 1.0, 10.0)  # type: ignore[arg-type]

    assert motion._night_mode_paused is True
    assert motion._night_mode_evidence == "probable_ir_mode_switch"
    assert classifier.pause_states == [False, True]
    assert recorder.finished == 1 and recorder.begin_calls == 0
    assert motion._active_events == 0 and len(motion._prebuffer) == 0
    assert motion.recent_events()[0]["notes"] == "night vision pause"

    day_result = SimpleNamespace(global_motion=SimpleNamespace(reason=None, colorfulness=20.0), groups=())
    for index in range(config.night_mode.exit_consecutive_frames):
        motion._update_night_mode(day_result, 2.0 + index)  # type: ignore[arg-type]

    assert motion._night_mode_paused is False
    assert motion._night_mode_evidence == "sustained_color_return"
    assert classifier.pause_states[-1] is False


def test_sustained_monochrome_frames_pause_a_process_started_after_dark(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)

    class Classifier:
        def __init__(self) -> None:
            self.paused = False

        def set_paused(self, paused: bool) -> None:
            self.paused = paused

    classifier = Classifier()
    motion = MotionProcessingService(
        SimpleNamespace(),
        config,
        classifier_service=classifier,  # type: ignore[arg-type]
    )
    mono_result = SimpleNamespace(global_motion=SimpleNamespace(reason=None, colorfulness=0.0))

    for index in range(config.night_mode.enter_consecutive_frames):
        motion._update_night_mode(mono_result, float(index))  # type: ignore[arg-type]

    assert motion._night_mode_paused is True
    assert motion._night_mode_evidence == "sustained_monochrome_frames"
    assert classifier.paused is True


def test_camera_failure_is_reported_and_reconnects_safely(tmp_path: Path) -> None:
    opens = 0
    released = threading.Event()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    class FailedCapture(ContinuousCapture):
        def read(self) -> tuple[bool, None]:
            return False, None

    def factory(_: CameraConfig):
        nonlocal opens
        opens += 1
        return FailedCapture(frame, threading.Event()) if opens == 1 else ContinuousCapture(frame, released)

    shared = SharedCameraConfig(True, 1, 0.01, 0.1, 1.0)
    service = CameraService(CameraConfig(0, 32, 24, 30, tmp_path), shared_settings=shared, capture_factory=factory, platform_checker=lambda: True)
    service.start()
    try:
        wait_until(lambda: service.status().online and service.status().camera_open_count >= 2)
        status = service.status()
        assert status.read_failures >= 1
        assert status.reconnects >= 1
        assert opens == 2
    finally:
        service.stop()


def test_motion_and_dashboard_share_exactly_one_camera_open(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    released = threading.Event()
    opens = 0
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def factory(_: CameraConfig) -> ContinuousCapture:
        nonlocal opens
        opens += 1
        return ContinuousCapture(frame, released)

    camera = CameraService(
        config.camera,
        shared_settings=config.shared_camera,
        capture_factory=factory,
        platform_checker=lambda: True,
        jpeg_quality=config.dashboard.jpeg_quality,
    )
    motion = MotionProcessingService(camera, config)
    runtime = ApplicationRuntime(config, camera=camera, motion=motion)
    runtime.start()
    runtime.start()
    try:
        wait_until(lambda: motion.status().frames_processed > 0 and camera.status().annotated_frames > 0)
        dashboard = create_app(
            app_config=config,
            camera_service=camera,
            motion_service=motion,
            start_camera=False,
            start_vision=False,
            temperature_reader=lambda: 44.0,
        )
        dashboard.config.update(TESTING=True)
        status = dashboard.test_client().get("/api/status")
        events = dashboard.test_client().get("/api/events")
        stream = dashboard.test_client().get("/video_feed", buffered=False)
        assert status.status_code == events.status_code == 200
        assert status.json["camera"]["camera_open_count"] == 1
        assert status.json["camera"]["dashboard_viewers"] == 0
        assert "pre_roll_target_fps" in status.json["camera"]
        assert "pre_roll_frames_copied" in status.json["camera"]
        assert "capture_thread_cpu_percent" in status.json["camera"]
        assert status.json["detector"]["frames_processed"] > 0
        assert status.json["detector"]["target_fps"] == config.motion.target_fps
        assert "detector_average_ms" in status.json["detector"]
        assert "motion_thread_cpu_percent" in status.json["detector"]
        assert status.json["application_mode"] == "shared-camera-motion-watch"
        assert next(stream.response).startswith(b"--frame")
        stream.close()
        motion.rebuild_report()
        assert dashboard.test_client().get("/reports/latest").status_code == 200
        assert opens == 1
    finally:
        runtime.stop()
    assert released.is_set()
    assert not camera.status().thread_alive
    assert not motion.status().thread_alive


def test_dashboard_never_constructs_or_reads_a_video_capture() -> None:
    source = inspect.getsource(web_dashboard_module)
    assert "cv2.VideoCapture" not in source
    assert "capture.read(" not in source


def test_combined_app_cli_uses_config_and_allows_safe_overrides(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    defaults = _apply_overrides(config, build_parser().parse_args([]))
    overridden = _apply_overrides(config, build_parser().parse_args(["--headless", "--no-dashboard", "--port", "5050"]))
    assert defaults.dashboard.host == "0.0.0.0" and defaults.dashboard.port == 5000
    assert overridden.runtime.headless is True
    assert overridden.dashboard.enabled is False and overridden.dashboard.port == 5050


def test_motion_exception_is_reported_without_opening_another_camera(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    released = threading.Event()
    opens = 0

    def factory(_: CameraConfig) -> ContinuousCapture:
        nonlocal opens
        opens += 1
        return ContinuousCapture(np.zeros((48, 64, 3), dtype=np.uint8), released)

    class FailingDetector:
        def process(self, frame: np.ndarray, *, now: float):
            del frame, now
            raise RuntimeError("synthetic shared detector failure")

    camera = CameraService(config.camera, shared_settings=config.shared_camera, capture_factory=factory, platform_checker=lambda: True)
    motion = MotionProcessingService(camera, config, detector=FailingDetector())  # type: ignore[arg-type]
    runtime = ApplicationRuntime(config, camera=camera, motion=motion)
    runtime.start()
    try:
        wait_until(lambda: motion.status().state == "ERROR" and camera.status().annotated_frames > 0)
        assert motion.status().thread_alive is True
        assert "synthetic shared detector failure" in str(motion.status().last_error)
        assert camera.status().online is True and opens == 1
    finally:
        runtime.stop()


def test_dashboard_request_exception_does_not_stop_shared_runtime(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    released = threading.Event()
    camera = CameraService(
        config.camera,
        shared_settings=config.shared_camera,
        capture_factory=lambda _: ContinuousCapture(np.zeros((48, 64, 3), dtype=np.uint8), released),
        platform_checker=lambda: True,
    )
    motion = MotionProcessingService(camera, config)
    runtime = ApplicationRuntime(config, camera=camera, motion=motion)
    runtime.start()
    try:
        wait_until(lambda: motion.status().frames_processed > 0)
        dashboard = create_app(app_config=config, camera_service=camera, motion_service=motion, start_camera=False, start_vision=False)

        def fail_request():
            raise RuntimeError("synthetic dashboard failure")

        dashboard.add_url_rule("/synthetic-failure", view_func=fail_request)
        dashboard.config.update(TESTING=False, PROPAGATE_EXCEPTIONS=False)
        response = dashboard.test_client().get("/synthetic-failure")
        assert response.status_code == 500
        assert camera.status().thread_alive and motion.status().thread_alive
    finally:
        runtime.stop()


def test_threaded_http_dashboard_runs_concurrently_with_motion(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    released = threading.Event()
    camera = CameraService(
        config.camera,
        shared_settings=config.shared_camera,
        capture_factory=lambda _: ContinuousCapture(np.zeros((48, 64, 3), dtype=np.uint8), released),
        platform_checker=lambda: True,
    )
    motion = MotionProcessingService(camera, config)
    runtime = ApplicationRuntime(config, camera=camera, motion=motion)
    runtime.start()
    server: DashboardServer | None = None
    try:
        wait_until(lambda: motion.status().frames_processed > 0)
        dashboard = create_app(app_config=config, camera_service=camera, motion_service=motion, start_camera=False, start_vision=False)
        server = DashboardServer(dashboard, "127.0.0.1", 0)
        server.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/status", timeout=2) as response:
            body = response.read()
        assert response.status == 200
        assert b"shared-camera-motion-watch" in body
        assert camera.status().camera_open_count == 1
        assert motion.status().thread_alive and server.alive
    finally:
        if server is not None:
            server.stop()
        runtime.stop()
