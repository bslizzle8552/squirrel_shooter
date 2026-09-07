from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from squirrel_shooter.camera_service import FramePacket
from squirrel_shooter.recording import RecordingConfig, RecordingError, RecordingService, validate_video


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "timed out waiting for recorder state"


class Clock:
    now = 100.0
    def __call__(self):
        return self.now


class Camera:
    def __init__(self, clock):
        self.clock = clock
        self.packets = deque(maxlen=65)
        self.sequence = 0
        self.condition = threading.Condition()
    def push(self, value=27, *, generation=1, shape=(12, 16, 3), timestamp=None):
        with self.condition:
            self.sequence += 1
            packet = FramePacket(self.sequence, np.full(shape, value, np.uint8), "2026-09-07T12:00:00Z",
                                 self.clock() if timestamp is None else timestamp, generation)
            self.packets.append(packet)
            self.condition.notify_all()
            return packet
    def buffered_frames(self, since_monotonic, *, until_monotonic, copy):
        assert copy is False
        with self.condition:
            return [p for p in self.packets if since_monotonic <= p.received_monotonic <= until_monotonic]
    def wait_for_frame(self, after_sequence, timeout, *, copy):
        assert copy is False
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > after_sequence, min(timeout, .02))
            return self.packets[-1] if self.packets and self.sequence > after_sequence else None


class Writer:
    def __init__(self, path, _codec, _fps, geometry, *, frames, gate=None):
        self.path, self.geometry, self.frames, self.gate = Path(path), geometry, frames, gate
        self.path.write_bytes(b"fake header")
    def isOpened(self):
        return True
    def write(self, image):
        if self.gate:
            self.gate.wait(2)
        assert image.shape[:2] == (self.geometry[1], self.geometry[0])
        self.frames.append(image.copy())
        with self.path.open("ab") as stream:
            stream.write(b"frame")
    def release(self):
        pass


@pytest.fixture
def rig(tmp_path):
    clock, frames = Clock(), []
    camera = Camera(clock)
    services = []
    def build(**overrides):
        gate = overrides.pop("gate", None)
        validator = overrides.pop("validator", lambda path, w, h, n: {"decoded_frames": n, "sha256": "a" * 64, "size_bytes": path.stat().st_size} if n else (_ for _ in ()).throw(RecordingError("empty_output")))
        config = replace(RecordingConfig(enabled=True, minimum_free_megabytes=0), **overrides)
        service = RecordingService(camera, config, tmp_path / str(len(services)), clock=clock,
                                   writer_factory=lambda *args: Writer(*args, frames=frames, gate=gate), validator=validator)
        services.append(service)
        service.start()
        wait_for(lambda: service.status()["ready"])
        return service
    yield SimpleNamespace(clock=clock, frames=frames, camera=camera, build=build, services=services)
    for service in services:
        service.stop()


def captured(rig, service, *, value=27, generation=1, shape=(12, 16, 3)):
    prior = service.status().get("written_frames", 0)
    packet = rig.camera.push(value, generation=generation, shape=shape)
    wait_for(lambda: service.status().get("written_frames", 0) > prior)
    return packet


def finished(service):
    wait_for(lambda: service.status()["pending_sessions"] == 0)
    return service.status()


def test_manual_thirty_uses_backend_deadline_and_reports_missing_tail(rig):
    service = rig.build()
    start = service.record_manual()
    assert start["manual_until"] == 130
    captured(rig, service)
    rig.clock.now = 129.9
    assert service.status()["active"]
    rig.clock.now = 130.1
    result = finished(service)
    assert not result["active"] and result["status"] == "degraded"
    assert any(x["action"] == "missing_tail" for x in result["timeline"])


def test_stop_manual_only_finishes_and_keeps_file(rig):
    service = rig.build()
    session = service.record_manual()["session_id"]
    captured(rig, service)
    service.stop_manual()
    result = finished(service)
    assert result["status"] == "complete" and result["stop_reason"] == "manual_stop"
    manifest = json.loads((service.directory / session / "session.json").read_text())
    assert manifest["segments"][0]["file"] == "segment-0000.avi"
    assert manifest["retention_protected"]


def test_manual_stop_keeps_automatic_reason_and_shared_session(rig):
    service = rig.build()
    session = service.record_manual()["session_id"]
    captured(rig, service)
    automatic = service.extend_automatic_recording(event_id="event-1", visit_id="visit-A", observed_monotonic=100, reason="future_detection")
    assert automatic["session_id"] == session
    stopped = service.stop_manual()
    assert stopped["active"] and stopped["automatic_active"] and stopped["manual_until"] is None
    rig.clock.now = 103.1
    assert not finished(service)["active"]


def test_automatic_repeated_extensions_use_capture_time_and_preserve_identities(rig):
    service = rig.build()
    first = service.extend_automatic_recording(event_id="e1", visit_id="v", observed_monotonic=100, reason="detection")
    captured(rig, service)
    for second in range(1, 5):
        rig.clock.now = 100 + second
        current = service.extend_automatic_recording(event_id="e1", visit_id="v", observed_monotonic=rig.clock.now - .25, reason="detection")
        assert current["session_id"] == first["session_id"]
        assert current["automatic_reasons"][0]["until"] == rig.clock.now + 2.75
    assert current["pending_sessions"] == 1
    assert current["source_identities"][0]["visit_id"] == "v"


@pytest.mark.parametrize("observed", [90, 101, float("nan"), float("inf"), True])
def test_invalid_or_stale_observation_does_not_create_session(rig, observed):
    service = rig.build()
    with pytest.raises(RecordingError):
        service.extend_automatic_recording(event_id="e", reason="detected", observed_monotonic=observed)
    assert service.status()["pending_sessions"] == 0


def test_old_duplicate_or_out_of_order_observation_never_extends(rig):
    service = rig.build()
    service.extend_automatic_recording(event_id="e", reason="detected", observed_monotonic=100)
    for observed in [100, 99.9]:
        with pytest.raises(RecordingError, match="duplicate_or_regressed"):
            service.extend_automatic_recording(event_id="e", reason="detected", observed_monotonic=observed)
    assert service.status()["automatic_reasons"][0]["until"] == 103


def test_preroll_once_and_private_pixels_do_not_change_with_overlays(rig):
    for i in range(24):
        rig.camera.push(10 + i, timestamp=98 + i / 12)
    originals = [p.frame.copy() for p in rig.camera.packets]
    service = rig.build()
    session = service.record_manual()["session_id"]
    wait_for(lambda: service.status().get("written_frames", 0) >= 20)
    before = service.status()["pre_roll_frames"]
    service.extend_automatic_recording(event_id="e", reason="future", observed_monotonic=100)
    assert service.status()["session_id"] == session
    assert service.status()["pre_roll_frames"] == before
    assert all(np.array_equal(p.frame, original) for p, original in zip(rig.camera.packets, originals))
    assert all(np.all(image == image[0, 0]) for image in rig.frames)
    review = rig.frames[0].copy()
    cv2.rectangle(review, (0, 0), (8, 8), (0, 255, 0), 1)
    assert not np.array_equal(review, rig.frames[0])


def test_slow_writer_queue_and_pending_sessions_are_bounded(rig):
    gate = threading.Event()
    service = rig.build(gate=gate, queue_capacity=2, pre_roll_seconds=0)
    service.record_manual()
    rig.camera.push()
    wait_for(lambda: service._segment is not None)
    started = time.monotonic()
    for i in range(25):
        rig.clock.now += .1
        rig.camera.push()
        time.sleep(.007)
    assert time.monotonic() - started < 1
    assert service.status()["queue_depth"] <= 2 and service.status()["queue_dropped"] > 0
    service.stop_manual()
    service.record_manual()
    service.stop_manual()
    with pytest.raises(RecordingError, match="backpressure"):
        service.record_manual()
    gate.set()
    finished(service)


@pytest.mark.parametrize("change", ["generation", "geometry", "duration"])
def test_segment_rotation_preserves_logical_session(rig, change):
    service = rig.build(maximum_segment_seconds=1)
    session = service.record_manual()["session_id"]
    captured(rig, service)
    rig.clock.now += 1.1 if change == "duration" else .2
    captured(rig, service, generation=2 if change == "generation" else 1,
             shape=(18, 24, 3) if change == "geometry" else (12, 16, 3))
    service.stop_manual()
    result = finished(service)
    assert result["session_id"] == session and len(result["segments"]) == 2
    assert [s["segment_index"] for s in result["segments"]] == [0, 1]
    assert result["segments"][0]["finalization_reason"] == ("maximum_segment" if change == "duration" else "camera_generation_or_geometry")


def test_maximum_session_caps_repeated_extension(rig):
    service = rig.build(maximum_session_seconds=30, maximum_segment_seconds=10)
    service.record_manual()
    rig.clock.now = 129
    state = service.extend_automatic_recording(event_id="e", reason="d", observed_monotonic=129)
    assert state["automatic_reasons"][0]["until"] == 130
    rig.clock.now = 130.1
    assert finished(service)["stop_reason"] == "maximum_session"


def test_no_source_frames_is_error_and_metadata_survives(rig):
    service = rig.build()
    identifier = service.record_manual()["session_id"]
    service.stop_manual()
    assert finished(service)["status"] == "error"
    assert (service.directory / identifier / "session.json").is_file()


@pytest.mark.parametrize("reason", ["empty_output", "undecodable_or_incomplete_output", "geometry_mismatch"])
def test_failed_validation_preserves_incomplete_file_and_never_calls_complete(rig, reason):
    def invalid(*args):
        raise RecordingError(reason)
    service = rig.build(validator=invalid)
    service.record_manual()
    captured(rig, service)
    service.stop_manual()
    result = finished(service)
    assert result["status"] == "error" and result["segments"][0]["status"] == "error"
    assert result["segments"][0]["file"].endswith(".incomplete.avi")


def test_real_offline_encoder_and_decoder_validate_clean_geometry(tmp_path):
    path = tmp_path / "clean.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (32, 24))
    for _ in range(6):
        writer.write(np.full((24, 32, 3), 65, np.uint8))
    writer.release()
    result = validate_video(path, 32, 24, 6)
    assert result["decoded_frames"] == 6 and len(result["sha256"]) == 64
    with pytest.raises(RecordingError):
        validate_video(path, 30, 24, 6)
    with pytest.raises(RecordingError):
        validate_video(path, 32, 24, 7)


@pytest.mark.parametrize("contents", [b"", b"not a video"])
def test_real_validator_rejects_empty_and_undecodable_files(tmp_path, contents):
    path = tmp_path / "damaged.avi"
    path.write_bytes(contents)
    with pytest.raises(RecordingError):
        validate_video(path, 32, 24, 1)
    assert path.read_bytes() == contents


def test_shutdown_and_restart_preserve_interrupted_evidence(rig):
    service = rig.build()
    session = service.record_manual()["session_id"]
    captured(rig, service)
    service.stop()
    assert service.status()["status"] == "interrupted"
    manifest = service.directory / session / "session.json"
    data = json.loads(manifest.read_text())
    data["status"] = "capturing"  # Deterministic crash checkpoint, no new process/hardware.
    manifest.write_text(json.dumps(data))
    before = {p.name: p.read_bytes() for p in manifest.parent.glob("*.avi")}
    recovered = RecordingService(rig.camera, service.config, service.directory, clock=rig.clock)
    rig.services.append(recovered)
    recovered.start()
    wait_for(lambda: recovered.status()["ready"])
    assert recovered.status()["recovered_sessions"] == 1
    assert json.loads(manifest.read_text())["status"] == "interrupted"
    assert before == {p.name: p.read_bytes() for p in manifest.parent.glob("*.avi")}


def test_low_disk_and_storage_quota_preserve_all_existing_files(tmp_path):
    directory = tmp_path / "recordings"
    directory.mkdir()
    protected = directory / "protected-training-source.avi"
    protected.write_bytes(b"protected evidence")
    service = RecordingService(Camera(Clock()), RecordingConfig(enabled=True), directory,
                               disk_usage=lambda _: SimpleNamespace(free=0))
    service.start()
    wait_for(lambda: service.status()["last_error"])
    with pytest.raises(RecordingError):
        service.record_manual()
    assert protected.read_bytes() == b"protected evidence"
    service.stop()


def test_no_control_dependencies_or_model_worker_in_recorder():
    import ast
    import squirrel_shooter.recording as module
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(any(word in (name or "") for word in ("valve", "pan_tilt", "manual_control", "auto_fire", "classifier")) for name in imports)


@pytest.mark.parametrize("overrides", [{"queue_capacity": 0}, {"queue_capacity": True}, {"pre_roll_seconds": 30},
                                     {"target_fps": float("nan")}, {"codec": "fake"}, {"maximum_session_seconds": 0},
                                     {"manual_duration_seconds": 601}, {"enabled": "yes"}])
def test_config_rejects_unbounded_or_invalid_recording_settings(overrides):
    with pytest.raises(ValueError):
        RecordingConfig(**overrides)


def test_byte_budget_includes_inflight_frame_and_private_pixels(rig):
    gate = threading.Event()
    service = rig.build(gate=gate, maximum_buffer_megabytes=1, pre_roll_seconds=0)
    service.record_manual()
    packet = rig.camera.push(27, shape=(512, 512, 3))
    wait_for(lambda: service._segment is not None)
    packet.frame[:] = 240  # Simulate a misbehaving injected overlay consumer.
    rig.clock.now += .1
    rig.camera.push(35, shape=(512, 512, 3))
    wait_for(lambda: service.status()["queue_dropped"] > 0)
    assert service.status()["buffered_bytes"] == 512 * 512 * 3
    assert service.status()["buffered_bytes"] <= service.status()["maximum_buffer_bytes"]
    gate.set()
    service.stop_manual()
    assert finished(service)["status"] == "degraded"
    assert np.all(rig.frames[0] == 27)
    assert service.status()["buffered_bytes"] == 0


def test_hung_writer_shutdown_wait_is_bounded_and_recoverable(rig):
    gate = threading.Event()
    entered = threading.Event()
    service = rig.build(shutdown_timeout_seconds=.05)
    class HungWriter(Writer):
        def write(self, image):
            entered.set()
            gate.wait()
            super().write(image)
    service._writer_factory = lambda *args: HungWriter(*args, frames=rig.frames)
    session = service.record_manual()["session_id"]
    rig.camera.push()
    assert entered.wait(3)
    started = time.monotonic()
    try:
        service.stop()
        assert time.monotonic() - started < .25
        assert service.status()["shutdown_timed_out"]
        assert (service.directory / session / "session.json").is_file()
        assert list((service.directory / session).glob("*.incomplete.avi"))
    finally:
        gate.set()
    assert finished(service)["status"] == "interrupted"


@pytest.mark.parametrize("failure", ["write", "release"])
def test_writer_exception_is_terminal_evidence_error(rig, failure):
    service = rig.build()
    class BrokenWriter(Writer):
        def write(self, image):
            if failure == "write":
                raise OSError("injected disk write failure")
            super().write(image)
        def release(self):
            if failure == "release":
                raise OSError("injected disk release failure")
    service._writer_factory = lambda *args: BrokenWriter(*args, frames=rig.frames)
    service.record_manual()
    rig.camera.push()
    if failure == "release":
        wait_for(lambda: service.status().get("written_frames", 0) == 1)
        service.stop_manual()
    state = finished(service)
    assert state["status"] == "error" and "OSError" in state["error"]
    assert list(service.directory.glob("*/*.avi"))
    # A new manual recording is independent of a previous failed file.
    service._writer_factory = lambda *args: Writer(*args, frames=rig.frames)
    assert service.record_manual()["active"]


def test_preparation_error_drains_already_accepted_preroll(rig):
    rig.camera.push(timestamp=99)
    bad = rig.camera.push(timestamp=99.2)
    rig.camera.packets[-1] = replace(bad, frame=bad.frame.reshape(12, 48))
    service = rig.build()
    service.record_manual()
    state = finished(service)
    assert state["status"] == "error" and state["stop_reason"] == "capture_error"
    assert state["buffered_bytes"] == 0


def test_bounded_timeline_preserves_start_reason_and_latest_identity(rig):
    service = rig.build()
    identifier = service.record_manual()["session_id"]
    for n in range(200):
        rig.clock.now += .01
        state = service.extend_automatic_recording(event_id="event", visit_id="visit", reason="future",
                                                    observed_monotonic=rig.clock.now)
    assert state["session_id"] == identifier
    assert len(state["timeline"]) == 128 and state["timeline_omitted"] == 73
    assert state["start_reasons"][0]["action"] == "manual_start_or_extend"
    assert len(state["source_identities"]) == 1
    assert state["source_identities"][0]["observed_monotonic"] == rig.clock.now


def test_source_identity_budget_rejects_without_changing_deadlines(rig):
    service = rig.build()
    for n in range(64):
        service.extend_automatic_recording(event_id=str(n), reason="future", observed_monotonic=100)
    with pytest.raises(RecordingError, match="source_identity_limit"):
        service.extend_automatic_recording(event_id="overflow", reason="future", observed_monotonic=100)
    assert len(service.status()["automatic_reasons"]) == 64


def test_real_service_streams_synthetic_frames_and_validates_sidecar(rig):
    service = rig.build()
    service._writer_factory = cv2.VideoWriter
    service._validator = validate_video
    service.record_manual()
    for _ in range(5):
        captured(rig, service, value=65, shape=(24, 32, 3))
        rig.clock.now += .1
    service.stop_manual()
    state = finished(service)
    assert state["status"] == "complete" and state["written_frames"] == 5
    segment = state["segments"][0]
    assert len(segment["sha256"]) == 64 and segment["decoded_frames"] == 5
    capture = cv2.VideoCapture(str(service.directory / state["session_id"] / segment["file"]))
    try:
        ok, pixels = capture.read()
        assert ok and pixels.shape == (24, 32, 3)
        assert np.max(np.abs(pixels.astype(int) - 65)) <= 2
    finally:
        capture.release()


def test_shared_camera_borrowed_pixels_reject_overlay_mutation(tmp_path):
    from squirrel_shooter.camera_service import CameraService
    from squirrel_shooter.config import CameraConfig
    from test_shared_runtime import ContinuousCapture
    released = threading.Event()
    raw = np.full((24, 32, 3), 45, np.uint8)
    camera = CameraService(CameraConfig(0, 32, 24, 30, tmp_path),
                           capture_factory=lambda _: ContinuousCapture(raw, released),
                           platform_checker=lambda: True, encode_jpeg=False,
                           frame_buffer_seconds=2, frame_buffer_fps=12)
    camera.start()
    try:
        wait_for(lambda: camera.status().frames_received > 0)
        packet = camera.wait_for_frame(-1, copy=False)
        with pytest.raises(ValueError):
            packet.frame[:] = 200
        review = packet.frame.copy()
        cv2.rectangle(review, (1, 1), (10, 10), (0, 255, 0), 2)
        camera.publish_annotated(packet.sequence, review)
        assert np.array_equal(packet.frame, raw)
        assert all(np.array_equal(p.frame, raw) for p in camera.buffered_frames(0, copy=False))
    finally:
        camera.stop()


def test_legacy_fire_adapter_preserves_event_identity_results_and_errors():
    from squirrel_shooter.manual_fire_recording import LegacyFireRecordingAdapter
    calls, event, state = [], object(), {"recording_enabled": True}
    sink = SimpleNamespace(record=lambda value: calls.append(value), status=lambda: state,
                           close=lambda: calls.append("close"))
    adapter = LegacyFireRecordingAdapter(sink)
    adapter.record(event)
    assert adapter.status() is state
    adapter.close()
    assert calls == [event, "close"]
    def broken(_):
        raise OSError("disk error remains separate from accepted hardware result")
    sink.record = broken
    with pytest.raises(OSError):
        adapter.record(event)


def test_old_config_without_recording_section_stays_disabled(tmp_path):
    import yaml
    from conftest import write_test_config
    from squirrel_shooter.config import load_config
    path = write_test_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.pop("recording")
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).recording.enabled is False


def test_periodic_storage_failure_reports_terminal_error_and_keeps_bytes(rig):
    service = rig.build()
    service.record_manual()
    captured(rig, service)
    def unavailable():
        raise OSError("injected storage accounting failure")
    service._storage_check = unavailable
    wait_for(lambda: service.status()["last_error"] is not None)
    state = finished(service)
    assert state["status"] == "error" and not state["ready"] and not state["active"]
    assert state["buffered_bytes"] == 0 and state["queue_depth"] == 0
    assert "storage accounting failure" in state["error"]
    assert list(service.directory.glob("*/*.avi"))
    with pytest.raises(RecordingError):
        service.record_manual()
