"""Deterministic receipt-time replay; no camera, model or physical device."""
import json
import numpy as np
import pytest

from squirrel_shooter.camera_service import FramePacket
from squirrel_shooter.recording import RecordingConfig, RecordingService
from test_recording import Clock, Camera, Writer


def replay(tmp_path, times, *, drain_every=1, pre_roll=0):
    clock, frames = Clock(), []
    service = RecordingService(Camera(clock), RecordingConfig(enabled=True, minimum_free_megabytes=0,
        pre_roll_seconds=pre_roll, queue_capacity=2), tmp_path, clock=clock,
        writer_factory=lambda *args: Writer(*args, frames=frames),
        validator=lambda p, w, h, n: {"decoded_frames": n})
    service._ready = True
    service.record_manual()
    session = service._active
    session.prepared = True
    originals = []
    def drain():
        while session.pre_packets or not service._queue.empty():
            pre = bool(session.pre_packets)
            payload = session.pre_packets.popleft() if pre else service._queue.get_nowait()
            service._write_packet(payload)
            session.queued -= 1
            if service._held_packet is not payload:
                service._buffered_bytes -= payload[-1].nbytes
            if not pre:
                service._queue.task_done()
    for i, t in enumerate(times):
        image = np.full((12, 16, 3), i % 255, np.uint8)
        originals.append((image, image.copy()))
        packet = FramePacket(i + 1, image, "wall", 100 + t, 1)
        service._admit(session, packet, pre_roll=t < 0)
        assert service._queue.qsize() <= 2
        assert service._buffered_bytes <= service.config.maximum_buffer_megabytes * 1024**2
        if (i + 1) % drain_every == 0:
            drain()
    drain()
    clock.now = 130
    service._expire(clock.now)
    service._close_segment("deadlines_expired")
    assert all(np.array_equal(a, b) for a, b in originals)
    assert service._buffered_bytes == 0
    return service, session, frames


@pytest.mark.parametrize("kind", ["irregular", "burst", "gap", "slow_queue", "preroll"])
def test_thirty_second_receipt_timeline_survives_sampling_and_drops(tmp_path, kind):
    # Camera bursts can average 20 Hz while leaving entire 12 Hz slots empty.
    times = [i * .2 + offset for i in range(150) for offset in (0, .02, .04, .06)]
    if kind == "irregular":
        times = [i * .101 for i in range(298)]
    if kind == "gap":
        times = [t for t in times if not 10 < t < 12]
    if kind == "preroll":
        times = [-2 + i / 12 for i in range(24)] + times
    service, session, frames = replay(tmp_path, times, drain_every=25 if kind == "slow_queue" else 1,
                                      pre_roll=2 if kind == "preroll" else 0)
    segment = session.segments[0]
    assert abs(len(frames) / 12 - (130 - segment["first_capture_monotonic"])) <= 1 / 12
    # A rejected tail is held explicitly through the deadline, with true source
    # span preserved separately. It must not be misreported as new observations.
    assert abs(segment["playback_duration_seconds"] - (130 - segment["first_capture_monotonic"])) <= 1 / 12
    assert segment["repeated_presentation_frames"] > 0
    assert segment["unique_source_frames"] + segment["repeated_presentation_frames"] == len(frames)
    rows = [json.loads(line) for line in (service.directory / session.id / segment["presentation_file"]).read_text().splitlines()]
    assert len(rows) == len(frames)
    unique = [r for r in rows if not r["repeated"]]
    assert all(b["source_sequence"] > a["source_sequence"] for a, b in zip(unique, unique[1:]))
    for a, b in zip(rows, rows[1:]):
        assert b["source_monotonic"] >= a["source_monotonic"]
        if b["repeated"]:
            assert b["source_sequence"] == a["source_sequence"]
            assert np.array_equal(frames[a["output_index"]], frames[b["output_index"]])
    if kind == "slow_queue":
        assert session.queue_dropped > 0
        assert segment["status"] == "degraded"
    if kind == "gap":
        assert segment["maximum_source_gap_seconds"] == 2
        assert segment["status"] == "degraded"


def test_full_queue_retains_old_packets_and_records_rejected_identity(tmp_path):
    service, session, _ = replay(tmp_path, [0, .1, .2, .3, 29.9], drain_every=4)
    drops = [e for e in session.timeline if e["action"] == "admission_drop"]
    assert [(e["sequence"], e["cause"]) for e in drops] == [(3, "queue_full"), (4, "queue_full")]
    segment = session.segments[0]
    rows = [json.loads(line) for line in (service.directory / session.id / segment["presentation_file"]).read_text().splitlines()]
    assert [r["source_sequence"] for r in rows if not r["repeated"]] == [1, 2, 5]


def test_large_missing_tail_is_held_but_remains_degraded(tmp_path):
    service, session, frames = replay(tmp_path, [0, .1])
    segment = session.segments[0]
    assert len(frames) == 360
    assert segment["unique_source_frames"] == 2
    assert segment["repeated_presentation_frames"] == 358
    assert segment["status"] == "degraded"
    assert segment["capture_span_seconds"] == pytest.approx(.1)


def test_delayed_first_source_is_not_invented(tmp_path):
    _, session, frames = replay(tmp_path, [2, 2.1, 29.9])
    assert len(frames) == 336
    assert session.segments[0]["first_capture_monotonic"] == 102
    assert session.segments[0]["status"] == "degraded"


def test_normal_segment_rotation_does_not_report_missing_session_head(tmp_path):
    from dataclasses import replace
    clock = Clock()
    service = RecordingService(None, replace(RecordingConfig(enabled=True, minimum_free_megabytes=0),
        maximum_segment_seconds=1), tmp_path, clock=clock,
        writer_factory=lambda *args: Writer(*args, frames=[]), validator=lambda *args: {})
    service.directory.mkdir(exist_ok=True)
    service._ready = True
    service.record_manual()
    session = service._active
    for i in range(24):
        image = np.zeros((12, 16, 3), np.uint8)
        service._buffered_bytes += image.nbytes
        service._write_packet((session, i+1, 100+i/12, 'wall', 1, image))
    session.ended_monotonic = 102
    service._close_segment('manual_stop')
    assert len(session.segments) == 2
    assert all(s['status'] == 'complete' and s['written_frames'] == 12 for s in session.segments)
    assert service._buffered_bytes == 0
