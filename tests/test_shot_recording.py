"""Regression coverage for the lost September 19 shot; no real hardware."""
from dataclasses import replace
import json
import threading

import pytest

from squirrel_shooter.recording import RecordingError
from squirrel_shooter.camera_service import FramePacket
from test_recording import rig, captured, finished, wait_for
from test_collector_control import control_rig, observation, finish


ATTEMPT = 'aa667246f3fb4ce7b90d2891b1021810'


def test_shot_hold_keeps_actual_frames_before_and_ten_seconds_after_close(rig):
    service = rig.build(target_fps=8)
    rig.camera.push(timestamp=99, value=10)
    link = service.record_firing(ATTEMPT, 'attempt_started', {})
    captured(rig, service, value=20)
    rig.clock.now = 101
    captured(rig, service, value=30)
    wait_for(lambda: service.status()['segments'][0]['last_capture_monotonic'] == 101)
    service.record_firing(ATTEMPT, 'pulse_intent', {})
    rig.clock.now = 101.25
    captured(rig, service, value=40)
    service.record_firing(ATTEMPT, 'pulse_result', {'valve_close_returned': True})
    # No later detector result and no manual recording reason are needed.
    rig.clock.now = 111.5
    captured(rig, service, value=50)
    wait_for(lambda: service.status()['segments'][0]['last_capture_monotonic'] == 111.5)
    rig.clock.now = 112.3
    result = finished(service)
    assert not result['error']
    assert result['segments'][0]['first_capture_monotonic'] == 99
    assert result['segments'][0]['last_capture_monotonic'] == 111.5
    assert {int(frame[0, 0, 0]) for frame in rig.frames} == {10, 20, 30, 40, 50}
    manifest = json.loads((service.directory / link['session_id'] / 'session.json').read_text())
    assert {e['attempt_id'] for e in manifest['firing_events']} == {ATTEMPT}


def test_stalled_video_rejects_automatic_pulse_but_never_obstructs_closure(control_rig):
    r = control_rig(complete=True, auto=True, dry=False)
    class FailedRecorder:
        def record_firing(self, attempt_id, event, details):
            assert not (r.events and r.events[-1] == ('open',))
            raise RecordingError('firing_video_not_current')
    r.control._recording = FailedRecorder()
    r.control.observe(observation())
    result = finish(r.control)
    assert result['outcome'] == 'rejected'
    assert 'requires current video' in result['error']
    assert ('open',) not in r.events
    attempt = r.control.firing_history()['recent'][0]
    assert attempt['recording']['error'] == 'firing_video_not_current'


def test_shot_metadata_link_and_callbacks_only_outside_open_interval(control_rig):
    r = control_rig(complete=True, auto=True, dry=False)
    calls = []
    class Recorder:
        def record_firing(self, attempt_id, event, details):
            transitions = [x[0] for x in r.events if x[0] in ('open', 'close')]
            assert transitions[-1] == 'close'
            calls.append((attempt_id, event))
            return {'session_id': 'recorded-session', 'post_fire_seconds': 10}
    r.control._recording = Recorder()
    r.control.observe(observation())
    assert finish(r.control)['outcome'] == 'fired'
    assert [e for _, e in calls] == ['attempt_started', 'pulse_intent', 'before_open', 'pulse_result', 'attempt_finished']
    attempt = r.control.firing_history()['recent'][0]
    assert attempt['recording']['session_id'] == 'recorded-session'
    assert {i for i, _ in calls} == {attempt['id']}


def test_current_frame_gate_detects_stalled_encoder(rig):
    service = rig.build()
    service.record_firing(ATTEMPT, 'attempt_started', {})
    with pytest.raises(RecordingError, match='not_current'):
        service.record_firing(ATTEMPT, 'pulse_intent', {})
    captured(rig, service)
    service.record_firing(ATTEMPT, 'pulse_intent', {})
    rig.clock.now += .6
    with pytest.raises(RecordingError, match='not_current'):
        service.record_firing(ATTEMPT, 'pulse_intent', {})


def test_late_or_failed_still_does_not_discard_admitted_shot_video(rig, monkeypatch):
    import squirrel_shooter.trip_evidence as trip
    service = rig.build(target_fps=8, observation_max_age_seconds=1.5)
    service.record_firing(ATTEMPT, 'attempt_started', {})
    captured(rig, service)
    packet = rig.camera.push(timestamp=100.125)
    rig.clock.now = 100.125
    entered, release = threading.Event(), threading.Event()
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return dict(kwargs['observation']) if 'observation' in kwargs else dict(args[3])
    monkeypatch.setattr(trip, 'preserve_trip', delayed)
    metadata = dict(confidence=.8, source_monotonic=100.125, source_sequence=packet.sequence)
    errors = []
    def save():
        try:
            service.extend_automatic_recording(event_id='test', reason='squirrel',
                observed_monotonic=100.125, observation=metadata, evidence_packet=packet)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=save)
    worker.start()
    try:
        assert entered.wait(2)
        rig.clock.now = 102
        captured(rig, service, value=90)
    finally:
        release.set()
        worker.join(3)
    assert not errors and service.status()['error'] is None
    assert service.status()['evidence_delay_count'] == 1
    assert service.status()['active']
    def failed(*args, **kwargs):
        raise OSError('still write failed')
    monkeypatch.setattr(trip, 'preserve_trip', failed)
    with pytest.raises(RecordingError, match='still write failed'):
        service.extend_automatic_recording(event_id='test', reason='squirrel',
            observed_monotonic=102, observation=dict(metadata, source_monotonic=102), evidence_packet=packet)
    rig.clock.now = 103
    captured(rig, service, value=100)
    assert service.status()['error'] is None and service.status()['active']


def test_incremental_quota_counts_active_growth_without_rescanning_history(rig, monkeypatch):
    import squirrel_shooter.recording as module
    service = rig.build()
    historical = service.directory / 'historical'
    historical.mkdir()
    (historical / 'old.avi').write_bytes(b'x' * 100)
    service._storage_check()
    service.record_firing(ATTEMPT, 'attempt_started', {})
    captured(rig, service)
    walk, roots = module.os.walk, []
    def checked(root, *args, **kwargs):
        roots.append(root)
        assert root != historical and root != service.directory
        return walk(root, *args, **kwargs)
    monkeypatch.setattr(module.os, 'walk', checked)
    service._storage_check(incremental=True)
    before = service.status()['storage_bytes']
    path = service.directory / service.status()['session_id'] / 'extra.png.pending'
    path.write_bytes(b'x' * 500)
    service._storage_check(incremental=True)
    assert service.status()['storage_bytes'] >= before + 500
    assert roots
