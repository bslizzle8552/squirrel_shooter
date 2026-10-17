"""Shot-only field policy; all controls and camera frames are synthetic."""
from dataclasses import asdict
from types import SimpleNamespace
import inspect

import pytest

from squirrel_shooter.collector_app import CollectorRuntime
from squirrel_shooter.collector_policy import AutomaticRecordingConfig, SquirrelRecordingObserver
from squirrel_shooter.watch_zone import WatchZoneConfig
from test_collector_control import control_rig, observation, finish
from test_recording import rig, captured, finished


def test_disabled_detection_recording_still_delivers_every_detection_to_control(rig):
    recorder = rig.build()
    kwargs = {'clock': rig.clock}
    if 'watch_zone' in inspect.signature(SquirrelRecordingObserver).parameters:
        kwargs['watch_zone'] = WatchZoneConfig()
    observer = SquirrelRecordingObserver(recorder,
        AutomaticRecordingConfig(enabled=False, squirrel_confidence=.25), 1.5, **kwargs)
    seen = []
    runtime = SimpleNamespace(observer=observer, control=SimpleNamespace(observe=seen.append))
    for sequence in range(1, 101):
        CollectorRuntime._observe(runtime, observation(sequence=sequence), None)
    assert len(seen) == 100
    assert not recorder.status()['active']
    assert recorder.status().get('session_id') is None
    assert not list(recorder.directory.rglob('*.avi'))
    assert not list(recorder.directory.rglob('*.png'))


@pytest.mark.parametrize('reason', ['below_threshold', 'outside_calibration', 'stale', 'target_left'])
def test_rejected_automatic_candidate_keeps_audit_without_starting_video(control_rig, reason):
    r = control_rig(complete=True, auto=True, dry=False)
    calls = []
    r.control._recording = SimpleNamespace(record_firing=lambda *args: calls.append(args))
    candidate = observation()
    if reason == 'below_threshold':
        candidate = observation(boxes=(((600, 320, 680, 400), .01),))
    elif reason == 'outside_calibration':
        candidate = observation(boxes=(((0, 0, 20, 20), .9),))
    elif reason == 'stale':
        candidate = observation(at=90)
    else:
        r.detector['last_result'] = asdict(observation(boxes=(((800, 320, 880, 400), .9),), sequence=2))
    r.control.observe(candidate)
    result = finish(r.control)
    assert result['outcome'] == 'rejected'
    assert calls == []
    assert ('open',) not in r.events
    if reason in ('stale', 'target_left'):
        assert r.control.firing_history()['recent'][0]['outcome'] == 'rejected'


def test_automatic_shot_reserves_video_after_target_checks_before_movement(control_rig):
    r = control_rig(complete=True, auto=True, dry=False)
    calls = []
    def record(attempt, event, details):
        transitions = [x[0] for x in r.events if x[0] in ('open', 'close')]
        assert transitions[-1] == 'close'
        if event == 'attempt_started':
            assert not any(x[0] == 'angle' for x in r.events)
        calls.append(event)
        return dict(session_id='synthetic-recording', post_fire_seconds=10)
    r.control._recording = SimpleNamespace(record_firing=record)
    r.control.observe(observation())
    assert finish(r.control)['outcome'] == 'fired'
    assert calls == ['attempt_started', 'pulse_intent', 'before_open', 'pulse_result', 'attempt_finished']
    shot = r.control.firing_history()['last_shot']
    assert shot['pulse_evidence'] == 'open_and_close_returned'
    assert shot['recording']['session_id'] == 'synthetic-recording'


def test_manual_recording_remains_independent_of_detection_recording(rig):
    recorder = rig.build()
    recorder.record_manual()
    captured(rig, recorder)
    recorder.stop_manual()
    result = finished(recorder)
    assert result['manual_requested']
    assert result['segments'] and result['error'] is None
