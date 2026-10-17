from squirrel_shooter.watch_zone import WatchZoneConfig
from dataclasses import asdict, replace
from types import SimpleNamespace
import time
import json

import pytest
import yaml

from conftest import write_test_config
from squirrel_shooter.collector_calibration import CalibrationStore
from squirrel_shooter.collector_control import CollectorControl, ControlError
from squirrel_shooter.collector_firing_history import FiringHistory
from squirrel_shooter.collector_control_config import PhysicalControlConfig, AutomaticFireConfig
from squirrel_shooter.collector_app import create_collector_app
from squirrel_shooter.config import load_config, ConfigError
from squirrel_shooter.detector import DetectionOutput, SquirrelBox
from squirrel_shooter.inference_worker import DetectorObservation
from squirrel_shooter.pan_tilt import PanTiltController


def grid(store, complete=True):
    for row, y in enumerate((100, 360, 620)):
        for col, x in enumerate((100, 640, 1180)):
            store.lock_point(row * 3 + col + 1, x, y, (30, 85, 150)[col], (70, 110, 150)[row], 1280, 720)
    if complete:
        store.complete(1280, 720)


def observation(boxes=(((600, 320, 680, 400), .9),), *, sequence=1, at=100., generation=1):
    return DetectorObservation(generation, sequence, 1280, 720, at, 'test', at, at + .1, .1,
        DetectionOutput(True, None, tuple(SquirrelBox(box, score) for box, score in boxes)))


@pytest.fixture
def control_rig(tmp_path):
    created = []
    def build(*, complete=False, auto=False, dry=True, water=True, enabled=True):
        events = []
        class Driver:
            def configure_channel(self, *args): pass
            def set_angle(self, channel, angle): events.append(('angle', channel, angle))
            def release(self, channel): pass
            def cleanup(self): events.append(('servo_cleanup',))
        class Valve:
            def close(self): events.append(('close',))
            def open(self): events.append(('open',))
            def cleanup(self): self.close(); events.append(('valve_cleanup',))
        config = PhysicalControlConfig(enabled=enabled, valve_enabled=water,
            calibration_file=tmp_path / f'cal-{len(created)}.json')
        store = CalibrationStore(config.calibration_file, config.pan_tilt)
        if complete:
            grid(store)
        servo = PanTiltController(replace(config.pan_tilt, park_on_cleanup=False),
                                  driver=Driver(), sleep=lambda delay: events.append(('move_wait', delay)))
        now = [100.]
        camera = dict(online=True, error=None, width=1280, height=720, frame_generation=1, last_frame_age_seconds=.1)
        detector = dict(paused=False, result_available=True, last_result=asdict(observation()))
        control = CollectorControl(config, AutomaticFireConfig(enabled=auto, dry_run=dry, minimum_confidence=.1),
            watch_zone=WatchZoneConfig(), camera_status=lambda: camera, detector_status=lambda: detector,
            maximum_age=1.5, clock=lambda: now[0],
            servo=servo, valve=Valve(), store=store)
        control._wait = lambda delay: events.append(('wait', delay))
        control.start()
        rig = SimpleNamespace(control=control, events=events, camera=camera, detector=detector,
                              servo=servo, now=now, store=store)
        created.append(rig)
        return rig
    yield build
    for rig in created:
        rig.control.stop()


def finish(control):
    deadline = time.monotonic() + 3
    while control.status()['busy'] and time.monotonic() < deadline:
        time.sleep(.002)
    assert not control.status()['busy']
    return control.status()['last_result']


def test_start_status_and_stop_never_initialize_servo(control_rig, monkeypatch):
    rig = control_rig()
    rig.control.stop()
    def forbidden(*args, **kwargs):
        raise AssertionError('startup must not initialize PCA9685')
    monkeypatch.setattr('squirrel_shooter.collector_control.PanTiltController', forbidden)
    control = CollectorControl(rig.control.config, rig.control.automatic,
        watch_zone=WatchZoneConfig(), camera_status=lambda: rig.camera, detector_status=lambda: rig.detector,
        maximum_age=1.5, valve=rig.control._valve)
    try:
        control.start()
        assert control.status()['position'] is None
        with pytest.raises(ControlError, match='command and settle'):
            control.request('reticle', point=1, pixel_x=10, pixel_y=10)
            control.request('lock', point=1, pixel_x=10, pixel_y=10)
        control.request('move', direction='left', step=1)
        assert finish(control)['outcome'] == 'rejected'
    finally:
        control.stop()


@pytest.mark.parametrize('step', [5, 3, 1, .5])
def test_manual_steps_directions_commanded_tracking_and_clamp(control_rig, step):
    r = control_rig()
    assert r.control.status()['position'] is None
    r.control.request('center'); finish(r.control)
    assert r.control.status()['position'] == dict(pan=85, tilt=85)
    for direction, expected in [('left', (85 + step, 85)), ('right', (85, 85)),
                                 ('up', (85, 85 + step)), ('down', (85, 85))]:
        r.control.request('move', direction=direction, step=step); finish(r.control)
        assert r.control.status()['position'] == dict(zip(('pan', 'tilt'), expected))
    r.servo.move_to(149.9, 149.9)
    r.control.request('move', direction='left', step=step); finish(r.control)
    assert r.servo.position.pan == 150
    r.control.request('park'); finish(r.control)
    assert r.control.status()['position'] == dict(pan=85, tilt=82)
    assert 'no servo feedback' in r.control.status()['position_basis']
    assert not any(e[0] == 'open' for e in r.events)


def test_lock_uses_backend_angles_requires_explicit_relock_and_complete_hides_setup(control_rig):
    r = control_rig()
    r.control.request('center'); finish(r.control)
    r.control.request('reticle', point=1, pixel_x=100, pixel_y=100)
    r.control.request('lock', point=1, pixel_x=100, pixel_y=100, pan=999, tilt=999)
    assert r.store.status()['points'][0]['pan'] == 85
    with pytest.raises(ValueError, match='[Rr]elock|locked'):
        r.control.request('lock', point=1, pixel_x=100, pixel_y=100)
    r.control.request('lock', point=1, pixel_x=100, pixel_y=100, relock=True)
    with pytest.raises(ValueError): r.control.request('complete')
    r.store.reset(1280, 720); grid(r.store, complete=False)
    r.control.request('complete')
    assert not r.control.status()['calibration_mode']
    assert not r.control.status()['automatic_fire_enabled']
    with pytest.raises(ControlError, match='closed'):
        r.control.request('move', direction='left', step=1)
    r.control.request('enter_calibration')
    assert r.control.status()['calibration_mode']


@pytest.mark.parametrize('editing', [False, True])
def test_manual_calibration_can_repeat_pulses_before_pixel_save(control_rig, editing):
    r = control_rig(auto=True, complete=editing)
    if editing:
        r.control.request('enter_calibration')
    r.control.request('center'); finish(r.control)
    assert r.control.status()['active_reticle'] is None
    r.control.request('fire'); assert finish(r.control)['outcome'] == 'manual_pulse_closed'
    with pytest.raises(ControlError, match='cooldown'):
        r.control.request('fire')
    r.now[0] += 11
    r.control.request('move', direction='left', step=1); finish(r.control)
    r.control.request('fire'); assert finish(r.control)['outcome'] == 'manual_pulse_closed'
    assert [e[0] for e in r.events].count('open') == 2
    assert r.servo.position.pan == 86 and r.servo.position.tilt == 85
    r.control.request('reticle', point=1, pixel_x=100, pixel_y=100)
    r.control.request('lock', point=1, pixel_x=100, pixel_y=100, relock=editing)
    assert r.store.status()['points'][0]['pan'] == 86
    assert not r.control.status()['automatic_fire_enabled']


def test_automatic_uses_best_box_center_and_parks_after_dry_or_wet_sequence(control_rig):
    for dry in (True, False):
        r = control_rig(complete=True, auto=True, dry=dry)
        obs = observation((((900, 500, 1000, 600), .5), ((600, 320, 680, 400), .9)))
        r.detector['last_result'] = asdict(obs)
        r.control.observe(obs)
        result = finish(r.control)
        assert result['pixel'] == [640., 360.]
        assert (result['pan'], result['tilt']) == (85., 110.)
        assert result['outcome'] == ('dry_run' if dry else 'fired')
        assert r.control.status()['state'] == 'COOLDOWN'
        assert r.control.status()['position'] == dict(pan=85, tilt=82)
        assert any(e[0] == 'open' for e in r.events) is (not dry)


@pytest.mark.parametrize('case', ['incomplete', 'dimensions', 'outside', 'stale', 'generation', 'no_box', 'night', 'moved'])
def test_invalid_auto_targets_never_open_valve(control_rig, case):
    r = control_rig(complete=case != 'incomplete', auto=True, dry=False)
    obs = observation()
    if case == 'dimensions': r.camera['width'] = 640
    if case == 'outside': obs = observation((((0, 0, 20, 20), .9),))
    if case == 'stale': r.now[0] = 103
    if case == 'generation': r.camera['frame_generation'] = 2
    if case == 'no_box': r.detector['last_result'] = asdict(observation(()))
    if case == 'night': r.detector['paused'] = True
    if case == 'moved': r.detector['last_result'] = asdict(observation((((900, 500, 1000, 600), .9),), sequence=2))
    r.control.observe(obs); finish(r.control)
    assert not any(e[0] == 'open' for e in r.events)


@pytest.mark.parametrize('physical,auto', [(False, True), (True, False)])
def test_disabled_paths_do_not_move_or_fire(control_rig, physical, auto):
    r = control_rig(complete=True, enabled=physical, auto=auto, dry=False)
    r.control.observe(observation())
    assert not r.control.status()['busy']
    assert not any(e[0] in ('open', 'angle') for e in r.events)
    if not physical:
        with pytest.raises(ControlError): r.control.request('center')


def test_dry_aim_and_manual_test_fire_park_without_mutating_calibration(control_rig):
    r = control_rig(complete=True)
    before = r.store.path.read_bytes() if hasattr(r.store, 'path') else r.control.config.calibration_file.read_bytes()
    r.control.request('dry_aim', pixel_x=370, pixel_y=230)
    assert finish(r.control)['outcome'] == 'aimed_without_water'
    assert not any(e[0] == 'open' for e in r.events)
    r.control.request('fire'); finish(r.control)
    assert r.control.status()['position'] == dict(pan=85, tilt=82)
    assert r.control.config.calibration_file.read_bytes() == before
    r.now[0] += 11
    r.control.request('fire'); assert finish(r.control)['outcome'] == 'rejected'


@pytest.mark.parametrize('water', [True, False])
def test_collector_home_water_test_button_and_token_protected_action(tmp_path, control_rig, water):
    r = control_rig(water=water)
    config = load_config(write_test_config(tmp_path))
    app = create_collector_app(SimpleNamespace(control=r.control, config=config))
    client = app.test_client()
    page = client.get('/')
    assert page.status_code == 200
    assert b'id="water-test"' in page.data
    assert b'No aiming needed.' in page.data
    assert client.post('/api/control/action', json={'action': 'water_test'}).status_code == 403
    assert not any(e[0] == 'open' for e in r.events)
    headers = {'X-Control-Token': app.config['COLLECTOR_CONTROL_TOKEN']}
    response = client.post('/api/control/action', json={'action': 'water_test'}, headers=headers)
    assert response.status_code == (202 if water else 409)
    if water:
        assert finish(r.control)['outcome'] == 'manual_pulse_closed'
        assert client.post('/api/control/action', json={'action': 'water_test'}, headers=headers).status_code == 409
    assert sum(e[0] == 'open' for e in r.events) == int(water)


def test_control_routes_csrf_setup_lifecycle_and_no_auto_enable_endpoint(tmp_path, control_rig):
    r = control_rig()
    config = load_config(write_test_config(tmp_path))
    app = create_collector_app(SimpleNamespace(control=r.control, config=config))
    client = app.test_client()
    assert client.get('/calibration').status_code == 200
    assert client.get('/aim-test').status_code == 404
    assert client.post('/api/control/action', json={'action': 'center'}).status_code == 403
    headers = {'X-Control-Token': app.config['COLLECTOR_CONTROL_TOKEN']}
    assert client.post('/api/control/action', json=[], headers=headers).status_code == 400
    assert client.post('/api/control/action', json={'action': 'enable_auto'}, headers=headers).status_code == 409
    grid(r.store, complete=False)
    assert client.post('/api/control/action', json={'action': 'complete'}, headers=headers).status_code == 200
    assert client.get('/calibration').status_code == 404
    assert client.get('/aim-test').status_code == 200
    page = client.get('/control-setup').data
    assert b'Enter calibration' in page and b'preview.mjpg' not in page
    assert client.post('/api/control/action', json={'action': 'enter_calibration'}, headers=headers).status_code == 200
    assert client.get('/calibration').status_code == 200


def test_config_keeps_new_authority_separate_and_calibration_beside_config(tmp_path):
    path = write_test_config(tmp_path)
    config = load_config(path)
    assert not config.physical_control.enabled and not config.automatic_fire.enabled
    assert config.physical_control.calibration_file == tmp_path / 'collector-calibration.json'
    raw = yaml.safe_load(path.read_text()); raw['automatic_recording'] = dict(enabled=False, squirrel_confidence=.1)
    raw['automatic_fire'] = dict(enabled=True, dry_run=True, minimum_confidence=.5)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match='physical_control'): load_config(path)
    raw['physical_control'] = dict(enabled=True)
    raw['detector'] = dict(enabled=True)
    path.write_text(yaml.safe_dump(raw)); config = load_config(path)
    assert config.automatic_recording.squirrel_confidence == .1
    raw['automatic_fire']['dry_run'] = False
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match='valve_enabled'): load_config(path)


def test_enabled_collector_hands_existing_result_to_control_after_recording_without_new_capture(tmp_path, control_rig, monkeypatch):
    import numpy as np
    import squirrel_shooter.collector_control as module
    from squirrel_shooter.camera_service import FramePacket
    from squirrel_shooter.collector_app import CollectorRuntime
    from squirrel_shooter.detector import DetectorConfig
    r = control_rig(complete=True, auto=True)
    monkeypatch.setattr(module, 'CollectorControl', lambda *args, **kwargs: r.control)
    calls = []
    config = replace(load_config(write_test_config(tmp_path)), physical_control=r.control.config,
                     automatic_fire=r.control.automatic, detector=DetectorConfig(enabled=True))
    camera = SimpleNamespace(status=lambda: r.camera, stop=lambda: calls.append('camera_stop'))
    detector = SimpleNamespace(status=lambda: {}, infer=lambda frame: observation().output)
    recorder = SimpleNamespace(stop=lambda: calls.append('recording_stop'))
    runtime = CollectorRuntime(config, camera=camera, detector=detector, recording=recorder)
    runtime.worker._clock = lambda: 100.1
    runtime.observer.observe = lambda obs, packet: calls.append(('record', obs.source_sequence, packet.sequence))
    original = r.control.observe
    r.control.observe = lambda obs: (calls.append(('control', obs.source_sequence)), original(obs))
    frame = np.full((720, 1280, 3), 51, np.uint8)
    frame.flags.writeable = False
    packet = FramePacket(1, frame, 'test', 100., 1)
    try:
        assert runtime.camera is runtime.worker.camera is camera
        assert runtime.worker.submit(packet) and runtime.worker.run_once()
        assert calls[:2] == [('record', 1, 1), ('control', 1)]
        assert finish(r.control)['outcome'] == 'dry_run'
        assert np.all(frame == 51)
    finally:
        runtime.stop()
    assert calls[-2:] == ['recording_stop', 'camera_stop']


def test_normal_status_omits_setup_map_and_does_not_resample_camera_telemetry(control_rig):
    r = control_rig(complete=True)
    def unnecessary_camera_read():
        raise AssertionError('normal collector already obtains camera status')
    r.control._camera_status = unnecessary_camera_read
    status = r.control.status(compact=True)
    assert status['calibration'] == dict(complete=True, error=None)
    assert 'points' not in status['calibration']


def test_firing_history_retains_shot_after_empty_frames_and_restart(control_rig):
    r = control_rig(complete=True, auto=True, dry=False)
    r.control.observe(observation())
    assert finish(r.control)['outcome'] == 'fired'
    first = r.control.firing_history()
    assert first['shots'] == 1 and first['attempts'] == 1
    shot = first['last_shot']
    assert shot['confidence'] == .9 and shot['source_sequence'] == 1
    assert shot['valve_open_returned'] and shot['valve_close_returned']
    assert shot['pulse_evidence'] == 'open_and_close_returned'
    assert shot['pixel'] == [640., 360.]
    r.now[0] += 11
    r.control.observe(observation((), sequence=2, at=r.now[0]))
    finish(r.control)
    assert r.control.firing_history()['last_shot'] == shot
    r.control.stop()
    reopened = FiringHistory(r.control.history.path)
    reopened.start()
    assert reopened.snapshot()['last_shot'] == shot
    events = [json.loads(line) for line in reopened.export()]
    assert [e['event'] for e in events] == ['attempt_started', 'target_selected', 'pulse_intent', 'pulse_result', 'attempt_finished']
    assert events[2]['valve_open_returned'] is False


def test_firing_history_cancellation_and_dry_run_are_not_shots(control_rig):
    r = control_rig(complete=True, auto=True, dry=False)
    r.detector['last_result'] = asdict(observation(()))
    r.control.observe(observation())
    assert finish(r.control)['outcome'] == 'rejected'
    h = r.control.firing_history()
    assert h['shots'] == 0 and h['attempts'] == 1
    assert h['recent'][0]['error'] == 'no qualifying squirrel box'
    dry = control_rig(complete=True, auto=True, dry=True)
    dry.control.observe(observation())
    finish(dry.control)
    assert dry.control.firing_history()['shots'] == 0
    assert dry.control.firing_history()['recent'][0]['outcome'] == 'dry_run'


def test_history_write_failure_prevents_valve_open(control_rig, monkeypatch):
    r = control_rig(complete=True, auto=True, dry=False)
    original = r.control.history.update
    def fail(attempt, event, **details):
        if event == 'pulse_intent':
            raise OSError('disk full')
        return original(attempt, event, **details)
    monkeypatch.setattr(r.control.history, 'update', fail)
    r.control.observe(observation())
    finish(r.control)
    assert not any(e[0] == 'open' for e in r.events)
    assert not r.control.status()['automatic_fire_enabled']
    assert r.control.status()['error']
    assert r.control.firing_history()['shots'] == 0


def test_shot_evidence_survives_park_failure(control_rig, monkeypatch):
    r = control_rig(complete=True, auto=True, dry=False)
    move = r.control._move
    def fail_park(pan, tilt, *, parking=False):
        if parking:
            raise RuntimeError('park failed')
        return move(pan, tilt, parking=parking)
    monkeypatch.setattr(r.control, '_move', fail_park)
    r.control.observe(observation())
    assert finish(r.control)['outcome'] == 'error'
    h = r.control.firing_history()
    assert h['shots'] == 1
    assert h['last_shot']['error'] == 'park failed'
    assert h['last_shot']['valve_close_returned']


def test_manual_pulse_is_audited_and_no_storage_writes_while_open(control_rig, monkeypatch):
    r = control_rig()
    r.control.request('center'); finish(r.control)
    update = r.control.history.update
    def guard(*args, **kwargs):
        transitions = [e[0] for e in r.events if e[0] in ('open', 'close')]
        assert not transitions or transitions[-1] == 'close'
        return update(*args, **kwargs)
    monkeypatch.setattr(r.control.history, 'update', guard)
    r.control.request('fire'); finish(r.control)
    h = r.control.firing_history()
    assert h['shots'] == 1 and h['last_shot']['action'] == 'fire'
    assert h['last_shot']['calibration_mode']


def test_restart_marks_unfinished_intent_uncertain_without_inventing_shot(tmp_path):
    h = FiringHistory(tmp_path / 'history.sqlite3'); h.start()
    attempt = h.begin(action='automatic')
    h.update(attempt, 'pulse_intent', pulse_evidence='open_intent_only')
    recovered = FiringHistory(h.path); recovered.start()
    state = recovered.snapshot()
    assert state['shots'] == 0
    assert state['recent'][0]['outcome'] == 'interrupted'
    assert state['recent'][0]['pulse_evidence'] == 'open_intent_only'
    assert len(list(recovered.export())) == 3


def test_restart_retains_completed_pulse_before_final_outcome(tmp_path):
    h = FiringHistory(tmp_path / 'history.sqlite3'); h.start()
    attempt = h.begin(action='automatic')
    h.update(attempt, 'pulse_result', valve_open_returned=True, valve_close_returned=True,
             pulse_evidence='open_and_close_returned')
    h.start()
    assert h.snapshot()['shots'] == 1
    assert h.snapshot()['last_shot']['outcome'] == 'interrupted'


def test_history_api_export_contains_persistent_events(tmp_path, control_rig):
    r = control_rig(complete=True, auto=True, dry=False)
    r.control.observe(observation()); finish(r.control)
    config = load_config(write_test_config(tmp_path))
    client = create_collector_app(SimpleNamespace(control=r.control, config=config)).test_client()
    assert client.get('/api/control/history').json['shots'] == 1
    response = client.get('/api/control/history/export')
    assert response.status_code == 200
    assert 'attachment' in response.headers['Content-Disposition']
    assert len([json.loads(line) for line in response.data.splitlines()]) == 5


def test_target_disappears_after_durable_intent_logs_no_shot(control_rig, monkeypatch):
    r = control_rig(complete=True, auto=True, dry=False)
    update = r.control.history.update
    def disappear(attempt, event, **details):
        result = update(attempt, event, **details)
        if event == 'pulse_intent':
            r.detector['last_result'] = asdict(observation(()))
        return result
    monkeypatch.setattr(r.control.history, 'update', disappear)
    r.control.observe(observation()); finish(r.control)
    row = r.control.firing_history()['recent'][0]
    assert row['outcome'] == 'rejected'
    assert row['pulse_evidence'] == 'not_opened'
    assert not any(e[0] == 'open' for e in r.events)


def test_audit_failure_after_pulse_preserves_intent_and_inhibits_next_shot(control_rig, monkeypatch):
    r = control_rig(complete=True, auto=True, dry=False)
    update = r.control.history.update
    def fail_after(attempt, event, **details):
        if event in ('pulse_result', 'attempt_finished'):
            raise OSError('disk unavailable after pulse')
        return update(attempt, event, **details)
    monkeypatch.setattr(r.control.history, 'update', fail_after)
    r.control.observe(observation()); finish(r.control)
    assert any(e[0] == 'open' for e in r.events)
    assert [e[0] for e in r.events if e[0] in ('open', 'close')][-1] == 'close'
    assert r.control.firing_history()['recent'][0]['pulse_evidence'] == 'open_intent_only'
    assert not r.control.status()['automatic_fire_enabled']
    r.now[0] += 11
    r.control.observe(observation(sequence=2, at=r.now[0]))
    assert not r.control.status()['busy']


def test_failed_open_is_uncertain_and_does_not_increment_shots(control_rig, monkeypatch):
    r = control_rig(complete=True, auto=True, dry=False)
    def fail():
        raise RuntimeError('GPIO open failed')
    monkeypatch.setattr(r.control._valve, 'open', fail)
    r.control.observe(observation()); finish(r.control)
    h = r.control.firing_history()
    assert h['shots'] == 0
    assert h['recent'][0]['pulse_evidence'] == 'open_not_confirmed'
    assert h['recent'][0]['error'] == 'GPIO open failed'
