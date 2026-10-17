"""Adversarial safety checks using fake outputs; never imports live hardware adapters."""

from squirrel_shooter.watch_zone import WatchZoneConfig

from dataclasses import asdict, dataclass
from types import SimpleNamespace
import threading

import pytest

from squirrel_shooter.collector_control import CollectorControl, ControlError
from squirrel_shooter.collector_calibration import CalibrationStore, CalibrationError
from squirrel_shooter.collector_control_config import AutomaticFireConfig, PhysicalControlConfig
from squirrel_shooter.pan_tilt import PanTiltPosition, PanTiltConfig
from squirrel_shooter.valve import ValveState


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeValve:
    def __init__(self, events):
        self.events = events
        self.state = ValveState.CLOSED
        self.cleaned = False
        self.fail_after_open = False
        self.fail_close_when_open = False

    def close(self):
        self.events.append(('valve_close',))
        if self.fail_close_when_open and self.state == ValveState.OPEN:
            raise RuntimeError('valve close failed')
        self.state = ValveState.CLOSED

    def open(self):
        self.events.append(('valve_open',))
        self.state = ValveState.OPEN
        if self.fail_after_open:
            raise RuntimeError('output failed after opening')

    def cleanup(self):
        self.close()
        self.cleaned = True
        self.events.append(('valve_cleanup',))


class FakeServo:
    def __init__(self, events, valve, *, known=True):
        self.events, self.valve = events, valve
        self.position = PanTiltPosition(85, 85) if known else None
        self.cleaned = False
        self.fail_move = False

    def move_to_smooth(self, pan, tilt, **kwargs):
        assert self.valve.state == ValveState.CLOSED, 'servo moved while valve was open'
        self.events.append(('servo_move', pan, tilt))
        if self.fail_move:
            raise RuntimeError('servo write failed')
        self.position = PanTiltPosition(pan, tilt)
        return self.position

    def cleanup(self):
        assert self.valve.state == ValveState.CLOSED
        self.cleaned = True
        self.events.append(('servo_cleanup',))


class FakeStore:
    def __init__(self, complete, path):
        self.is_complete = complete
        self.locked = []
        self.limits = PanTiltConfig()
        self.geometry = CalibrationStore(path, self.limits)
        for row, y in enumerate((100, 360, 620)):
            for col, x in enumerate((100, 640, 1180)):
                self.geometry.lock_point(row * 3 + col + 1, x, y, 90, 95, 1280, 720)
        self.geometry.complete(1280, 720)

    def region(self, width, height):
        if not self.is_complete:
            raise CalibrationError('Calibration is incomplete')
        return self.geometry.region(width, height)

    def status(self):
        return {'complete': self.is_complete, 'points': []}

    def aim(self, x, y, width, height):
        return SimpleNamespace(pan=90.0, tilt=95.0)

    def lock_point(self, *args, **kwargs):
        self.locked.append((args, kwargs))


@dataclass
class FakeObservation:
    camera_generation: int = 1
    source_sequence: int = 1
    source_monotonic: float = 100.0
    native_width: int = 1280
    native_height: int = 720
    dropped_reason: str | None = None
    output: dict | None = None

    def __post_init__(self):
        if self.output is None:
            self.output = {'available': True, 'error': None,
                           'detections': [{'confidence': .9, 'xyxy': [600, 320, 680, 400]}]}


def idle(control):
    with control._condition:
        assert control._condition.wait_for(
            lambda: not control._active and control._job is None, timeout=3), 'worker did not finish'
    return control.status()['last_result']


@pytest.fixture
def make_control(tmp_path):
    controls = []

    def make(*, complete=True, automatic=False, dry=False, known=True, fast_wait=True):
        events, clock = [], FakeClock()
        valve = FakeValve(events)
        servo = FakeServo(events, valve, known=known)
        store = FakeStore(complete, tmp_path / f'geometry-{len(controls)}.json')
        camera = {'online': True, 'error': None, 'last_frame_age_seconds': 0,
                  'width': 1280, 'height': 720, 'frame_generation': 1}
        observation = FakeObservation()
        detector = {'paused': False, 'result_available': True, 'last_result': asdict(observation)}
        control = CollectorControl(
            PhysicalControlConfig(enabled=True, valve_enabled=True,
                                  calibration_file=tmp_path / f'calibration-{len(controls)}.json'),
            AutomaticFireConfig(enabled=automatic, dry_run=dry,
                                minimum_confidence=.8 if automatic else None),
            watch_zone=WatchZoneConfig(), camera_status=lambda: dict(camera), detector_status=lambda: detector,
            maximum_age=3, clock=clock, servo=servo, valve=valve, store=store)
        if fast_wait:
            def wait(seconds):
                if control._stop.is_set():
                    raise ControlError('control is stopping')
                events.append(('wait', seconds, control._state))
                clock.now += seconds
            control._wait = wait
        controls.append(control)
        control.start()
        return SimpleNamespace(control=control, events=events, clock=clock, valve=valve,
                               servo=servo, store=store, camera=camera, detector=detector,
                               observation=observation)

    yield make
    for control in controls:
        control._valve.fail_close_when_open = False
        control.stop()


@pytest.mark.parametrize('complete', [False, True])
def test_water_test_needs_no_aim_camera_or_servo_and_retains_history(make_control, complete):
    rig = make_control(complete=complete, known=False)
    rig.camera.update(online=False, last_frame_age_seconds=None)
    rig.control._servo = None
    events = []
    rig.control._recording = SimpleNamespace(
        record_firing=lambda attempt, event, details: events.append(event) or {})
    rig.control.request('water_test')
    assert idle(rig.control) == dict(action='water_test', outcome='manual_pulse_closed')
    assert [event for event in rig.events if event[0] == 'valve_open'] == [('valve_open',)]
    assert rig.valve.state == ValveState.CLOSED
    assert rig.control._servo is None
    assert rig.control._settled is False
    assert rig.control.status()['calibration_mode'] is (not complete)
    assert rig.control.status()['pulse_seconds'] == .4
    assert rig.control.status()['cooldown_remaining_seconds'] == 10
    history = rig.control.firing_history()
    assert history['shots'] == 1
    assert history['last_shot']['action'] == 'water_test'
    assert 'before_open' in events and 'attempt_finished' in events
    for action in ('water_test', 'fire', 'dry_sequence'):
        with pytest.raises(ControlError, match='cooldown'):
            rig.control.request(action)
    rig.clock.now += 10
    rig.control.request('water_test')
    assert idle(rig.control)['outcome'] == 'manual_pulse_closed'
    assert rig.control.firing_history()['shots'] == 2


def test_water_test_open_failure_closes_valve_and_starts_cooldown(make_control):
    rig = make_control(known=False)
    rig.valve.fail_after_open = True
    rig.control.request('water_test')
    result = idle(rig.control)
    assert result['outcome'] == 'error'
    assert rig.valve.state == ValveState.CLOSED
    assert rig.control.status()['cooldown_remaining_seconds'] == 10
    assert rig.control.status()['error']


def test_water_test_cannot_overlap_other_control_actions(make_control):
    rig = make_control(known=False)
    opened, release = threading.Event(), threading.Event()
    def wait(_seconds):
        opened.set()
        assert release.wait(3)
    rig.control._wait = wait
    try:
        rig.control.request('water_test')
        assert opened.wait(3)
        for action in ('water_test', 'park', 'fire'):
            with pytest.raises(ControlError, match='busy'):
                rig.control.request(action)
    finally:
        release.set()
    assert idle(rig.control)['outcome'] == 'manual_pulse_closed'
    assert rig.valve.state == ValveState.CLOSED


def test_disabled_master_never_constructs_gpio_or_servo(monkeypatch, tmp_path):
    import squirrel_shooter.collector_control as module

    def unexpected(*args, **kwargs):
        raise AssertionError('disabled control initialized hardware')

    monkeypatch.setattr(module, 'GPIOValveController', unexpected)
    monkeypatch.setattr(module, 'PanTiltController', unexpected)
    control = CollectorControl(PhysicalControlConfig(calibration_file=tmp_path / 'calibration.json'),
                               AutomaticFireConfig(), watch_zone=WatchZoneConfig(), camera_status=lambda: {},
                               detector_status=lambda: {}, maximum_age=3,
                               store=FakeStore(False, tmp_path / 'geometry.json'))
    control.start()
    control.stop()
    assert control._thread is None


def test_startup_closes_output_but_does_not_move_unknown_servos(make_control):
    rig = make_control(known=False)
    assert rig.valve.state == ValveState.CLOSED
    assert rig.servo.position is None
    assert all(event[0] != 'servo_move' for event in rig.events)


def test_unknown_start_waits_conservative_travel_before_settling(make_control):
    rig = make_control(known=False, complete=False)
    rig.control.request('center')
    assert idle(rig.control)['outcome'] == 'position_commanded'
    waits = [event[1] for event in rig.events if event[0] == 'wait']
    assert waits == pytest.approx([120 / 45, .15])
    assert not any(event[0] == 'valve_open' for event in rig.events)


def test_automatic_wet_sequence_settles_then_closes_before_park(make_control):
    rig = make_control(automatic=True)
    rig.control.observe(rig.observation)
    assert idle(rig.control)['outcome'] == 'fired'
    events = rig.events
    opening = events.index(('valve_open',))
    assert ('servo_move', 90.0, 95.0) in events[:opening]
    assert ('wait', .15, 'SETTLING') in events[:opening]
    pulse_wait = events.index(('wait', .4, 'FIRING'))
    parked = events.index(('servo_move', 85.0, 82.0))
    assert opening < pulse_wait < parked
    assert ('valve_close',) in events[pulse_wait + 1:parked]
    assert rig.control.status()['cooldown_remaining_seconds'] > 0


def test_exception_after_open_still_closes_before_any_park(make_control):
    rig = make_control(automatic=True)
    rig.valve.fail_after_open = True
    rig.control.observe(rig.observation)
    result = idle(rig.control)
    assert result['outcome'] == 'error'
    assert 'output failed' in result['error']
    assert rig.valve.state == ValveState.CLOSED
    opening = rig.events.index(('valve_open',))
    parked = rig.events.index(('servo_move', 85.0, 82.0))
    assert ('valve_close',) in rig.events[opening + 1:parked]
    with pytest.raises(ControlError, match='fault'):
        rig.control.request('park')


def test_valve_close_failure_blocks_park_and_future_requests(make_control):
    rig = make_control(automatic=True)
    rig.valve.fail_close_when_open = True
    rig.control.observe(rig.observation)
    result = idle(rig.control)
    assert result['outcome'] == 'error'
    assert not any(event == ('servo_move', 85.0, 82.0) for event in rig.events)
    assert 'valve close failed' in rig.control.status()['error']
    with pytest.raises(ControlError, match='fault'):
        rig.control.request('park')


def test_movement_failure_never_opens_valve(make_control):
    rig = make_control(automatic=True)
    rig.servo.fail_move = True
    rig.control.observe(rig.observation)
    assert idle(rig.control)['outcome'] == 'error'
    assert not any(event[0] == 'valve_open' for event in rig.events)
    assert rig.valve.state == ValveState.CLOSED


def test_stale_final_result_after_settle_rejects_water(make_control):
    rig = make_control(automatic=True)
    initial_wait = rig.control._wait

    def wait(seconds):
        initial_wait(seconds)
        if rig.control._state == 'SETTLING':
            rig.clock.now += 4

    rig.control._wait = wait
    rig.control.observe(rig.observation)
    assert idle(rig.control)['outcome'] == 'rejected'
    assert not any(event[0] == 'valve_open' for event in rig.events)


def test_new_box_must_still_cover_the_original_aim_pixel(make_control):
    rig = make_control(automatic=True)
    initial_wait = rig.control._wait

    def wait(seconds):
        initial_wait(seconds)
        if rig.control._state == 'SETTLING':
            rig.detector['last_result']['source_sequence'] = 2
            rig.detector['last_result']['output']['detections'][0]['xyxy'] = [800, 320, 880, 400]

    rig.control._wait = wait
    rig.control.observe(rig.observation)
    assert idle(rig.control)['outcome'] == 'rejected'
    assert not any(event[0] == 'valve_open' for event in rig.events)


def test_dry_automatic_sequence_does_not_open_gpio(make_control):
    rig = make_control(automatic=True, dry=True)
    rig.control.observe(rig.observation)
    assert idle(rig.control)['outcome'] == 'dry_run'
    assert ('servo_move', 85.0, 82.0) in rig.events
    assert not any(event[0] == 'valve_open' for event in rig.events)


def test_shutdown_interrupts_an_open_pulse_and_prevents_park(make_control):
    rig = make_control(automatic=True, fast_wait=False)
    pulsing = threading.Event()
    original_wait = rig.control._wait

    def wait(seconds):
        if rig.control._state == 'FIRING':
            pulsing.set()
        original_wait(seconds)

    rig.control._wait = wait
    rig.control.observe(rig.observation)
    assert pulsing.wait(2), 'pulse never started'
    assert rig.valve.state == ValveState.OPEN
    with pytest.raises(ControlError, match='busy'):
        rig.control.request('park')
    rig.control.stop()
    assert rig.valve.state == ValveState.CLOSED
    assert rig.valve.cleaned and rig.servo.cleaned
    assert not rig.control._thread.is_alive()
    assert ('servo_move', 85.0, 82.0) not in rig.events


def test_picked_up_move_is_cancelled_when_shutdown_already_requested(make_control):
    rig = make_control(complete=False)
    rig.control._stop.set()  # Simulate shutdown after worker pickup, before _move's first write.
    with pytest.raises(ControlError):
        rig.control._move(90, 95)
    assert not any(event[0] == 'servo_move' for event in rig.events)


def test_manual_fire_parks_outside_calibration_and_invalidates_old_aim(make_control):
    rig = make_control(complete=True)
    rig.control.request('dry_aim', pixel_x=640, pixel_y=360)
    assert idle(rig.control)['outcome'] == 'aimed_without_water'
    rig.control.request('fire')
    assert idle(rig.control)['outcome'] == 'manual_pulse_closed'
    assert ('servo_move', 85.0, 82.0) in rig.events
    rig.clock.now += 11
    rig.control.request('fire')
    assert idle(rig.control)['outcome'] == 'rejected'
    assert sum(event[0] == 'valve_open' for event in rig.events) == 1


@pytest.mark.parametrize('intervening_action', ['dry_sequence', 'automatic'])
def test_intervening_sequence_cannot_leave_old_manual_aim_authorized(make_control, intervening_action):
    rig = make_control(automatic=intervening_action == 'automatic')
    rig.control.request('dry_aim', pixel_x=640, pixel_y=360)
    assert idle(rig.control)['outcome'] == 'aimed_without_water'
    if intervening_action == 'automatic':
        rig.control.observe(rig.observation)
    else:
        rig.control.request('dry_sequence', pixel_x=650, pixel_y=370)
    assert idle(rig.control)['outcome'] in {'fired', 'dry_run'}
    assert rig.servo.position == PanTiltPosition(85, 82)
    openings_before = sum(event[0] == 'valve_open' for event in rig.events)
    rig.clock.now += 11
    try:
        rig.control.request('fire')
    except ControlError:
        pass
    else:
        assert idle(rig.control)['outcome'] == 'rejected'
    assert sum(event[0] == 'valve_open' for event in rig.events) == openings_before


@pytest.mark.parametrize('camera_change', [{'frame_generation': 2}, {'width': 640}])
@pytest.mark.parametrize('action', ['fire', 'lock'])
def test_calibration_reference_cannot_survive_camera_source_change(make_control, camera_change, action):
    rig = make_control(complete=False)
    rig.control.request('center')
    idle(rig.control)
    rig.control.request('reticle', point=1, pixel_x=300, pixel_y=300)
    rig.camera.update(camera_change)
    try:
        rig.control.request(action, point=1, pixel_x=300, pixel_y=300)
    except ControlError:
        pass
    else:
        result = idle(rig.control)
        assert result and result['outcome'] == 'rejected'
    assert not rig.store.locked
    assert not any(event[0] == 'valve_open' for event in rig.events)


def test_worker_start_failure_preserves_original_error_and_cleans_outputs(monkeypatch, tmp_path):
    events = []
    valve = FakeValve(events)
    servo = FakeServo(events, valve)
    control = CollectorControl(
        PhysicalControlConfig(enabled=True, valve_enabled=True, calibration_file=tmp_path / 'cal.json'),
        AutomaticFireConfig(), watch_zone=WatchZoneConfig(), camera_status=lambda: {}, detector_status=lambda: {},
        maximum_age=3, servo=servo, valve=valve,
        store=FakeStore(False, tmp_path / 'geometry.json'))

    def failed_start(self):
        raise RuntimeError('worker start failed')

    monkeypatch.setattr(threading.Thread, 'start', failed_start)
    try:
        with pytest.raises(RuntimeError, match='worker start failed'):
            control.start()
        assert valve.cleaned and servo.cleaned
        assert valve.state == ValveState.CLOSED
    finally:
        control._thread = None  # Make cleanup possible even on the intentionally adversarial failure.
        control.stop()
