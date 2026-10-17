"""One serialized actuator worker; consumes collector results, never camera frames."""
from __future__ import annotations

import math
import threading
from dataclasses import asdict, replace
from time import monotonic

from .collector_calibration import CalibrationError, CalibrationStore
from .collector_firing_history import FiringHistory, utc_now
from .pan_tilt import PanTiltController, PanTiltPosition
from .valve import GPIOValveController, ValveConfig, pulse_valve


class ControlError(ValueError):
    pass


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError(f'{name} must be finite')
    try:
        number = float(value)
    except OverflowError as exc:
        raise ControlError(f'{name} must be finite') from exc
    if not math.isfinite(number):
        raise ControlError(f'{name} must be finite')
    return number


class CollectorControl:
    """No hardware until start; no movement until an explicit request or armed result."""

    def __init__(self, config, automatic, *, camera_status, detector_status, watch_zone,
                 maximum_age, clock=monotonic, servo=None, valve=None, store=None, recording=None):
        self.config, self.automatic = config, automatic
        self._camera_status, self._detector_status = camera_status, detector_status
        self._watch_zone, self._maximum_age, self._clock = watch_zone, maximum_age, clock
        self.store = store or CalibrationStore(config.calibration_file, config.pan_tilt)
        self._servo, self._valve = servo, valve
        self._condition = threading.Condition(threading.RLock())
        self._valve_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._started = False
        self._job = None
        self._active = False
        self._state = 'IDLE'
        self._fault = None
        self._last = None
        self._cooldown_until = 0.0
        self._calibrating = not self.store.status()['complete']
        self._automatic_inhibited = self._calibrating
        self._selected = 1
        self._reticle = None
        self._reticle_source = None
        self._settled = False
        self._dry_aim = None
        self._last_identity = (-1, -1)
        self.history = FiringHistory(config.calibration_file.with_suffix('.firing.sqlite3'))
        self._history_ready = False
        self._history_error = None
        self._attempt = None
        self._recording = recording
        self._recording_link = None
        self._attempt_action = None

    def start(self):
        if not self.config.enabled or self._started:
            return
        if self._stop.is_set():
            raise ControlError('control is closed')
        try:
            self.history.start()
            self._history_ready = True
            # Reserve the valve output LOW even in servo-only / automatic dry mode.
            if self._valve is None:
                self._valve = GPIOValveController(ValveConfig(
                    enabled=True, gpio_pin=self.config.valve_gpio_pin,
                    active_high=self.config.valve_active_high))
            self._close_valve()
            # Defer PCA9685 initialization until an explicit movement request.
            # Its library resets oscillator registers and can resume retained PWM.
            self._started = True
            self._thread = threading.Thread(target=self._run, name='collector-control', daemon=True)
            self._thread.start()
        except BaseException:
            self.stop()
            raise

    def _wait(self, seconds):
        if self._stop.wait(seconds):
            raise ControlError('control is stopping')

    def _close_valve(self):
        with self._valve_lock:
            if self._valve is not None:
                self._valve.close()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._job = None
            self._condition.notify_all()
        try:
            self._close_valve()
        finally:
            if self._thread is not None and self._thread.ident is not None:
                self._thread.join(5)
            try:
                if self._valve is not None:
                    self._valve.cleanup()
            finally:
                # Never race cleanup/movement if a hardware call itself has hung.
                if self._servo is not None and not (self._thread and self._thread.is_alive()):
                    self._servo.cleanup()

    def _camera(self):
        camera = self._camera_status()
        if not isinstance(camera, dict):
            camera = asdict(camera)
        age = camera.get('last_frame_age_seconds')
        if (not camera.get('online') or camera.get('error') or age is None
                or not 0 <= _number(age, 'camera age') <= self._maximum_age):
            raise ControlError('fresh camera frame required')
        return camera

    def _pixel(self, x, y, camera):
        x, y = _number(x, 'pixel_x'), _number(y, 'pixel_y')
        if not 0 <= x < camera['width'] or not 0 <= y < camera['height']:
            raise ControlError('pixel outside current camera image')
        return x, y

    def _set_state(self, value):
        with self._condition:
            self._state = value

    def _move(self, pan, tilt, *, parking=False):
        self._close_valve()
        if self._stop.is_set():
            raise ControlError('movement cancelled')
        self._dry_aim = None
        self._set_state('PARKING' if parking else 'AIMING')
        self._settled = False
        if self._servo is None:
            self._servo = PanTiltController(
                replace(self.config.pan_tilt, park_on_cleanup=False), sleep=self._wait)
        unknown_start = self._servo.position is None
        result = self._servo.move_to_smooth(pan, tilt, settling_delay_seconds=0, release=False)
        if not parking:
            self._set_state('SETTLING')
        if unknown_start:
            # Startup position is unknown; do not treat the driver's center assumption as feedback.
            p = self.config.pan_tilt
            self._wait(max(p.pan_max - p.pan_min, p.tilt_max - p.tilt_min) / p.movement_speed)
        self._wait(self.config.pan_tilt.settling_delay_seconds)
        self._settled = True
        return result

    def _pulse(self, final_check=None, *, require_settled=True):
        if not self.config.valve_enabled:
            raise ControlError('water is disabled')
        if require_settled and not self._settled:
            raise ControlError('movement and settling must finish before firing')
        duration = self.config.pulse_seconds
        if not 0 < duration <= self.config.maximum_pulse_seconds <= .40:
            raise ControlError('pulse exceeds bound')
        self._set_state('FIRING')
        # Lock only the open transition, not the delay: stop() can close immediately.
        owner = self
        evidence = dict(pulse_evidence='open_intent_only', valve_open_returned=False,
                        valve_close_returned=False, valve_open_call_started=False,
                        requested_pulse_seconds=duration)
        opened_at = None

        class GuardedValve:
            def close(self):
                owner._close_valve()
                if opened_at is not None:
                    evidence['valve_close_returned'] = True
                    evidence['closed_at'] = utc_now()
                    evidence['command_elapsed_seconds'] = monotonic() - opened_at

            def open(self):
                nonlocal opened_at
                # Persist intent BEFORE opening. Recheck the target after disk I/O.
                owner._audit('pulse_intent', **evidence)
                if final_check is not None:
                    final_check()
                owner._record_event('before_open', evidence)
                with owner._valve_lock:
                    if owner._stop.is_set() or (require_settled and not owner._settled):
                        raise ControlError('firing cancelled')
                    evidence['valve_open_call_started'] = True
                    owner._valve.open()
                    opened_at = monotonic()
                    evidence.update(valve_open_returned=True, opened_at=utc_now())

        try:
            pulse_valve(GuardedValve(), duration, sleep=self._wait)
        finally:
            # No storage work during the pulse. Retain opening evidence even if park fails.
            evidence['pulse_evidence'] = ('open_and_close_returned' if evidence['valve_close_returned']
                else 'open_returned_close_unconfirmed' if evidence['valve_open_returned']
                else 'open_not_confirmed' if evidence['valve_open_call_started'] else 'not_opened')
            self._audit('pulse_result', **evidence)

    def _audit(self, event, **details):
        if self._attempt is None:
            return
        if event in ('pulse_intent', 'pulse_result', 'attempt_finished'):
            self._record_event(event, details)
        if self._recording_link is not None:
            details['recording'] = self._recording_link
        try:
            self.history.update(self._attempt, event, **details)
        except Exception as exc:
            self._history_error = str(exc)
            self._fault = 'Firing history write failed; firing inhibited'
            self._automatic_inhibited = True
            raise ControlError(self._fault) from exc

    def _record_event(self, event, details):
        if self._recording is None or self._attempt is None:
            return
        if self._attempt_action == 'automatic' and event != 'attempt_started' and self._recording_link is None:
            # Rejected candidates must not start video through final audit callbacks.
            return
        try:
            self._recording_link = self._recording.record_firing(self._attempt, event, details)
        except Exception as exc:
            self._recording_link = dict(self._recording_link or {}, error=str(exc))
            # A recording failure must never interfere with closing the valve.
            # Reject an automatic shot before opening if fresh video is absent.
            if event in ('pulse_intent', 'before_open') and self._attempt_action == 'automatic':
                raise ControlError('automatic firing requires current video: ' + str(exc)) from exc

    def firing_history(self):
        if not self._history_ready:
            return dict(available=False, error=self._history_error)
        try:
            return dict(self.history.snapshot(), available=True, error=self._history_error)
        except Exception as exc:
            return dict(available=False, error=str(exc))

    def _begin_attempt(self, action, payload):
        self._attempt = None
        self._recording_link = None
        self._attempt_action = action
        if action not in ('automatic', 'fire', 'water_test', 'dry_sequence'):
            return
        details = dict(action=action, dry_run=action == 'dry_sequence' or
                       (action == 'automatic' and self.automatic.dry_run),
                       minimum_confidence=self.automatic.minimum_confidence,
                       requested_pulse_seconds=self.config.pulse_seconds)
        if action == 'automatic':
            detections = payload.get('output', {}).get('detections', [])
            # Ordinary empty/below-threshold frames must not bury actual attempts.
            candidates = [d for d in detections if d['confidence'] >= self.automatic.minimum_confidence]
            if not candidates:
                return
            best = max(candidates, key=lambda d: d['confidence'])
            details.update(confidence=best['confidence'], native_box=list(best['xyxy']),
                           source_sequence=payload['source_sequence'],
                           source_generation=payload['camera_generation'],
                           source_wall_time=payload.get('source_wall_time'))
        try:
            self._attempt = self.history.begin(**details)
        except Exception as exc:
            self._history_error = str(exc)
            self._fault = 'Firing history write failed; firing inhibited'
            self._automatic_inhibited = True
            raise ControlError(self._fault) from exc
        if action != 'automatic':
            self._record_event('attempt_started', details)

    def _target(self, observation):
        if not observation or observation.get('dropped_reason'):
            raise ControlError('no valid detector result')
        output = observation['output']
        age = self._clock() - _number(observation['source_monotonic'], 'source time')
        if not output['available'] or output.get('error') or not 0 <= age <= self._maximum_age:
            raise ControlError('detector result unavailable or stale')
        width, height = observation['native_width'], observation['native_height']
        candidates = []
        for d in output['detections']:
            score = _number(d['confidence'], 'confidence')
            box = tuple(d['xyxy'])
            zone = self._watch_zone.test_box(box, width, height)
            if self.automatic.minimum_confidence <= score <= 1 and zone['watch_zone_inside']:
                candidates.append(d)
        if not candidates:
            raise ControlError('no qualifying squirrel box')
        # Same convention as the recording observer: highest score; stable first on a tie.
        best = max(candidates, key=lambda d: d['confidence'])
        box = tuple(best['xyxy'])
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        camera = self._camera()
        if (camera['width'], camera['height'], camera['frame_generation']) != (
                width, height, observation['camera_generation']):
            raise ControlError('camera source changed')
        aim = self.store.aim(x, y, width, height)
        return dict(pixel=(x, y), box=box, aim=aim, confidence=best['confidence'],
                    generation=observation['camera_generation'], sequence=observation['source_sequence'])

    def observe(self, observation):
        """A bounded, non-blocking handoff from the existing inference callback."""
        if not self.config.enabled or not self.automatic.enabled or self._automatic_inhibited:
            return
        with self._condition:
            identity = (observation.camera_generation, observation.source_sequence)
            if (not self._started or self._stop.is_set() or self._fault or self._calibrating
                    or self._active or self._job is not None or self._clock() < self._cooldown_until
                    or identity <= self._last_identity):
                return
            self._last_identity = identity
            self._job = ('automatic', asdict(observation))
            self._condition.notify()

    def _final_target(self, original):
        # Called outside our status lock, avoiding lock inversion with inference callbacks.
        status = self._detector_status()
        if status.get('paused') or not status.get('result_available'):
            raise ControlError('target unavailable before pulse')
        current = self._target(status.get('last_result'))
        x, y = original['pixel']
        box = current['box']
        if (current['generation'] != original['generation']
                or current['sequence'] < original['sequence']
                or not box[0] <= x <= box[2] or not box[1] <= y <= box[3]):
            raise ControlError('target moved away from aimed pixel')
        if self._calibrating or self._automatic_inhibited or self._stop.is_set():
            raise ControlError('automatic firing inhibited')

    def _sequence(self, target, *, automatic, dry):
        attempted = False
        try:
            aim = target['aim']
            self._move(aim.pan, aim.tilt)
            if automatic:
                self._final_target(target)
            else:
                camera = self._camera()
                self.store.aim(*target['pixel'], camera['width'], camera['height'])
            attempted = True
            if dry:
                self._set_state('FIRING')
                self._wait(self.config.pulse_seconds)  # Same sequence; GPIO never opens.
            else:
                self._pulse((lambda: self._final_target(target)) if automatic else None)
            return dict(action='automatic' if automatic else 'dry_sequence', outcome='dry_run' if dry else 'fired',
                        pixel=list(target['pixel']), pan=aim.pan, tilt=aim.tilt)
        finally:
            self._close_valve()  # If closure fails, do not intentionally move a possibly open valve.
            if not self._stop.is_set():
                p = self.config.pan_tilt
                self._move(p.park_pan, p.park_tilt, parking=True)
            if attempted:
                self._cooldown_until = self._clock() + self.config.cooldown_seconds

    def _execute(self, action, payload):
        if action == 'automatic':
            target = self._target(payload)
            self._audit('target_selected', pixel=list(target['pixel']),
                        pan=target['aim'].pan, tilt=target['aim'].tilt,
                        confidence=target['confidence'], native_box=list(target['box']))
            self._final_target(target)
            # Reserve video after the existing target checks; pre-roll retains the lead-in.
            self._record_event('attempt_started', dict(action='automatic',
                source_sequence=payload['source_sequence'],
                source_generation=payload['camera_generation'],
                source_wall_time=payload.get('source_wall_time')))
            return self._sequence(target, automatic=True, dry=self.automatic.dry_run)
        if action in ('dry_aim', 'dry_sequence'):
            camera = self._camera()
            pixel = self._pixel(payload['pixel_x'], payload['pixel_y'], camera)
            aim = self.store.aim(*pixel, camera['width'], camera['height'])
            if action == 'dry_sequence':
                return self._sequence(dict(pixel=pixel, aim=aim), automatic=False, dry=True)
            self._move(aim.pan, aim.tilt)
            self._dry_aim = (pixel, camera['width'], camera['height'], camera['frame_generation'])
            return dict(action=action, outcome='aimed_without_water', pixel=list(pixel), pan=aim.pan, tilt=aim.tilt)
        if action == 'water_test':
            # An explicit supply check uses the serialized valve worker without
            # requiring a camera target or initializing/moving the servo driver.
            try:
                self._pulse(require_settled=False)
            finally:
                try:
                    self._close_valve()
                finally:
                    self._cooldown_until = self._clock() + self.config.cooldown_seconds
            return dict(action=action, outcome='manual_pulse_closed')
        if action == 'fire':
            camera = self._camera()
            if self._calibrating:
                # Attended calibration must allow aim/pulse/observe before the
                # landing pixel is known. Only saving requires a native reticle.
                if self._reticle is not None:
                    if self._reticle_source != (camera['width'], camera['height'], camera['frame_generation']):
                        raise ControlError('camera changed since reticle selection')
                    self._pixel(*self._reticle, camera)
            else:
                if self._dry_aim is None:
                    raise ControlError('dry aim inside calibration before manual fire')
                pixel, width, height, generation = self._dry_aim
                if (width, height, generation) != (camera['width'], camera['height'], camera['frame_generation']):
                    raise ControlError('camera changed since dry aim')
                self.store.aim(*pixel, width, height)
            try:
                position = self._servo.position if self._servo is not None else None
                self._audit('manual_aim', pan=position.pan if position else None,
                            tilt=position.tilt if position else None, calibration_mode=self._calibrating)
                self._pulse()
            finally:
                self._close_valve()
                if not self._calibrating:
                    self._dry_aim = None
                    if not self._stop.is_set():
                        p = self.config.pan_tilt
                        self._move(p.park_pan, p.park_tilt, parking=True)
                self._cooldown_until = self._clock() + self.config.cooldown_seconds
            # Calibration manual fire deliberately holds its commanded aim for trimming.
            return dict(action=action, outcome='manual_pulse_closed')
        self._dry_aim = None
        p = self.config.pan_tilt
        if action == 'move':
            position = self._servo.position if self._servo is not None else None
            if position is None:
                raise ControlError('command CENTER or PARK before relative movement')
            signs = dict(left=(1, 0), right=(-1, 0), up=(0, 1), down=(0, -1))
            dx, dy = signs[payload['direction']]
            pan, tilt = position.pan + dx * payload['step'], position.tilt + dy * payload['step']
        elif action == 'select':
            pan, tilt = payload['pan'], payload['tilt']
        else:
            pan, tilt = (p.park_pan, p.park_tilt) if action == 'park' else (p.pan_center, p.tilt_center)
        position = self._move(pan, tilt, parking=action == 'park')
        return dict(action=action, outcome='position_commanded', **asdict(position))

    def _run(self):
        while not self._stop.is_set():
            with self._condition:
                while self._job is None and not self._stop.is_set():
                    remaining = self._cooldown_until - self._clock()
                    self._state = 'COOLDOWN' if remaining > 0 else 'IDLE'
                    self._condition.wait(timeout=remaining if remaining > 0 else None)
                if self._stop.is_set():
                    return
                (action, payload), self._job = self._job, None
                self._active = True
            try:
                result = dict(action=action, outcome='error', error='action interrupted')
                self._begin_attempt(action, payload)
                result = self._execute(action, payload)
            except (ControlError, CalibrationError) as exc:
                result = dict(action=action, outcome='rejected', error=str(exc))
            except BaseException as exc:
                self._fault = str(exc)
                self._automatic_inhibited = True
                result = dict(action=action, outcome='error', error=str(exc))
            finally:
                try:
                    self._close_valve()
                except Exception as exc:
                    self._fault = f'valve close failed: {exc}'
                    self._automatic_inhibited = True
                    result = dict(result, outcome='error', error=self._fault)
                try:
                    self._audit('attempt_finished', **result, finished_at=utc_now())
                except ControlError:
                    result = dict(result, history_error=self._history_error)
                self._attempt = None
                with self._condition:
                    self._last = result
                    self._active = False
                    self._state = 'COOLDOWN' if self._clock() < self._cooldown_until else 'IDLE'
                    self._condition.notify_all()

    def request(self, action, **payload):
        with self._condition:
            if not self.config.enabled or not self._started or self._stop.is_set():
                raise ControlError('physical control disabled or not started')
            if self._fault:
                raise ControlError(f'control fault; restart required: {self._fault}')
            if self._active or self._job is not None:
                raise ControlError('control busy')
            if action == 'enter_calibration':
                self._calibrating = self._automatic_inhibited = True
                self._dry_aim = None
                return self.status()
            if action in ('select', 'reticle', 'lock', 'complete', 'reset', 'move', 'center') and not self._calibrating:
                raise ControlError('calibration mode is closed')
            if action in ('fire', 'water_test', 'dry_sequence') and self._clock() < self._cooldown_until:
                raise ControlError('shot cooldown active')
            if action in ('fire', 'water_test') and not self.config.valve_enabled:
                raise ControlError('water is disabled')
            if action == 'select':
                number = payload.get('point')
                if type(number) is not int or not 1 <= number <= 9:
                    raise ControlError('point must be 1..9')
                self._selected = number
                camera = self._camera()
                point = next((p for p in self.store.status()['points'] if p['point'] == number), None)
                calibration = self.store.status()
                if point and (calibration['camera_width'], calibration['camera_height']) != (camera['width'], camera['height']):
                    raise ControlError('reset calibration for the new camera dimensions')
                self._reticle = (point['pixel_x'], point['pixel_y']) if point else None
                self._reticle_source = (camera['width'], camera['height'], camera['frame_generation']) if point else None
                if point is None:
                    return self.status()
                payload = point
            elif action in ('reticle', 'lock'):
                if payload.get('point') != self._selected:
                    raise ControlError('selected point changed; select it again')
                camera = self._camera()
                pixel = self._pixel(payload.get('pixel_x'), payload.get('pixel_y'), camera)
                if action == 'lock':
                    if self._reticle != pixel or self._reticle_source != (
                            camera['width'], camera['height'], camera['frame_generation']):
                        raise ControlError('confirm reticle in the current camera before locking')
                    position = self._servo.position if self._servo is not None else None
                    if position is None or not self._settled:
                        raise ControlError('command and settle an aim before locking')
                    relock = payload.get('relock', False)
                    if type(relock) is not bool:
                        raise ControlError('relock must be boolean')
                    self.store.lock_point(self._selected, *pixel, position.pan, position.tilt,
                                          camera['width'], camera['height'], relock=relock)
                self._reticle = pixel
                self._reticle_source = (camera['width'], camera['height'], camera['frame_generation'])
                return self.status()
            elif action == 'reset':
                if payload.get('confirm') is not True:
                    raise ControlError('reset requires explicit confirmation')
                camera = self._camera()
                self.store.reset(camera['width'], camera['height'])
                self._reticle = self._reticle_source = self._dry_aim = None
                self._selected = 1
                return self.status()
            elif action == 'complete':
                camera = self._camera()
                self.store.complete(camera['width'], camera['height'])
                self._calibrating = False
                self._automatic_inhibited = True  # Completing setup never arms water.
                self._reticle = None
                return self.status()
            elif action == 'move':
                step = _number(payload.get('step'), 'step')
                if step not in (5, 3, 1, .5) or payload.get('direction') not in ('up', 'down', 'left', 'right'):
                    raise ControlError('invalid D-pad direction or step')
            elif action in ('dry_aim', 'dry_sequence'):
                if self._calibrating:
                    raise ControlError('complete calibration before interpolation tests')
                self._pixel(payload.get('pixel_x'), payload.get('pixel_y'), self._camera())
            elif action not in ('park', 'center', 'fire', 'water_test'):
                raise ControlError('unknown control action')
            self._job = (action, payload)
            self._condition.notify()
            return self.status()

    def status(self, *, compact=False):
        history = self.firing_history()
        camera = {} if compact else self._camera_status()
        if not isinstance(camera, dict):
            camera = asdict(camera)
        with self._condition:
            position = self._servo.position if self._servo is not None else None
            calibration = self.store.status()
            if compact:
                calibration = {key: calibration.get(key) for key in ('complete', 'error')}
            return dict(enabled=self.config.enabled, valve_enabled=self.config.valve_enabled,
                        pulse_seconds=self.config.pulse_seconds,
                        automatic_fire_enabled=self.automatic.enabled and not self._automatic_inhibited,
                        automatic_fire_configured=self.automatic.enabled, dry_run=self.automatic.dry_run,
                        firing_history=history, firing_minimum_confidence=self.automatic.minimum_confidence,
                        state=self._state, busy=self._active or self._job is not None, error=self._fault,
                        last_result=self._last, position=asdict(position) if position else None,
                        position_basis='last commanded angles; no servo feedback',
                        cooldown_remaining_seconds=max(0, self._cooldown_until - self._clock()),
                        calibration_mode=self._calibrating, calibration=calibration,
                        selected_point=self._selected,
                        active_reticle=dict(zip(('pixel_x', 'pixel_y'), self._reticle)) if self._reticle else None,
                        camera={k: camera.get(k) for k in ('online', 'width', 'height', 'last_frame_age_seconds')},
                        worker_alive=bool(self._thread and self._thread.is_alive()))
