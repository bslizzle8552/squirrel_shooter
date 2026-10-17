from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from conftest import write_test_config
from squirrel_shooter.camera_service import FramePacket, CameraStatus
from squirrel_shooter.collector_app import CollectorRuntime, create_collector_app
from squirrel_shooter.collector_night import CollectorNightConfig, NightVisionMonitor
from squirrel_shooter.config import load_config, ConfigError
from squirrel_shooter.detector import DetectorConfig
from squirrel_shooter.inference_worker import LatestFrameInferenceWorker
from squirrel_shooter.collector_policy import AutomaticRecordingConfig
from test_recording import Clock, rig, captured, finished
from test_inference_worker import Detector, packet
from test_collector import forbid_controls
from test_collector_usability import observation


def source(seq, clock, spread=0, gen=1):
    frame = np.full((48, 64, 3), 60, np.uint8)
    frame[:, :, 1] += spread
    frame.flags.writeable = False
    return FramePacket(seq, frame, '2026-09-08T20:00:00Z', clock(), gen)


def feed(monitor, clock, start, seconds, spread):
    for i in range(seconds + 1):
        clock.now = start + i
        monitor.sample(source(int(clock.now), clock, spread))


def test_both_transitions_hysteresis_and_no_flapping():
    clock = Clock()
    monitor = NightVisionMonitor(CollectorNightConfig(enabled=True), clock=clock)
    assert monitor.status()['paused']
    feed(monitor, clock, 100, 4, 0)
    assert monitor.status()['mode'] == 'CHECKING'
    feed(monitor, clock, 105, 0, 0)
    assert monitor.status()['mode'] == 'NIGHT'
    # Brief color or ambiguous values cannot restart inference.
    feed(monitor, clock, 106, 5, 20)
    feed(monitor, clock, 112, 0, 5)
    feed(monitor, clock, 113, 14, 20)
    assert monitor.status()['mode'] == 'NIGHT'
    feed(monitor, clock, 128, 0, 20)
    assert monitor.status()['mode'] == 'DAY'
    feed(monitor, clock, 129, 3, 0)
    feed(monitor, clock, 133, 0, 5)
    assert monitor.status()['mode'] == 'DAY'
    feed(monitor, clock, 134, 5, 0)
    assert monitor.status()['mode'] == 'NIGHT'


def test_sample_budget_duplicates_gaps_generation_and_source_immutable():
    clock = Clock()
    monitor = NightVisionMonitor(CollectorNightConfig(enabled=True), clock=clock)
    p = source(1, clock)
    before = p.frame.tobytes()
    for _ in range(1000): monitor.sample(p)
    assert monitor.status()['sample_count'] == 1
    assert p.frame.tobytes() == before
    clock.now = 105
    monitor.sample(p)  # stale/duplicate cannot confirm a night transition
    assert monitor.status()['sample_count'] == 1
    feed(monitor, clock, 110, 4, 0)
    assert monitor.status()['mode'] == 'CHECKING'
    feed(monitor, clock, 115, 0, 0)
    assert monitor.status()['mode'] == 'NIGHT'
    feed(monitor, clock, 116, 15, 20)
    assert monitor.status()['mode'] == 'DAY'
    clock.now += 1
    monitor.sample(source(200, clock, 20, gen=2))
    assert monitor.status()['mode'] == 'CHECKING'


@pytest.mark.parametrize('value', [dict(enabled='yes'), dict(sample_seconds=.1), dict(enter_seconds=float('nan')),
                                 dict(color_p90=2), dict(exit_seconds=5, enter_seconds=6)])
def test_config_validation(tmp_path, value):
    import yaml
    p = write_test_config(tmp_path)
    raw = yaml.safe_load(p.read_text())
    raw['collector_night'] = value
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError): load_config(p)


def test_pending_and_inflight_results_cannot_trip_at_night_and_day_resumes():
    clock, entered, release = Clock(), threading.Event(), threading.Event()
    detector = Detector(lambda: (entered.set(), release.wait(3)))
    seen = []
    worker = LatestFrameInferenceWorker(None, detector, DetectorConfig(enabled=True), clock=clock, observer=seen.append)
    worker.submit(packet(1))
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    try:
        assert entered.wait(2)
        worker.submit(packet(2))
        worker.set_paused(True)
        release.set()
        thread.join(2)
        assert not seen and not worker.status()['pending_frame']
        for n in range(3, 100):
            assert not worker.submit(packet(n))
            assert not worker.run_once()
        assert len(detector.frames) == 1
        assert worker.status()['worker_state'] == 'night_paused'
        worker.set_paused(False)
        clock.now = 101
        assert worker.submit(packet(100, at=101)) and worker.run_once()
        assert len(detector.frames) == 2 and len(seen) == 1
    finally:
        release.set()
        thread.join(2)


def runtime_for(tmp_path, rig):
    config = load_config(write_test_config(tmp_path))
    config = replace(config, collector_night=CollectorNightConfig(enabled=True), detector=DetectorConfig(enabled=True),
                     automatic_recording=AutomaticRecordingConfig(enabled=True))
    service = rig.build(target_fps=8, automatic_tail_seconds=10)
    rig.camera.status = lambda: CameraStatus(True, 16, 12, 12, None)
    rig.camera.set_buffering_enabled = lambda enabled: None
    rig.camera.stop = lambda: None
    runtime = CollectorRuntime(config, camera=rig.camera, detector=Detector(), recording=service)
    runtime.night = NightVisionMonitor(config.collector_night, clock=rig.clock)
    runtime.worker._clock = rig.clock
    runtime.observer._clock = rig.clock
    return runtime, service


def test_runtime_night_blocks_manual_auto_evidence_preview_works_and_resumes(tmp_path, rig, monkeypatch):
    forbid_controls(monkeypatch)
    runtime, service = runtime_for(tmp_path, rig)
    rig.camera.mjpeg_frames = lambda **kwargs: iter([b'live camera'])
    app = create_collector_app(runtime)
    client = app.test_client()
    headers = {'X-Recording-Token': app.config['COLLECTOR_RECORDING_TOKEN']}
    try:
        for i in range(6):
            rig.clock.now = 100+i
            runtime._monitor_frame(source(i, rig.clock))
        assert runtime.status()['night_mode']['mode'] == 'NIGHT'
        assert client.post('/api/recording/start', headers=headers).status_code == 409
        runtime.observer.observe(observation(at=rig.clock(), seq=1))
        assert not service.status()['active']
        assert runtime.observer.status()['raw_diagnostics'] == []
        assert not list(service.directory.rglob('trip-*.png'))
        assert client.get('/preview.mjpg').data == b'live camera'
        page = client.get('/').data.decode()
        assert 'NIGHT MODE — DETECTION / RECORDING PAUSED' in page
        assert 'el(\'record\').disabled=paused' in page
        for i in range(16):
            rig.clock.now = 106+i
            runtime._monitor_frame(source(100+i, rig.clock, 20))
        assert not runtime.worker.status()['paused']
        assert client.post('/api/recording/start', headers=headers).status_code == 200
        captured(rig, service)
        client.post('/api/recording/stop', headers=headers)
        finished(service)
    finally: runtime.stop()


def test_existing_tail_finishes_without_extension_or_new_evidence(tmp_path, rig):
    runtime, service = runtime_for(tmp_path, rig)
    try:
        # Establish day through the same stable-state monitor, then an exact trip.
        for i in range(16):
            rig.clock.now = 100+i
            runtime._monitor_frame(source(i, rig.clock, 20))
        p = rig.camera.push()
        obs = replace(observation(at=rig.clock(), seq=p.sequence), source_wall_time=p.received_at)
        runtime.observer.observe(obs, p)
        assert runtime.observer.status()['qualifying_count'] == 1
        deadline = service.status()['automatic_reasons'][0]['until']
        for i in range(6):
            rig.clock.now = 116+i
            runtime._monitor_frame(source(100+i, rig.clock))
            captured(rig, service)
        assert runtime.worker.status()['paused'] and service.status()['active']
        count = len(list(service.directory.rglob('trip-*.png')))
        p = rig.camera.push()
        runtime.observer.observe(observation(at=rig.clock(), seq=p.sequence), p)
        assert service.status()['automatic_reasons'][0]['until'] == deadline
        assert len(list(service.directory.rglob('trip-*.png'))) == count == 1
        rig.clock.now = deadline-.1
        captured(rig, service)
        rig.clock.now = deadline+.1
        result = finished(service)
        assert result['status'] in ('complete', 'degraded') and not result['error']
        assert not result['active'] and result['pending_sessions'] == 0
        assert result['segments'] and all(s['decoded_frames'] > 0 for s in result['segments'])
    finally: runtime.stop()


def test_night_idle_skips_periodic_storage_work_and_wakes(rig, monkeypatch):
    service = rig.build()
    calls = []
    monkeypatch.setattr(service, '_storage_check', lambda **kwargs: calls.append(1))
    service.set_idle_requested(True)
    time.sleep(.15)
    before = len(calls)
    time.sleep(1.15)
    assert len(calls) == before
    service.set_idle_requested(False)
    service.record_manual()
    captured(rig, service)
    assert calls
