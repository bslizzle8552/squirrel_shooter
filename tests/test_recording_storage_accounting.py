"""Disk accounting and terminal-quota behavior, without camera or Pi access."""
from types import SimpleNamespace
import json
import threading

import pytest

from squirrel_shooter.recording import RecordingConfig, RecordingError, RecordingService
from test_recording import Camera, Clock, wait_for, rig, captured


def recorder(directory, budget_bytes=1024):
    clock = Clock()
    return RecordingService(
        Camera(clock),
        RecordingConfig(enabled=True, minimum_free_megabytes=0,
                        storage_budget_megabytes=budget_bytes / 1024**2),
        directory, clock=clock, disk_usage=lambda _: SimpleNamespace(free=1024**3))


def test_storage_counts_existing_files_not_buffers_or_lifetime_writes(tmp_path):
    session = tmp_path / 'session'
    session.mkdir()
    files = {
        'segment-0000.incomplete.avi': 100,
        'segment-0000.presentation.jsonl': 20,
        'trip-g1-s1.png': 200,
        'trip-g1-s1.png.json': 30,
        'trip-g1-s2.png.pending': 40,
        'session.json.tmp': 50,
    }
    for name, size in files.items():
        (session / name).write_bytes(b'x' * size)
    service = recorder(tmp_path)
    service._buffered_bytes = 128 * 1024**2
    for _ in range(3):
        service._storage_check()
        assert service.status()['storage_bytes'] == sum(files.values())
    (session / 'segment-0000.incomplete.avi').rename(session / 'segment-0000.avi')
    service._storage_check()
    assert service.status()['storage_bytes'] == 440
    (session / 'trip-g1-s1.png').unlink()
    (session / 'session.json.tmp').write_bytes(b'new')
    service._storage_check()
    assert service.status()['storage_bytes'] == 193
    with (session / 'segment-0000.avi').open('ab') as stream:
        stream.write(b'additional')
    service._storage_check()
    assert service.status()['storage_bytes'] == 203
    restarted = recorder(tmp_path)
    restarted._recover()
    assert restarted.status()['storage_bytes'] == 203


@pytest.mark.parametrize('size,exhausted', [(1023, False), (1024, True), (1025, True)])
def test_storage_quota_boundary_preserves_evidence(tmp_path, size, exhausted):
    path = tmp_path / 'evidence.png'
    contents = b'x' * size
    path.write_bytes(contents)
    service = recorder(tmp_path)
    if exhausted:
        with pytest.raises(RecordingError, match='recording_storage_budget_exhausted'):
            service._recover()
    else:
        service._recover()
    assert service.status()['storage_bytes'] == size
    assert path.read_bytes() == contents


def test_quota_failure_latches_old_size_after_removal_until_new_service(tmp_path):
    """Reproduce an exhausted dashboard alongside a now-under-budget directory."""
    path = tmp_path / 'evidence.png'
    path.write_bytes(b'x' * 1024)
    service = recorder(tmp_path)
    restarted = None
    try:
        service.start()
        wait_for(lambda: service.status()['last_error'])
        service._encoder.join(timeout=2)
        assert not service._encoder.is_alive()
        path.unlink()  # Only a test-owned temporary file.
        assert sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file()) == 0
        state = service.status()
        assert state['storage_bytes'] == 1024 and not state['ready']
        with pytest.raises(RecordingError, match='recording_storage_budget_exhausted'):
            service.record_manual()
        service.start()  # start() on the same object does not recreate its encoder.
        assert not service._encoder.is_alive()
        restarted = recorder(tmp_path)
        restarted.start()
        wait_for(lambda: restarted.status()['ready'])
        assert restarted.status()['storage_bytes'] == 0
        assert restarted.status()['last_error'] is None
    finally:
        service.stop()
        if restarted:
            restarted.stop()


@pytest.fixture
def quota_failure(tmp_path):
    path = tmp_path / 'test-evidence.png'
    path.write_bytes(b'x' * (64 * 1024))
    service = recorder(tmp_path, budget_bytes=64 * 1024)
    service.start()
    wait_for(lambda: service.status()['last_error'])
    service._encoder.join(timeout=2)
    assert not service._encoder.is_alive()
    yield service, path
    service.stop()


def assert_admission_blocked(service):
    with pytest.raises(RecordingError):
        service.record_manual()
    with pytest.raises(RecordingError):
        service.extend_automatic_recording(event_id='test', reason='test', observed_monotonic=service._clock())


def test_deliberate_recovery_rescans_and_replaces_encoder_then_records(quota_failure):
    service, path = quota_failure
    encoder, collector = service._encoder, service._collector
    service._storage_bytes = 0  # A cached under-budget total must never authorize recovery.
    assert not service.recover_storage_quota()
    assert service.status()['storage_bytes'] == 64 * 1024
    assert_admission_blocked(service)
    path.unlink()
    assert service.recover_storage_quota()
    wait_for(lambda: service.status()['ready'])
    assert service._encoder is not encoder and service._encoder.is_alive()
    assert service._collector is collector and collector.is_alive()
    state = service.status()
    assert state['last_error'] is None and state['storage_bytes'] == 0
    assert state['storage']['recovery_count'] == 1
    assert state['storage']['quota_error_state'] == 'historical'
    assert state['storage']['last_quota_error']['measurement']['bytes'] == 64 * 1024
    sid = service.record_manual()['session_id']
    result = service.extend_automatic_recording(event_id='test', reason='test', observed_monotonic=service._clock())
    assert result['session_id'] == sid and result['automatic_active']
    service.camera.push()
    wait_for(lambda: service.status().get('written_frames', 0) > 0)


@pytest.mark.parametrize('failure', [OSError('disk disconnected'), RecordingError('insufficient_free_disk'),
                                    RuntimeError('recording_storage_budget_exhausted')])
def test_unrelated_terminal_errors_never_recover(tmp_path, failure):
    service = recorder(tmp_path)
    def fail(_):
        raise failure
    service._disk_usage = fail
    try:
        service.start()
        wait_for(lambda: service.status()['last_error'])
        service._encoder.join(timeout=2)
        encoder = service._encoder
        prior = service.status()['last_error']
        def forbidden(_):
            pytest.fail('unrelated terminal failure attempted a recovery scan')
        service._disk_usage = forbidden
        assert not service.recover_storage_quota()
        assert service._encoder is encoder and service.status()['last_error'] == prior
        assert_admission_blocked(service)
    finally:
        service.stop()


def test_recovery_io_failure_is_latched_even_if_disk_later_returns(quota_failure):
    service, path = quota_failure
    path.unlink()
    def denied(_):
        raise PermissionError('recovery scan denied')
    service._disk_usage = denied
    assert not service.recover_storage_quota()
    assert service.status()['storage']['recovery_state'] == 'blocked'
    assert 'recovery scan denied' in service.status()['storage']['recovery_error']
    service._disk_usage = lambda _: SimpleNamespace(free=1024**3)
    assert not service.recover_storage_quota()
    assert_admission_blocked(service)


def test_recovery_requires_minimum_free_space(quota_failure):
    from dataclasses import replace
    service, path = quota_failure
    path.unlink()
    service.config = replace(service.config, minimum_free_megabytes=1)
    service._disk_usage = lambda _: SimpleNamespace(free=1024**2-1)
    assert not service.recover_storage_quota()
    state = service.status()['storage']
    assert state['bytes'] == 0 and not state['recovery_eligible']
    assert state['recovery_state'] == 'waiting_for_free_space'
    assert_admission_blocked(service)
    service._disk_usage = lambda _: SimpleNamespace(free=1024**2)
    assert service.recover_storage_quota()
    wait_for(lambda: service.status()['ready'])


@pytest.mark.parametrize('attribute,value', [('_writer', object()), ('_segment', {}),
    ('_segment_session', object()), ('_presentation', object()), ('_held_packet', object()),
    ('_evidence_inflight', 1), ('_buffered_bytes', 1)])
def test_recovery_refuses_inflight_resources(quota_failure, attribute, value):
    service, path = quota_failure
    path.unlink()
    original = getattr(service, attribute)
    setattr(service, attribute, value)
    try:
        assert not service.recover_storage_quota()
        assert service.status()['storage']['recovery_state'] == 'waiting_for_finalization'
        assert not service.status()['storage']['recovery_eligible']
        assert_admission_blocked(service)
    finally:
        setattr(service, attribute, original)


def test_recovery_refuses_pending_session_and_queued_packet(quota_failure):
    from squirrel_shooter.recording import _Session
    service, path = quota_failure
    path.unlink()
    session = _Session('f'*32, 100, '2026-09-14T12:00:00Z', 700, state='finalizing')
    with service._lock:
        service._sessions.append(session)
    try:
        assert not service.recover_storage_quota()
    finally:
        with service._lock:
            service._sessions.clear()
    service._queue.put_nowait(object())
    try:
        assert not service.recover_storage_quota()
    finally:
        service._queue.get_nowait()
        service._queue.task_done()


def test_recovery_waits_for_real_writer_finalization(rig):
    service = rig.build(storage_budget_megabytes=0.0625)
    sid = service.record_manual()['session_id']
    captured(rig, service)
    entered, release = threading.Event(), threading.Event()
    original_release = service._writer.release
    def delayed_release():
        entered.set()
        assert release.wait(5)
        original_release()
    service._writer.release = delayed_release
    path = service.directory / 'quota-test.bin'
    path.write_bytes(b'x' * 64 * 1024)
    try:
        assert entered.wait(3)
        encoder = service._encoder
        assert encoder.is_alive()
        path.unlink()
        assert not service.recover_storage_quota()
        assert service._encoder is encoder
        assert_admission_blocked(service)
    finally:
        release.set()
    encoder.join(timeout=3)
    assert not encoder.is_alive()
    manifest = service.directory / sid / 'session.json'
    before = manifest.read_bytes()
    assert json.loads(before)['status'] == 'error'
    assert service.recover_storage_quota()
    wait_for(lambda: service.status()['ready'])
    assert manifest.read_bytes() == before
    assert list(manifest.parent.glob('*.avi'))


def test_final_manifest_failure_blocks_quota_recovery(rig):
    service = rig.build(storage_budget_megabytes=0.0625)
    service.record_manual()
    captured(rig, service)
    original = service._manifest
    def fail_final_manifest(session):
        if service._last_error:
            raise OSError('final manifest failed')
        original(session)
    service._manifest = fail_final_manifest
    path = service.directory / 'quota-test.bin'
    path.write_bytes(b'x' * 64 * 1024)
    wait_for(lambda: service.status()['last_error'])
    service._encoder.join(timeout=3)
    path.unlink()
    service._manifest = original
    assert not service.recover_storage_quota()
    assert 'final manifest failed' in service.status()['storage']['recovery_error']


def test_storage_status_is_snapshot_and_distinguishes_history(quota_failure):
    service, path = quota_failure
    storage = service.status()['storage']
    assert storage['quota_error_state'] == 'current'
    assert storage['root'] == str(service.directory.absolute())
    assert storage['budget_bytes'] == 64 * 1024 and storage['basis'] == 'apparent_file_bytes'
    assert storage['measured_at'] and storage['measurement_age_seconds'] >= 0
    path.unlink()
    service._storage_snapshot['measured_monotonic'] -= 11
    stale = service.status()['storage']
    assert stale['bytes'] == 64 * 1024 and stale['measurement_stale']
    assert stale['quota_error_state'] == 'unverified' and not stale['recovery_eligible']
    service._storage_check()
    current = service.status()['storage']
    assert current['bytes'] == 0 and current['quota_error_state'] == 'historical'
    assert current['recovery_eligible']
    assert_admission_blocked(service)  # Measurement alone is not lifecycle recovery.


def test_missing_recordings_root_never_counts_as_empty(tmp_path):
    service = recorder(tmp_path / 'missing')
    with pytest.raises(FileNotFoundError):
        service._storage_check()
    storage = service.status()['storage']
    assert storage['bytes'] is None and storage['measurement_stale']
    assert 'FileNotFoundError' in storage['measurement_error']


def test_periodic_rechecks_are_rate_limited(tmp_path, monkeypatch):
    import squirrel_shooter.recording as module
    service = recorder(tmp_path)
    service._last_error = 'RecordingError: recording_storage_budget_exhausted'
    now, calls = [10.0], []
    monkeypatch.setattr(module, 'monotonic', lambda: now[0])
    monkeypatch.setattr(service, 'recover_storage_quota', lambda: calls.append(now[0]))
    for _ in range(20):
        service._maybe_recover_storage_quota()
    assert calls == [10.0]
    now[0] = 19.99
    service._maybe_recover_storage_quota()
    assert len(calls) == 1
    now[0] = 20
    service._maybe_recover_storage_quota()
    assert calls == [10.0, 20]


def test_capture_loop_recovers_without_dashboard_or_process_restart(quota_failure):
    service, path = quota_failure
    path.unlink()
    service._next_quota_recheck = 0  # Advance the maintenance deadline without a ten-second sleep.
    service._wake.set()
    wait_for(lambda: service.status()['ready'])
    assert service.status()['storage']['recovery_count'] == 1


def test_replacement_startup_rechecks_quota_before_admission(quota_failure):
    service, path = quota_failure
    path.unlink()
    entered, release = threading.Event(), threading.Event()
    recover = service._recover
    def pause_startup():
        entered.set()
        assert release.wait(5)
        recover()
    service._recover = pause_startup
    try:
        assert service.recover_storage_quota()
        assert entered.wait(2)
        assert service.status()['storage']['recovery_state'] == 'restarting'
        assert_admission_blocked(service)
        path.write_bytes(b'x' * 64 * 1024)
    finally:
        release.set()
    service._encoder.join(timeout=3)
    assert_admission_blocked(service)
    assert service.status()['storage']['recovery_count'] == 0
    service._recover = recover
    path.unlink()
    assert service.recover_storage_quota()
    wait_for(lambda: service.status()['ready'])


def test_shutdown_during_recovery_scan_does_not_restart(quota_failure):
    service, path = quota_failure
    path.unlink()
    scan = service._storage_check
    entered, release = threading.Event(), threading.Event()
    result = []
    def delayed_scan():
        entered.set()
        assert release.wait(5)
        scan()
    service._storage_check = delayed_scan
    worker = threading.Thread(target=lambda: result.append(service.recover_storage_quota()))
    worker.start()
    try:
        assert entered.wait(2)
        encoder = service._encoder
        service.stop()
    finally:
        release.set()
        worker.join(timeout=3)
    assert result == [False]
    assert service._encoder is encoder and not service.status()['ready']


def test_concurrent_recovery_requests_launch_only_one_encoder(quota_failure):
    service, path = quota_failure
    path.unlink()
    scan = service._storage_check
    entered, release = threading.Event(), threading.Event()
    result = []
    def delayed_scan():
        entered.set()
        assert release.wait(5)
        scan()
    service._storage_check = delayed_scan
    worker = threading.Thread(target=lambda: result.append(service.recover_storage_quota()))
    worker.start()
    try:
        assert entered.wait(2)
        assert not service.recover_storage_quota()
    finally:
        release.set()
        worker.join(timeout=3)
    assert result == [True]
    wait_for(lambda: service.status()['ready'])
    assert service.status()['storage']['recovery_count'] == 1
