"""Pure nine-point persistence/geometry checks; no camera, servos, or GPIO."""
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from squirrel_shooter import collector_calibration as calibration
from squirrel_shooter.collector_calibration import CalibrationError, CalibrationStore
from squirrel_shooter.pan_tilt import PanTiltConfig, PanTiltPosition


SIZE = (1280, 720)


def grid():
    return [(row * 3 + col + 1, 100 + 100 * col, 100 + 100 * row,
             (150, 90, 30)[col], (70, 110, 150)[row]) for row in range(3) for col in range(3)]


def store(tmp_path, records=None, *, complete=False):
    result = CalibrationStore(tmp_path / 'calibration.json', PanTiltConfig())
    for record in records if records is not None else grid():
        result.lock_point(*record, *SIZE)
    if complete:
        result.complete(*SIZE)
    return result


def test_lock_is_durable_and_all_nine_require_explicit_completion(tmp_path):
    result = CalibrationStore(tmp_path / 'calibration.json', PanTiltConfig())
    assert result.status() == dict(camera_width=None, camera_height=None, complete=False, points=[], error=None)
    for record in grid():
        state = result.lock_point(*record, *SIZE)
        assert not state['complete']
        assert CalibrationStore(result.path, PanTiltConfig()).status() == state
        with pytest.raises(CalibrationError, match='incomplete'):
            result.aim(150, 150, *SIZE)
    assert result.complete(*SIZE)['complete']
    assert CalibrationStore(result.path, PanTiltConfig()).status()['complete']
    assert json.loads(result.path.read_text())['camera_width'] == 1280


def test_missing_anchor_rejects_complete_without_replacing_file(tmp_path):
    result = store(tmp_path, grid()[:-1])
    before = result.path.read_bytes()
    with pytest.raises(CalibrationError, match='all nine'):
        result.complete(*SIZE)
    assert result.path.read_bytes() == before
    assert not result.status()['complete']


def test_relock_is_explicit_and_invalidates_completed_map(tmp_path):
    result = store(tmp_path, complete=True)
    before = result.path.read_bytes()
    with pytest.raises(CalibrationError, match='already locked'):
        result.lock_point(5, 200, 200, 91, 111, *SIZE)
    assert result.path.read_bytes() == before
    state = result.lock_point(5, 200, 200, 91, 111, *SIZE, relock=True)
    assert not state['complete'] and len(state['points']) == 9
    assert state['points'][4]['pan'] == 91
    with pytest.raises(CalibrationError, match='incomplete'):
        result.aim(200, 200, *SIZE)
    result.complete(*SIZE)
    assert result.aim(200, 200, *SIZE) == PanTiltPosition(91, 111)


def test_unlocked_point_cannot_be_relocked_and_bool_is_required(tmp_path):
    result = store(tmp_path, [])
    with pytest.raises(CalibrationError, match='not locked'):
        result.lock_point(*grid()[0], *SIZE, relock=True)
    with pytest.raises(CalibrationError, match='explicitly'):
        result.lock_point(*grid()[0], *SIZE, relock='false')


@pytest.mark.parametrize('failure', ['write', 'fsync', 'replace'])
def test_atomic_failure_preserves_original_file_memory_and_completion(tmp_path, monkeypatch, failure):
    result = store(tmp_path, complete=True)
    before, state = result.path.read_bytes(), result.status()

    def fail(*args, **kwargs):
        raise OSError('injected storage failure')

    if failure == 'write':
        real_fdopen = calibration.os.fdopen

        class FailingStream:
            def __init__(self, fd, mode):
                self.stream = real_fdopen(fd, mode)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            write = fail

        monkeypatch.setattr(calibration.os, 'fdopen', FailingStream)
    else:
        monkeypatch.setattr(calibration.os, failure, fail)
    with pytest.raises(OSError, match='injected'):
        result.lock_point(5, 200, 200, 91, 111, *SIZE, relock=True)
    assert result.path.read_bytes() == before
    assert result.status() == state
    assert result.aim(200, 200, *SIZE) == PanTiltPosition(90, 110)
    assert list(tmp_path.glob('.*.tmp')) == []


def test_atomic_temporary_names_are_unique_and_fsync_precedes_replace(tmp_path, monkeypatch):
    events = []
    real_sync, real_replace = calibration.os.fsync, calibration.os.replace

    def sync(fd):
        events.append('fsync')
        real_sync(fd)

    def replace_file(source, destination):
        assert events[-1] == 'fsync'
        assert Path(source).parent == Path(destination).parent
        events.append(str(source))
        real_replace(source, destination)

    monkeypatch.setattr(calibration.os, 'fsync', sync)
    monkeypatch.setattr(calibration.os, 'replace', replace_file)
    result = store(tmp_path, grid()[:2])
    assert len(set(events[1::2])) == 2
    assert len(result.status()['points']) == 2


@pytest.mark.parametrize('pixel', [(150.5, 150.25), (100, 250), (200, 150), (300, 300)])
def test_subpixel_centers_and_shared_outer_edges_map_without_extrapolation(tmp_path, pixel):
    result = store(tmp_path, complete=True)
    target = result.aim(*pixel, *SIZE)
    assert target.pan == pytest.approx(210 - .6 * pixel[0])
    assert target.tilt == pytest.approx(30 + .4 * pixel[1])


def test_every_anchor_is_exact_and_skewed_real_field_cells_interpolate(tmp_path):
    records = [(1, 446, 172, 84, 79), (2, 668, 180, 74, 78), (3, 893, 192, 61, 75),
               (4, 448, 227, 84, 93), (5, 640, 246, 70, 88), (6, 1003, 242, 58, 88),
               (7, 139, 436, 101, 109), (8, 693, 434, 73, 109), (9, 1121, 412, 48, 104)]
    result = store(tmp_path, records, complete=True)
    for _, x, y, pan, tilt in records:
        assert result.aim(x, y, *SIZE) == PanTiltPosition(pan, tilt)
    weights = (.28, .12, .18, .42)  # u=.3, v=.6, ordered TL/TR/BR/BL.
    for ids in ((1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)):
        corners = [records[number - 1] for number in ids]
        x, y, pan, tilt = [sum(weight * point[index] for weight, point in zip(weights, corners))
                           for index in (1, 2, 3, 4)]
        target = result.aim(x, y, *SIZE)
        assert target.pan == pytest.approx(pan, abs=1e-6)
        assert target.tilt == pytest.approx(tilt, abs=1e-6)


@pytest.mark.parametrize('pixel', [(99.99, 150), (301, 200), (200, 99.99), (200, 301),
                                   (-1, 150), (1280, 150), (150, 720)])
def test_outside_region_or_frame_rejected(tmp_path, pixel):
    with pytest.raises(CalibrationError, match='outside'):
        store(tmp_path, complete=True).aim(*pixel, *SIZE)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, True, '150'])
def test_nonfinite_and_non_numeric_targets_fail_closed(tmp_path, value):
    with pytest.raises(CalibrationError, match='finite'):
        store(tmp_path, complete=True).aim(value, 150, *SIZE)


@pytest.mark.parametrize('change', ['duplicate', 'collinear', 'folded', 'crossed_boundary', 'concave'])
def test_degenerate_folded_and_overlapping_grid_rejected(tmp_path, change):
    records = grid()
    if change == 'duplicate':
        records[4] = (5, 100, 100, 90, 110)
    elif change == 'collinear':
        records = [(i + 1, 100 + i * 20, 100, 90, 110) for i in range(9)]
    elif change == 'folded':
        records[4] = (5, 350, 200, 90, 110)
    elif change == 'crossed_boundary':
        records[1], records[7] = (2, 200, 300, 90, 70), (8, 200, 100, 90, 150)
    else:
        records[4] = (5, 105, 105, 90, 110)
    result = store(tmp_path, records)
    before = result.path.read_bytes()
    with pytest.raises(CalibrationError):
        result.complete(*SIZE)
    assert result.path.read_bytes() == before and not result.status()['complete']


def test_consistently_mirrored_grid_remains_a_valid_map(tmp_path):
    records = [(point, 400 - x, y, pan, tilt) for point, x, y, pan, tilt in grid()]
    result = store(tmp_path, records, complete=True)
    assert result.aim(250, 150, *SIZE) == PanTiltPosition(120, 90)


@pytest.mark.parametrize('operation', ['lock', 'complete', 'aim'])
def test_dimension_mismatch_rejects_without_rescaling(tmp_path, operation):
    result = store(tmp_path, complete=True)
    before = result.path.read_bytes()
    with pytest.raises(CalibrationError, match='dimensions mismatch'):
        if operation == 'lock':
            result.lock_point(*grid()[0], 640, 480, relock=True)
        elif operation == 'complete':
            result.complete(640, 480)
        else:
            result.aim(150, 150, 640, 480)
    assert result.path.read_bytes() == before
    state = result.reset(640, 480)
    assert state['camera_width'] == 640 and not state['complete'] and not state['points']


@pytest.mark.parametrize('record', [(True, 100, 100, 85, 85), (10, 100, 100, 85, 85),
                                    (1, 1280, 100, 85, 85), (1, 100, -1, 85, 85),
                                    (1, 100, 100, 151, 85), (1, 100, 100, 85, 69),
                                    (1, 100, 100, math.nan, 85)])
def test_invalid_anchor_is_rejected_before_any_file_write(tmp_path, record):
    result = store(tmp_path, [])
    with pytest.raises(CalibrationError):
        result.lock_point(*record, *SIZE)
    assert not result.path.exists() and not result.status()['points']


@pytest.mark.parametrize('corruption', ['json', 'schema_bool', 'complete_number', 'duplicate_id',
                                       'dimension', 'bad_angle', 'invalid_complete', 'fold'])
def test_malformed_file_is_inert_and_requires_explicit_reset(tmp_path, corruption):
    result = store(tmp_path, complete=True)
    raw = json.loads(result.path.read_text())
    if corruption == 'schema_bool':
        raw['schema_version'] = True
    elif corruption == 'complete_number':
        raw['complete'] = 1
    elif corruption == 'duplicate_id':
        raw['points'][1]['point'] = 1
    elif corruption == 'dimension':
        raw['camera_width'] = 200
    elif corruption == 'bad_angle':
        raw['points'][0]['pan'] = 999
    elif corruption == 'invalid_complete':
        raw['points'].pop()
    elif corruption == 'fold':
        raw['points'][4]['pixel_x'] = 350
    blob = b'{broken json' if corruption == 'json' else json.dumps(raw).encode()
    result.path.write_bytes(blob)
    loaded = CalibrationStore(result.path, PanTiltConfig())
    assert loaded.status()['error'] and not loaded.status()['complete']
    for operation in (lambda: loaded.aim(150, 150, *SIZE), lambda: loaded.complete(*SIZE),
                      lambda: loaded.lock_point(*grid()[0], *SIZE)):
        with pytest.raises(CalibrationError, match='reset'):
            operation()
    assert result.path.read_bytes() == blob
    assert loaded.reset(*SIZE)['error'] is None
    assert not loaded.status()['complete'] and loaded.status()['points'] == []


def test_saved_angles_are_revalidated_against_current_limits(tmp_path):
    result = store(tmp_path, complete=True)
    tighter = replace(PanTiltConfig(), pan_min=40, pan_max=140)
    assert CalibrationStore(result.path, tighter).status()['error']


def test_failed_completion_never_arms_in_memory_or_on_disk(tmp_path, monkeypatch):
    result = store(tmp_path)
    before = result.path.read_bytes()

    def fail_replace(*args):
        raise OSError('replace failed')

    monkeypatch.setattr(calibration.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='replace failed'):
        result.complete(*SIZE)
    assert result.path.read_bytes() == before
    assert not result.status()['complete']
    assert not CalibrationStore(result.path, PanTiltConfig()).status()['complete']


def test_status_and_aim_use_detached_cached_map_without_disk_reads(tmp_path, monkeypatch):
    result = store(tmp_path, complete=True)
    status = result.status()
    status['points'][0]['pan'] = 999
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: pytest.fail('unexpected runtime file read'))
    monkeypatch.setattr(calibration, '_validated_cells', lambda *args: pytest.fail('unexpected runtime geometry validation'))
    assert result.status()['points'][0]['pan'] == 150
    assert result.aim(150.5, 150.25, *SIZE).pan == pytest.approx(119.7)
