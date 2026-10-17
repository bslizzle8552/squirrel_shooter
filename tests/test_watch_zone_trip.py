import hashlib
import json
from dataclasses import replace
import cv2
import numpy as np
import pytest
from squirrel_shooter.watch_zone import WatchZoneConfig
from squirrel_shooter.collector_policy import SquirrelRecordingObserver,AutomaticRecordingConfig
from squirrel_shooter.recording import RecordingService,RecordingConfig,RecordingError
from squirrel_shooter.camera_service import FramePacket
from test_recording import Clock,Camera
from test_collector_usability import observation


@pytest.mark.parametrize('polygon',[[],[(0,0),(1,1)],[(0,0),(1,0),(2,1)],[(0,0),(float('nan'),0),(0,1)],[(0,0),(1,1),(0,1),(1,0)],[(0,0),(0,0),(1,1)]])
def test_invalid_polygons(polygon):
    with pytest.raises(ValueError):WatchZoneConfig(polygon=polygon)


def test_inside_boundary_scaling_and_nonmutating_overlay():
    z=WatchZoneConfig(polygon=((.1,.2),(.8,.2),(.8,.9),(.1,.9)))
    assert z.contains(.2,.3) and z.contains(.1,.5) and not z.contains(.9,.5)
    assert z.test_box((100,200,300,400),1280,720)['watch_zone_test_point']==[200,400]
    assert np.max(abs(z.pixels(1280,720)-2*z.pixels(640,360)))<=1
    frame=np.zeros((360,640,3),np.uint8);frame.flags.writeable=False
    rendered=z.overlay(frame)
    assert np.count_nonzero(rendered)>0 and np.count_nonzero(frame)==0


def setup(tmp_path, width=16, height=12):
    c=Clock();r=RecordingService(Camera(c),RecordingConfig(enabled=True,target_fps=8,minimum_free_megabytes=0),tmp_path,clock=c)
    r._ready=True
    z=WatchZoneConfig(polygon=((0,0),(.5,0),(.5,1),(0,1)))
    o=SquirrelRecordingObserver(r,AutomaticRecordingConfig(enabled=True),1.5,clock=c,watch_zone=z,require_evidence=True)
    return c,r,o


def test_all_exact_sources_survive_saturated_avi_queue_and_peak_last(tmp_path):
    c,r,o=setup(tmp_path)
    for i in range(25):
        c.now=100+i*.5
        obs=observation(seq=i+1,at=c.now,confidence=.43 if i==8 else .08)
        image=np.full((12,16,3),i,np.uint8);image.flags.writeable=False
        packet=FramePacket(i+1,image,obs.source_wall_time,c.now,1)
        o.observe(obs,packet)
        r._admit(r._active,packet)
    status=r.status();assert status['queue_dropped']>0
    assert status['qualifying_observation_count']==25
    assert status['maximum_squirrel_confidence']==.43
    assert len(list(tmp_path.rglob('trip-*.png')))==25
    for item in status['squirrel_observations']:
        path=tmp_path/status['session_id']/item['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['image_sha256']
        decoded=cv2.imread(str(path));assert np.all(decoded==item['source_sequence']-1)
        assert hashlib.sha256(decoded.tobytes()).hexdigest()==item['source_pixels_sha256']
        assert item['media_role']=='clean_detector_evidence' and item['watch_zone_inside']
    assert status['first_qualifying_observation']['source_sequence']==1
    assert status['last_qualifying_observation']['source_sequence']==25
    assert next(x for x in status['squirrel_observations'] if x['confidence']==.43)['new_peak']
    r._active.observation_count=2048;c.now+=.5
    o.observe(observation(seq=26,at=c.now),FramePacket(26,image,obs.source_wall_time,c.now,1))
    assert 'trip_evidence_session_limit' in o.status()['error']
    assert len(list(tmp_path.rglob('trip-*.png')))==25


def test_outside_never_trips_but_remains_diagnosable(tmp_path):
    c,r,o=setup(tmp_path)
    obs=observation();obs=replace(obs,output=replace(obs.output,detections=(replace(obs.output.detections[0],xyxy=(10,2,14,4)),)))
    o.observe(obs)
    assert not r.status()['active']
    assert o.status()['raw_diagnostics'][-1]['watch_zone_inside'] is False
    assert not o.status()['detected_recently']


def test_missing_or_failed_evidence_never_accepts_trip(tmp_path,monkeypatch):
    c,r,o=setup(tmp_path);obs=observation()
    o.observe(obs);assert o.status()['error']=='exact_trip_source_missing' and not r.status()['active']
    monkeypatch.setattr(cv2,'imencode',lambda *args:(False,None))
    o.observe(obs,FramePacket(1,np.zeros((12,16,3),np.uint8),obs.source_wall_time,100,1))
    assert o.status()['qualifying_count']==0
    assert r.status()['qualifying_observation_count']==0
    assert 'trip_evidence_failed' in r.status()['error']


def test_owner_zone_excludes_every_historical_right_edge_box():
    from pathlib import Path
    z=WatchZoneConfig(**json.loads(Path('config/watch-zone-current-site.json').read_text()))
    assert not z.test_box((1144.2314453125,310.11932373046875,1261.785400390625,362.3919372558594),1280,720)['watch_zone_inside']
    assert z.contains(.5,.95) and not z.contains(.5,.1)
    assert z.contains(.02,.87)  # Smooth outer left contour, not the accidental notch.


def test_gallery_serves_only_hashed_accepted_trip(tmp_path):
    from squirrel_shooter.collector_media import CollectorMedia
    c,r,o=setup(tmp_path);obs=observation()
    o.observe(obs,FramePacket(1,np.zeros((12,16,3),np.uint8),obs.source_wall_time,100,1))
    r._manifest(r._active)
    media=CollectorMedia(tmp_path);sid=r._active.id
    item=r.status()['last_qualifying_observation']
    path,entry=media.trip(sid,item['filename'])
    assert path.is_file() and media.recent()[0]['peak_trip']==item['filename']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):media.trip(sid,item['filename'])
    with pytest.raises(ValueError):media.trip(sid,'../trip.png')


def test_slow_persistence_keeps_video_and_retains_fresh_at_admission_still(tmp_path,monkeypatch):
    import squirrel_shooter.trip_evidence as trip
    c,r,o=setup(tmp_path);obs=observation();save=trip.preserve_trip
    def delayed(*args,**kwargs):
        result=save(*args,**kwargs);c.now+=4;return result
    monkeypatch.setattr(trip,'preserve_trip',delayed)
    o.observe(obs,FramePacket(1,np.zeros((12,16,3),np.uint8),obs.source_wall_time,100,1))
    assert r.status()['qualifying_observation_count']==1
    assert o.status()['error'] is None
    assert r.status()['error'] is None
    assert r.status()['evidence_delay_count']==1
    assert r.status()['source_identities'][0]['until']==103
    assert not r.status()['automatic_reasons']
    # The three-second source-time tail expired during the save. It is never
    # refreshed using the completion time, nor does the save erase queued video.
    r._expire(c.now)
    assert not r.status()['active']
    assert len(list(tmp_path.rglob('*.png')))==1


@pytest.mark.parametrize('dimensions', [(16, 12), (1280, 720)])
def test_level_one_png_preserves_source_pixels_file_hash_and_provenance(tmp_path, monkeypatch, dimensions):
    width, height = dimensions
    c, r, o = setup(tmp_path, width, height)
    obs = replace(observation(), native_width=width, native_height=height)
    image = np.random.default_rng(47).integers(0, 256, (height, width, 3), dtype=np.uint8)
    original_pixels = image.tobytes()
    image.flags.writeable = False
    encode, options = cv2.imencode, []
    def checked_encode(extension, pixels, params):
        options.append((extension, params))
        return encode(extension, pixels, params)
    monkeypatch.setattr(cv2, 'imencode', checked_encode)
    o.observe(obs, FramePacket(1, image, obs.source_wall_time, 100, 1))
    assert options == [('.png', [cv2.IMWRITE_PNG_COMPRESSION, 1])]
    assert o.status()['qualifying_count'] == 1
    entry = r.status()['last_qualifying_observation']
    path = tmp_path / entry['session_id'] / entry['filename']
    saved = path.read_bytes()
    sidecar = json.loads(path.with_name(path.name + '.json').read_text())
    assert saved.startswith(b'\x89PNG\r\n\x1a\n')
    assert cv2.imread(str(path)).tobytes() == original_pixels == image.tobytes()
    assert hashlib.sha256(saved).hexdigest() == entry['image_sha256'] == sidecar['image_sha256']
    assert hashlib.sha256(original_pixels).hexdigest() == entry['source_pixels_sha256'] == sidecar['source_pixels_sha256']
    assert sidecar['source_sequence'] == 1 and sidecar['source_generation'] == 1
    assert sidecar['source_monotonic'] == 100 and sidecar['source_wall_time'] == obs.source_wall_time
    assert sidecar['media_role'] == 'clean_detector_evidence'
    r._manifest(r._active)
    from squirrel_shooter.collector_media import CollectorMedia
    served, attestation = CollectorMedia(tmp_path).trip(entry['session_id'], entry['filename'])
    assert served.read_bytes() == saved and attestation['image_sha256'] == entry['image_sha256']
