from dataclasses import replace
import json
import threading
import time

import cv2
import numpy as np
import pytest

from conftest import write_test_config
from squirrel_shooter.collector_app import CollectorRuntime, create_collector_app
from squirrel_shooter.collector_media import CollectorMedia
from squirrel_shooter.collector_policy import AutomaticRecordingConfig, CollectorPreviewConfig, SquirrelRecordingObserver
from squirrel_shooter.config import ConfigError, load_config, CameraConfig
from squirrel_shooter.camera_service import CameraService
from squirrel_shooter.detector import DetectionOutput, DetectorConfig, SquirrelBox
from squirrel_shooter.inference_worker import LatestFrameInferenceWorker, DetectorObservation
from test_recording import rig, captured, finished
from test_inference_worker import Detector, IDENTITY, packet
from test_shared_runtime import ContinuousCapture, wait_until


def observation(seq=1, at=100, confidence=.03, available=True):
    return DetectorObservation(1,seq,16,12,at,'2026-09-08T12:00:00Z',at+.05,at+.2,.2,
        DetectionOutput(available,IDENTITY,(SquirrelBox((1,2,3,4),confidence),)))


@pytest.mark.parametrize('confidence,at,available',[(.019,100,True),(.03,98,True),(.03,101,True),(.03,100,False)])
def test_rejected_observation_does_not_start(rig,confidence,at,available):
    recorder=rig.build(automatic_tail_seconds=10)
    observer=SquirrelRecordingObserver(recorder,AutomaticRecordingConfig(enabled=True),1.5,clock=rig.clock)
    observer.observe(observation(confidence=confidence,at=at,available=available))
    assert not recorder.status()['active']


def test_worker_positive_starts_recording_without_ui_and_uses_source_time(rig):
    recorder=rig.build(automatic_tail_seconds=10)
    observer=SquirrelRecordingObserver(recorder,AutomaticRecordingConfig(enabled=True),1.5,clock=rig.clock)
    detector=Detector(hook=lambda:setattr(rig.clock,'now',100.2))
    worker=LatestFrameInferenceWorker(rig.camera,detector,DetectorConfig(enabled=True),clock=rig.clock,observer=observer.observe)
    worker.submit(packet(1));assert worker.run_once()
    state=recorder.status()
    assert state['active'] and state['automatic_reasons'][0]['until']==110
    assert state['source_identities'][0]['visit_id'] is None
    assert state['qualifying_observation_count']==1
    assert state['first_qualifying_observation']['source_sequence']==1
    assert state['first_qualifying_observation']['model_identity']['model_version']=='v3'


def test_repeated_results_join_manual_stop_preserves_auto_and_metadata(rig):
    recorder=rig.build(automatic_tail_seconds=10)
    observer=SquirrelRecordingObserver(recorder,AutomaticRecordingConfig(enabled=True),1.5,clock=rig.clock)
    observer.observe(observation())
    first=recorder.status();captured(rig,recorder)
    assert recorder.record_manual()['session_id']==first['session_id']
    rig.clock.now=101
    observer.observe(observation(seq=2,at=100.8,confidence=.6))
    observer.observe(observation(seq=2,at=100.8,confidence=.6))
    state=recorder.stop_manual()
    assert state['active'] and state['automatic_active'] and not state['manual_remaining_seconds']
    assert state['session_id']==first['session_id']
    assert state['automatic_reasons'][0]['until']==110.8
    assert state['qualifying_observation_count']==2 and state['extension_count']==1
    assert state['maximum_squirrel_confidence']==.6
    assert state['last_qualifying_observation']['native_box']==[1,2,3,4]
    rig.clock.now=111
    state=finished(recorder)
    retained=json.loads((recorder.directory/state['session_id']/'session.json').read_text())
    assert retained['manual_requested']
    assert retained['qualifying_observation_count']==2
    assert retained['first_qualifying_observation']['recording_only_confidence_threshold']==.02
    assert retained['end_monotonic']==110.8


def test_stale_inflight_result_never_calls_observer(rig):
    calls=[]
    worker=LatestFrameInferenceWorker(rig.camera,Detector(hook=lambda:setattr(rig.clock,'now',102)),
        DetectorConfig(enabled=True),clock=rig.clock,observer=calls.append)
    worker.submit(packet(1));worker.run_once()
    assert not calls and worker.status()['stale_after_inference_count']==1


def test_observer_failure_does_not_kill_inference(rig):
    def broken(_): raise RuntimeError('observer failed')
    worker=LatestFrameInferenceWorker(None,Detector(),DetectorConfig(enabled=True),clock=rig.clock,observer=broken)
    worker.submit(packet(1));assert worker.run_once()
    assert worker.status()['observer_error_count']==1
    assert worker.status()['result_available']


def test_observation_metadata_bounded_and_summary_survives_truncation(rig):
    recorder=rig.build(automatic_tail_seconds=10)
    observer=SquirrelRecordingObserver(recorder,AutomaticRecordingConfig(enabled=True),1.5,clock=rig.clock)
    for seq in range(2051):
        rig.clock.now=100+seq*.001
        observer.observe(observation(seq=seq,at=rig.clock.now,confidence=.9 if seq==0 else .03))
    state=recorder.status()
    assert state['qualifying_observation_count']==2051
    assert len(state['squirrel_observations'])==2048 and state['observations_omitted']==3
    assert state['maximum_squirrel_confidence']==.9
    assert state['first_qualifying_observation']['source_sequence']==0


@pytest.mark.parametrize('section,value',[('automatic_recording',{'enabled':True}),('automatic_recording',{'squirrel_confidence':float('nan')}),('collector_preview',{'maximum_fps':60}),('collector_preview',{'jpeg_quality':True})])
def test_collector_config_rejects_invalid_policy(tmp_path,section,value):
    import yaml
    path=write_test_config(tmp_path)
    raw=yaml.safe_load(path.read_text());raw[section]=value;path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):load_config(path)


def test_recording_policy_does_not_change_engagement_or_generic_tail(tmp_path):
    config=load_config(write_test_config(tmp_path))
    runtime=CollectorRuntime(config)
    try:
        assert runtime.recording.config.automatic_tail_seconds==10
        assert config.recording.automatic_tail_seconds==3
        assert config.automatic_recording.squirrel_confidence==.02
        assert runtime.config.auto_fire==config.auto_fire
    finally:runtime.stop()


def test_raw_preview_shares_one_owner_skips_backlog_and_sleeps(tmp_path,monkeypatch):
    opens=[]; released=threading.Event()
    def factory(_):opens.append(1);return ContinuousCapture(np.full((36,64,3),40,np.uint8),released)
    camera=CameraService(CameraConfig(0,64,36,30,tmp_path),capture_factory=factory,
        platform_checker=lambda:True,frame_buffer_seconds=0,jpeg_quality=65)
    camera.start(); stream=None
    try:
        wait_until(lambda:camera.status().frames_received>=3)
        time.sleep(.05);assert camera.status().dashboard_frames_encoded==0
        source=camera.wait_for_frame(-1,copy=False)
        stream=camera.mjpeg_frames(maximum_fps=2,annotated_only=False)
        first=next(stream);start=camera.status()
        time.sleep(1.1) # Viewer does not consume; capture and shared latest encoder keep going.
        current=camera.status();next(stream)
        assert current.frames_received>start.frames_received
        assert current.dashboard_frames_encoded-start.dashboard_frames_encoded<=2
        assert np.all(source.frame==40) and not source.frame.flags.writeable
        jpeg=first.split(b'\r\n\r\n',1)[1].rstrip(b'\r\n')
        decoded=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        assert decoded.shape==(36,64,3) and np.max(np.abs(decoded.astype(int)-40))<=2
        assert len(opens)==1
        stream.close();stream=None
        time.sleep(.6);count=camera.status().dashboard_frames_encoded
        time.sleep(.6);assert camera.status().dashboard_frames_encoded==count
    finally:
        if stream:stream.close()
        camera.stop()


def test_preview_route_passes_modest_raw_shared_request_and_disable(tmp_path):
    config=load_config(write_test_config(tmp_path));runtime=CollectorRuntime(config)
    seen=[]
    def frames(**kwargs):seen.append(kwargs);yield b'frame'
    runtime.camera.mjpeg_frames=frames
    try:
        client=create_collector_app(runtime).test_client()
        response=client.get('/preview.mjpg');assert response.data==b'frame'
        assert seen==[dict(maximum_fps=2,annotated_only=False)]
        runtime.config=replace(config,collector_preview=CollectorPreviewConfig(enabled=False))
        assert client.get('/preview.mjpg').status_code==404
    finally:runtime.stop()


def test_recent_degraded_and_safe_download_routes(tmp_path,rig):
    recorder=rig.build();recorder.record_manual();captured(rig,recorder)
    rig.clock.now=101;recorder.stop_manual();state=finished(recorder)
    config=load_config(write_test_config(tmp_path));runtime=CollectorRuntime(config,recording=recorder)
    runtime.media=CollectorMedia(recorder.directory)
    try:
        client=create_collector_app(runtime).test_client()
        row=client.get('/api/recordings').json['recordings'][0]
        assert row['status']=='degraded' and row['reason']=='manual'
        assert row['file_role']=='clean_authoritative'
        filename=row['clips'][0]['file'];base=f"/recordings/{state['session_id']}/{filename}"
        assert client.get(base).status_code==200
        assert client.get(base,headers={'Range':'bytes=0-3'}).status_code==206
        assert 'attachment' in client.get(base+'?download=1').headers['Content-Disposition']
        assert b'DEGRADED' in client.get(base.replace('/recordings/','/clips/')).data
        for url in ['/recordings/../../config/default.yaml',base.replace(filename,'session.json'),base.replace(filename,'segment-0000.incomplete.avi'),base.replace(state['session_id'],'bad')]:
            assert client.get(url).status_code==404
        (recorder.directory/state['session_id']/filename).write_bytes(b'changed')
        assert client.get(base).status_code==404
    finally:runtime.stop()


def test_media_rejects_wrong_role_and_symlink(tmp_path):
    session='a'*32;folder=tmp_path/session;folder.mkdir()
    path=folder/'session.json';path.write_text(json.dumps(dict(session_id=session,file_role='annotated_review')))
    media=CollectorMedia(tmp_path)
    with pytest.raises(ValueError):media.session(session)
    path.unlink()
    outside=tmp_path/'outside.json';outside.write_text(json.dumps(dict(session_id=session,file_role='clean_authoritative')))
    try:path.symlink_to(outside)
    except OSError:pytest.skip('host does not permit symlinks')
    with pytest.raises(ValueError):media.session(session)

def test_preview_resize_is_presentation_only(tmp_path):
    raw=np.full((180,320,3),71,np.uint8)
    camera=CameraService(CameraConfig(0,320,180,30,tmp_path),
        capture_factory=lambda _:ContinuousCapture(raw,threading.Event()),
        platform_checker=lambda:True,preview_maximum_width=160)
    camera.start();stream=None
    try:
        wait_until(lambda:camera.status().frames_received>0)
        source=camera.wait_for_frame(-1,copy=False)
        stream=camera.mjpeg_frames(maximum_fps=2,annotated_only=False)
        data=next(stream).split(b'\r\n\r\n',1)[1]
        decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        assert decoded.shape==(90,160,3)
        assert source.frame.shape==(180,320,3) and np.all(source.frame==71)
        assert not source.frame.flags.writeable
    finally:
        if stream:stream.close()
        camera.stop()
