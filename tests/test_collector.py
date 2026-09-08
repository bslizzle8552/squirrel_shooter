from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from conftest import write_test_config
from squirrel_shooter.camera_service import CameraStatus
from squirrel_shooter.collector_app import CollectorRuntime, create_collector_app
from squirrel_shooter.config import load_config
from squirrel_shooter.detector import DetectorConfig, preprocess
from squirrel_shooter.inference_worker import LatestFrameInferenceWorker
from test_inference_worker import Detector
from test_recording import rig, captured, finished


def forbid_controls(monkeypatch):
    def forbidden(*a,**k): pytest.fail('collector attempted physical-control construction')
    import squirrel_shooter.pan_tilt as pan
    import squirrel_shooter.valve as valve
    import squirrel_shooter.manual_control as manual
    import squirrel_shooter.auto_fire as auto
    import squirrel_shooter.app as app
    import squirrel_shooter.web_dashboard as web
    for module,name in [(pan,'PanTiltController'),(valve,'GPIOValveController'),(manual,'ManualControlService'),
                        (auto,'AutoFireService'),(app,'build_application_runtime'),(web,'create_app')]:
        monkeypatch.setattr(module,name,forbidden)
    # Fail even lower-level optional imports, if a new path attempts them.
    import sys
    for name in ('board','busio','gpiozero','RPi.GPIO','adafruit_pca9685','adafruit_servokit'):
        monkeypatch.setitem(sys.modules,name,SimpleNamespace(__getattr__=forbidden))


@pytest.mark.parametrize('legacy_controls_enabled',[False,True])
def test_collector_constructs_default_services_without_controls(tmp_path,monkeypatch,legacy_controls_enabled):
    forbid_controls(monkeypatch)
    config=load_config(write_test_config(tmp_path))
    # Exercise composition authority independently of operational config validation.
    config=replace(config,manual_control=replace(config.manual_control,servo_enabled=legacy_controls_enabled),
                   auto_fire=replace(config.auto_fire,enabled=legacy_controls_enabled))
    runtime=CollectorRuntime(config)  # Construction never starts a camera.
    try:
        assert set(runtime.__dict__) == {'config','camera','detector','worker','recording','observer','media','_closed'}
        client=create_collector_app(runtime).test_client()
        state=client.get('/api/status').json
        assert state['physical_control_available'] is False
        assert state['role']=='control_incapable_collector'
        assert client.get('/api/detector').json['worker_state']=='disabled'
        for route in ('/api/manual-control','/api/fire','/api/calibration'):
            assert client.post(route).status_code==404
    finally: runtime.stop()


def test_collector_partial_start_unwinds_only_owned_resources(tmp_path,monkeypatch):
    forbid_controls(monkeypatch)
    calls=[]
    camera=SimpleNamespace(start=lambda:calls.append('camera_start'),stop=lambda:calls.append('camera_stop'))
    def fail(): raise RuntimeError('recording failed')
    recorder=SimpleNamespace(start=fail,stop=lambda:calls.append('recording_stop'))
    runtime=CollectorRuntime(load_config(write_test_config(tmp_path)),camera=camera,detector=Detector(),recording=recorder)
    with pytest.raises(RuntimeError,match='recording failed'): runtime.start()
    assert calls==['camera_start','recording_stop','camera_stop']
    runtime.stop()
    assert len(calls)==3
    with pytest.raises(RuntimeError,match='closed'): runtime.start()


@pytest.mark.parametrize('threads', [1, 2])
def test_collector_applies_opencv_budget_before_service_construction(tmp_path, monkeypatch, threads):
    import squirrel_shooter.collector_app as collector
    calls = []
    config = load_config(write_test_config(tmp_path))
    config = replace(config, runtime=replace(config.runtime, opencv_threads=threads))
    monkeypatch.setattr(collector.cv2, 'setNumThreads', lambda value: calls.append(('threads', value)))
    def camera_factory(*args, **kwargs):
        assert calls == [('threads', threads)]
        calls.append(('camera', None))
        return SimpleNamespace(stop=lambda: None)
    monkeypatch.setattr(collector, 'CameraService', camera_factory)
    runtime = CollectorRuntime(config, detector=Detector(), recording=SimpleNamespace(stop=lambda: None))
    runtime.stop()
    assert calls == [('threads', threads), ('camera', None)]


def test_collector_recording_and_detector_share_pristine_source(tmp_path,rig,monkeypatch):
    forbid_controls(monkeypatch)
    service=rig.build()
    config=load_config(write_test_config(tmp_path))
    config=replace(config,detector=DetectorConfig(enabled=True))
    rig.camera.status=lambda:CameraStatus(True,16,12,12,None)
    rig.camera.start=lambda:None
    rig.camera.stop=lambda:None
    detector=Detector()
    runtime=CollectorRuntime(config,camera=rig.camera,detector=detector,recording=service)
    runtime.worker=LatestFrameInferenceWorker(rig.camera,detector,config.detector,clock=rig.clock)
    app=create_collector_app(runtime)
    client=app.test_client()
    assert client.post('/api/recording/start').status_code==403
    headers={'X-Recording-Token':app.config['COLLECTOR_RECORDING_TOKEN']}
    response=client.post('/api/recording/start',headers=headers)
    assert response.status_code==200 and response.json['manual_until']==130
    p=captured(rig,service,value=43)
    p.frame.setflags(write=False)
    def diagnostics():
        tensor,_=preprocess(p.frame)
        tensor[:]=0
        review=p.frame.copy()
        cv2.rectangle(review,(0,0),(8,8),(0,255,0),1)
    detector.hook=diagnostics
    runtime.worker.submit(p); runtime.worker.run_once()
    assert np.all(p.frame==43) and np.all(rig.frames[0]==43)
    status=client.get('/api/status').json
    assert status['detector']['last_source_sequence']==p.sequence
    assert status['recording']['session_id']==response.json['session_id']
    assert not status['automatic_recording_observer_connected']
    assert not status['recording']['automatic_active']
    assert b'Physical controls unavailable' in client.get('/').data
    assert client.post('/api/recording/stop',headers=headers).status_code==200
    finished(service)
    runtime.stop()


def test_intended_recording_extension_uses_source_time_without_inventing_visits(rig):
    service=rig.build()
    observation=100.0
    rig.clock.now=100.5
    result=service.extend_automatic_recording(event_id='fake-observation-group',visit_id=None,
                                             observed_monotonic=observation,reason='offline_contract_test')
    assert result['automatic_reasons'][0]['until']==103.0
    assert result['source_identities'][0]['visit_id'] is None
