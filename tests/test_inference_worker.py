import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from squirrel_shooter.camera_service import FramePacket
from squirrel_shooter.detector import DetectionOutput, DetectorConfig, SquirrelBox, preprocess
from squirrel_shooter.inference_worker import LatestFrameInferenceWorker
from squirrel_shooter.model_manifest import ModelIdentity
from test_recording import Camera, Clock, wait_for


IDENTITY=ModelIdentity('test','v3','a'*64,'b'*64,'c'*64,'d'*64)


class Detector:
    def __init__(self, hook=lambda:None):
        self.hook=hook
        self.frames=[]
    def status(self): return dict(enabled=True,configured=True,available=True,backend='fake',model_id='test',model_version='v3')
    def infer(self,frame):
        self.frames.append(frame)
        self.hook()
        return DetectionOutput(True,IDENTITY,(SquirrelBox((1,2,3,4),.02),))


def packet(seq, *, at=100, gen=1, shape=(12,16,3)):
    frame=np.full(shape,seq%256,np.uint8)
    frame.setflags(write=False)
    return FramePacket(seq,frame,'2026-09-07T12:00:00Z',at,gen)


def worker(hook=lambda:None,**config):
    clock=Clock()
    detector=Detector(hook)
    service=LatestFrameInferenceWorker(None,detector,DetectorConfig(enabled=True,**config),clock=clock)
    return clock,detector,service


def test_pending_replaces_without_fifo_or_duplicate_inference():
    clock,detector,w=worker()
    for n in range(1,101): w.submit(packet(n))
    s=w.status()
    assert s['pending_sequence']==100 and s['superseded_pending_count']==99
    assert w.run_once()
    assert len(detector.frames)==1 and detector.frames[0][0,0,0]==100
    assert not w.submit(packet(100))
    clock.now+=1
    assert not w.run_once()
    s=w.status()
    assert s['last_result']['source_sequence']==100
    assert s['last_result']['camera_generation']==1
    assert s['last_result']['output']['identity']['checkpoint_sha256']=='a'*64
    assert s['last_result']['native_width']==16 and s['last_result']['native_height']==12
    assert s['last_result']['source_monotonic']==100
    assert s['last_result']['event_id'] is None and s['last_result']['visit_id'] is None


def test_one_inflight_latest_pending_and_slow_inference_has_no_catchup():
    entered,release=threading.Event(),threading.Event()
    def hook():
        entered.set()
        assert release.wait(3)
    clock,detector,w=worker(hook)
    w.submit(packet(1))
    thread=threading.Thread(target=w.run_once)
    thread.start()
    try:
        assert entered.wait(2)
        clock.now=101
        for n in range(2,50): w.submit(packet(n,at=101))
        assert not w.run_once()
        assert w.status()['in_flight'] and w.status()['pending_sequence']==49
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert not w.run_once()
        clock.now=101.49
        assert not w.run_once()
        clock.now=101.5
        w.submit(packet(50,at=101.5))
        assert w.run_once()
        assert len(detector.frames)==2 and detector.frames[-1][0,0,0]==50
        assert not w.run_once()
    finally:
        release.set()
        thread.join(2)


def test_fast_inference_keeps_nominal_cadence_and_superseded_counters():
    clock,detector,w=worker()
    w.submit(packet(1)); w.run_once()
    clock.now=100.1
    w.submit(packet(2,at=100.1))
    clock.now=100.2
    w.submit(packet(3,at=100.2))
    assert not w.run_once()
    clock.now=100.5
    assert w.run_once()
    assert len(detector.frames)==2 and detector.frames[-1][0,0,0]==3
    s=w.status()
    assert s['skipped_cadence_count']==2 and s['superseded_pending_count']==1
    assert s['last_capture_to_result_age_seconds']==pytest.approx(.3)
    assert s['last_detection_count']==1 and s['last_top_squirrel_confidence']==.02


@pytest.mark.parametrize('at',[98,101,float('nan'),float('inf')])
def test_stale_or_invalid_source_time_rejected(at):
    clock,detector,w=worker()
    assert not w.submit(packet(1,at=at))
    assert not w.run_once() and not detector.frames
    assert w.status()['stale_before_inference_count']==1


def test_pending_can_expire_before_claim():
    clock,detector,w=worker()
    w.submit(packet(1))
    clock.now=101
    assert not w.run_once()
    assert not w.status()['pending_frame']
    assert w.status()['last_drop']['dropped_reason']=='stale_before_inference'


def test_stale_completion_and_later_expiry_hide_detections():
    clock,detector,w=worker()
    detector.hook=lambda:setattr(clock,'now',102)
    w.submit(packet(1)); w.run_once()
    s=w.status()
    assert not s['result_available'] and s['last_detection_count']==0
    assert s['stale_after_inference_count']==1
    clock.now=103
    detector.hook=lambda:None
    w.submit(packet(2,at=103)); w.run_once()
    assert w.status()['result_available']
    clock.now=105
    assert not w.status()['result_available'] and w.status()['last_detection_count']==0


@pytest.mark.parametrize('change',[{'gen':2},{'shape':(24,32,3)}])
def test_source_change_invalidates_pending_and_completion(change):
    clock,detector,w=worker()
    w.submit(packet(1))
    detector.hook=lambda:w.submit(packet(2,**change))
    assert w.run_once()
    s=w.status()
    assert s['last_result'] is None and not s['result_available']
    assert s['pending_sequence']==2
    assert s['last_drop']['dropped_reason']=='source_changed_during_inference'
    detector.hook=lambda:None
    clock.now=100.5
    assert w.run_once()
    s=w.status()
    assert s['last_result']['native_width']==change.get('shape',(12,16,3))[1]
    assert s['last_result']['camera_generation']==change.get('gen',1)


def test_previous_generation_cannot_replace_new_pending():
    clock,detector,w=worker()
    w.submit(packet(1,gen=2))
    assert not w.submit(packet(2,gen=1))
    assert w.status()['pending_sequence']==1


def test_adapter_exception_does_not_kill_worker():
    def bad(): raise RuntimeError('backend failed')
    clock,detector,w=worker(bad)
    w.submit(packet(1)); assert w.run_once()
    assert w.status()['worker_state']=='unavailable'
    assert w.status()['error_unavailable_count']==1
    detector.hook=lambda:None
    clock.now=100.5
    w.submit(packet(2,at=100.5)); assert w.run_once()
    assert w.status()['result_available']


def test_bounded_shutdown_with_blocked_native_call_and_no_restart():
    entered,release=threading.Event(),threading.Event()
    clock=Clock()
    camera=Camera(clock)
    def block():
        entered.set()
        release.wait(3)
    w=LatestFrameInferenceWorker(camera,Detector(block),DetectorConfig(enabled=True,shutdown_timeout_seconds=.05),clock=clock)
    w.start()
    camera.push()
    try:
        assert entered.wait(2)
        started=time.monotonic()
        w.stop()
        assert time.monotonic()-started < .5
        assert w.status()['shutdown_timed_out']
        assert not w.status()['pending_frame'] and w.status()['last_result'] is None
        assert not w.submit(packet(9))
        w.start()
        assert len(w._threads)==2
    finally:
        release.set()
        for thread in w._threads: thread.join(2)
    assert w.status()['last_result'] is None


def test_real_camera_consumer_thread_is_independent_of_motion_events():
    clock=Clock()
    camera=Camera(clock)
    detector=Detector()
    w=LatestFrameInferenceWorker(camera,detector,DetectorConfig(enabled=True),clock=clock)
    w.start()
    try:
        first=camera.push(value=17)
        wait_for(lambda:w.status()['inference_count']==1)
        clock.now+=.5
        second=camera.push(value=17)  # Stationary pixels, new source sequence.
        wait_for(lambda:w.status()['inference_count']==2)
        assert np.array_equal(first.frame,second.frame)
        assert w.status()['last_source_sequence']==second.sequence
    finally: w.stop()


def test_telemetry_bounded_over_many_frames():
    clock,detector,w=worker()
    for n in range(200):
        clock.now=100+n
        w.submit(packet(n,at=clock.now)); w.run_once()
    assert len(w.status()['rolling_latency_seconds'])==128
