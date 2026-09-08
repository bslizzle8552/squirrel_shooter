"""Reproduce field timeline defects without camera/model/hardware."""
import json
import threading
import time
import numpy as np
import pytest
from squirrel_shooter.recording import RecordingConfig, RecordingService
from squirrel_shooter.camera_service import FramePacket
from test_recording import Clock, Camera, Writer, wait_for


def make(tmp_path):
    clock=Clock();frames=[]
    service=RecordingService(Camera(clock),RecordingConfig(enabled=True,target_fps=8,minimum_free_megabytes=0),tmp_path,
        clock=clock,writer_factory=lambda *a:Writer(*a,frames=frames),validator=lambda p,w,h,n:{'decoded_frames':n})
    service._ready=True
    return service,clock,frames


def test_rotation_uses_same_sampling_grid(tmp_path):
    s,c,frames=make(tmp_path)
    s.record_manual();session=s._active;session.prepared=True
    # Repeated extensions keep the logical session alive on the original grid.
    times=[100+i*.5 for i in range(120)]+[160.05,160.13,160.51,161.01]
    for seq,t in enumerate(times):
        c.now=t
        s.extend_automatic_recording(event_id='test',reason='squirrel_detection',observed_monotonic=t,return_status=False)
        assert s._active is session
        packet=FramePacket(seq+1,np.zeros((12,16,3),np.uint8),'wall',t,1)
        s._admit(session,packet)
        data=s._queue.get_nowait();s._write_packet(data);s._queue.task_done();session.queued-=1
    s._finish(session,'test',161.25);s._close_segment('test')
    assert len(session.segments)==2
    assert session.segments[-1]['unique_source_frames']==4
    assert session.queue_dropped==0
    assert sum(x['playback_duration_seconds'] for x in session.segments)==pytest.approx(61.25)


def test_next_session_preroll_never_overtakes_old_queued_tail(tmp_path):
    s,c,frames=make(tmp_path)
    s.record_manual();old=s._active;old.prepared=True
    for seq,t in enumerate([100,100.2]):
        s._admit(old,FramePacket(seq+1,np.full((12,16,3),seq+1,np.uint8),'wall',t,1))
    s._finish(old,'manual_stop',100.3)
    c.now=100.4;s.record_manual();new=s._active;new.prepared=True
    s._admit(new,FramePacket(3,np.full((12,16,3),3,np.uint8),'wall',100.25,1),pre_roll=True)
    s._admit(new,FramePacket(4,np.full((12,16,3),4,np.uint8),'wall',100.5,1))
    s._finish(new,'manual_stop',100.6)
    s._stop.set();s._encode_loop()
    assert old.error is None and new.error is None
    assert len(old.segments)==len(new.segments)==1
    assert old.segments[0]['playback_duration_seconds']<=.375
    assert all(z['finalization_reason']!='new_session' for v in [old,new] for z in v.segments)



def test_long_automatic_overlap_gaps_and_two_rotations(tmp_path):
    s,c,frames=make(tmp_path);s.record_manual();session=s._active;session.prepared=True
    for seq in range(1001):
        t=100+seq/8;c.now=t
        if seq%4==0:
            s.extend_automatic_recording(event_id='test',reason='squirrel_detection',observed_monotonic=t,return_status=False)
        if seq==40:
            assert s.stop_manual()['automatic_active']
        assert s._active is session
        if 35<t-100<37:continue
        packet=FramePacket(seq+1,np.full((12,16,3),seq%255,np.uint8),'wall',t,1)
        s._admit(session,packet)
        data=s._queue.get_nowait();s._write_packet(data);s._queue.task_done();session.queued-=1
    s._finish(session,'test',225.125);s._close_segment('test')
    assert session.error is None and session.queue_dropped==0
    assert len(session.segments)==3
    assert sum(z['playback_duration_seconds'] for z in session.segments)==pytest.approx(125.125)
    assert session.repeated_presentation_frames==15
    rows=[]
    for segment in session.segments:
        rows += [json.loads(x) for x in (tmp_path/session.id/segment['presentation_file']).read_text().splitlines()]
    assert all(b['presentation_monotonic']>a['presentation_monotonic'] for a,b in zip(rows,rows[1:]))
    assert all(b['source_monotonic']>=a['source_monotonic'] for a,b in zip(rows,rows[1:]))
