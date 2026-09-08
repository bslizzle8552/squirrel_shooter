"""Bounded Pi usability/load check; fake adapter positives are explicitly labeled.

Uses one CameraService. Stops every owned component, never leaves test injection
in the deployed collector. Run only under explicit collector validation authority.
"""
import argparse
from dataclasses import replace
import gc
import importlib.abc
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

FORBIDDEN=('board','busio','gpiozero','lgpio','RPi','adafruit_pca9685','adafruit_servokit','adafruit_motor')
class BlockPhysical(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==n or fullname.startswith(n+'.') for n in FORBIDDEN):
            raise RuntimeError('Prohibited physical import: '+fullname)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--host',default='100.110.85.40')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
    sys.meta_path.insert(0,BlockPhysical())
    import cv2
    from squirrel_shooter.collector_app import CollectorRuntime,create_collector_app
    from squirrel_shooter.config import load_config
    from squirrel_shooter.detector import DetectionOutput,SquirrelBox
    from werkzeug.serving import make_server
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    cfg=load_config(args.config)
    assert cfg.detector.enabled and cfg.detector.target_hz==2 and cfg.detector.num_threads==2
    assert not cfg.auto_fire.enabled and not cfg.manual_control.servo_enabled and not cfg.valve.enabled
    assert not cfg.motion.enabled and not cfg.classifier.enabled
    def command(values):
        p=subprocess.run(values,capture_output=True,text=True,timeout=8)
        return dict(code=p.returncode,out=p.stdout,err=p.stderr)
    assert 'inactive' in command(['systemctl','--user','is-active','squirrel-squirter.service'])['out']
    owners=command(['sudo','-n','fuser','-v','/dev/video0'])
    assert owners==dict(code=1,out='',err=''),owners
    # Isolated test evidence directory; nothing is relabeled natural squirrel data.
    cfg=replace(cfg,camera=replace(cfg.camera,output_directory=out/'controlled_test_media'),
                automatic_recording=replace(cfg.automatic_recording,enabled=False))
    runtime=CollectorRuntime(cfg)
    assert runtime.detector.status()['available'],runtime.detector.status()
    app=create_collector_app(runtime)
    server=make_server(args.host,5094,app,threaded=True)
    http=threading.Thread(target=server.serve_forever,daemon=True,name='collector-test-http')
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    samples=[]; observations=[];events=[];last_sequence=-1
    base=f'http://{args.host}:5094'
    def event(kind,**fields):
        item=dict(kind=kind,monotonic=time.monotonic(),**fields);events.append(item)
        print(json.dumps(item),flush=True)
    def audit():
        prohibited={'AutoFireService','ManualControlService','PanTiltController','GPIOValveController','ClassifierWorker','MotionDetector','MotionTracker','SceneSafetyClassifier'}
        instances=[type(o).__name__ for o in gc.get_objects() if type(o).__name__ in prohibited]
        devices=[]
        for fd in Path('/proc/self/fd').iterdir():
            try:
                target=os.readlink(fd)
                if target.startswith('/dev/'):devices.append(target)
            except OSError:pass
        modules=[m for m in sys.modules if any(m==n or m.startswith(n+'.') for n in FORBIDDEN)]
        assert not instances and not modules and not any('gpio' in d or 'i2c' in d for d in devices)
        return dict(instances=instances,modules=modules,devices=devices,threads=[t.name for t in threading.enumerate()],opencv_threads=cv2.getNumThreads())
    def sample(name):
        stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
        flags=int(command(['vcgencmd','get_throttled'])['out'].split('=')[1],16)
        temperature=float(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
        data=dict(phase=name,monotonic=time.monotonic(),temperature_c=temperature,throttle_flags=flags,
            current_throttle_bits=flags&15,process_cpu_seconds=(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'),
            rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),status=runtime.status())
        samples.append(data)
        if temperature>=78 or flags&15:
            stop.set();raise RuntimeError('thermal guard')
    def phase(name,seconds,actions=()):
        nonlocal last_sequence
        event('phase_start',phase=name);begin=time.monotonic();next_sample=begin;pending=list(actions)
        while time.monotonic()-begin<seconds:
            if stop.is_set():raise RuntimeError('measurement stopped')
            now=time.monotonic()
            while pending and now-begin>=pending[0][0]:pending.pop(0)[1]()
            result=runtime.worker.status().get('last_result')
            if result and result['source_sequence']!=last_sequence:
                observations.append(dict(phase=name,result=result));last_sequence=result['source_sequence']
            if now>=next_sample:sample(name);next_sample=now+2
            stop.wait(.05)
        sample(name);event('phase_end',phase=name,audit=audit())
    def post(action):
        req=urllib.request.Request(base+'/api/recording/'+action,data=b'',method='POST',headers={'X-Recording-Token':app.config['COLLECTOR_RECORDING_TOKEN']})
        with urllib.request.urlopen(req,timeout=5) as response:result=json.load(response)
        event('manual_'+action,status=result)
        return result
    def drain():
        deadline=time.monotonic()+25
        while runtime.recording.status(compact=True)['pending_sessions'] and time.monotonic()<deadline:
            phase('finalization',2)
        assert runtime.recording.status(compact=True)['pending_sessions']==0, 'finalization timeout'
    viewer_stop=threading.Event();viewer=None
    def view():
        try:
            with urllib.request.urlopen(base+'/preview.mjpg',timeout=5) as response:
                while not viewer_stop.is_set() and not stop.is_set():
                    if not response.read(32768):break
        except Exception as exc:event('viewer_end',detail=str(exc))
    fake_remaining=[0];real_infer=runtime.detector.infer
    def controlled_infer(frame):
        result=real_infer(frame)
        if fake_remaining[0] and result.available:
            fake_remaining[0]-=1
            return DetectionOutput(True,replace(result.identity,model_id='INJECTED_TEST_NOT_MODEL_OUTPUT'),
                (SquirrelBox((1.,2.,30.,40.),.5 if fake_remaining[0] else .6),))
        return result
    succeeded=False
    try:
        runtime.start();http.start();phase('settle',5)
        assert runtime.camera.status().online
        event('ownership',owners=command(['sudo','-n','fuser','-v','/dev/video0']))
        phase('A_detector_only',20)
        viewer=threading.Thread(target=view,daemon=True,name='test-preview-viewer');viewer.start()
        phase('B_detector_preview',20)
        runtime.observer.config=replace(cfg.automatic_recording,enabled=True)
        runtime.config=replace(cfg,automatic_recording=runtime.observer.config)
        runtime.detector.infer=controlled_infer;fake_remaining[0]=2
        auto_session=[None]
        def check_auto():
            state=runtime.recording.status();assert state['active'] and state['automatic_active'],state
            auto_session[0]=state['session_id']
            assert state['qualifying_observation_count']>=2
            event('automatic_proven',status=state)
        def join_manual():assert post('start')['session_id']==auto_session[0]
        def stop_manual():
            state=post('stop');assert state['active'] and state['automatic_active']
            assert not state['manual_remaining_seconds'] and state['session_id']==auto_session[0]
        phase('C_preview_auto_manual_overlap',14,[(2,check_auto),(4,join_manual),(6,stop_manual)])
        runtime.detector.infer=real_infer
        drain()
        post('start');phase('C_preview_manual30',35)
        drain()
        with urllib.request.urlopen(base+'/api/recordings',timeout=5) as response:listing=json.load(response)
        assert listing['recordings'] and any(row['clips'] for row in listing['recordings']),listing
        for row in listing['recordings']:
            for clip in row['clips']:
                with urllib.request.urlopen(base+'/recordings/'+row['session_id']+'/'+clip['file']+'?download=1',timeout=10) as response:
                    assert response.read(4)==b'RIFF'
        event('recent_clips_proven',listing=listing)
        for route in ['/api/fire','/api/manual-control','/api/calibration','/api/aim']:
            assert app.test_client().post(route).status_code==404
        succeeded=True
    finally:
        viewer_stop.set();stop.set();runtime.stop()
        if http.is_alive():server.shutdown();http.join(3)
        server.server_close()
        if viewer:viewer.join(3)
        succeeded = succeeded and not runtime.recording.status(compact=True)['shutdown_timed_out']
        event('stopped',passed=succeeded,status=runtime.status(),audit=audit(),owners=command(['sudo','-n','fuser','-v','/dev/video0']))
        for name,data in [('samples',samples),('observations',observations),('events',events)]:
            (out/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
        (out/'result.json').write_text(json.dumps(dict(passed=succeeded,pid=os.getpid(),injection='two explicitly marked adapter outputs; no fake camera pixels; test media isolated')),encoding='utf-8')
    assert succeeded

if __name__=='__main__':main()
