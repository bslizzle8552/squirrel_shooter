"""One bounded real collector validation; explicitly injected outputs, isolated media."""
import argparse,json,os,pathlib,subprocess,sys,time,threading,signal,gc,importlib.abc
from dataclasses import replace
FORBIDDEN=('board','busio','gpiozero','lgpio','RPi','adafruit_pca9685','adafruit_servokit','adafruit_motor')
class BlockPhysical(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==n or fullname.startswith(n+'.') for n in FORBIDDEN):raise RuntimeError('physical import '+fullname)
sys.meta_path.insert(0,BlockPhysical())
from squirrel_shooter.collector_app import CollectorRuntime,create_collector_app
from squirrel_shooter.config import load_config
from squirrel_shooter.detector import DetectionOutput,SquirrelBox
from werkzeug.serving import make_server

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
 def cmd(v):
  r=subprocess.run(v,capture_output=True,text=True);return dict(code=r.returncode,out=r.stdout,err=r.stderr)
 assert cmd(['sudo','-n','fuser','/dev/video0'])['code']==1
 assert 'inactive' in cmd(['systemctl','--user','is-active','squirrel-squirter.service'])['out']
 cfg=load_config(a.config);cfg=replace(cfg,camera=replace(cfg.camera,output_directory=out/'controlled_test_media'))
 r=CollectorRuntime(cfg);app=create_collector_app(r)
 server=make_server('100.110.85.40',5094,app,threaded=True);http=threading.Thread(target=server.serve_forever,daemon=True)
 stage=['outside'];real=r.detector.infer;auto_start=[None];counter=[0]
 def infer(frame):
  result=real(frame)
  if not result.available:return result
  counter[0]+=1
  confidence=.43 if counter[0]%31==0 else .08
  box=(500.,360.,650.,500.) if stage[0]=='inside' else (1144.,310.,1262.,362.)
  return DetectionOutput(True,replace(result.identity,model_id='INJECTED_TEST_NOT_MODEL_OUTPUT'),(SquirrelBox(box,confidence),))
 r.detector.infer=infer
 samples=[];trace=[];originals={};stop=threading.Event();passed=False
 for name in ['_write_packet','_close_segment','_manifest','_storage_check']:
  original=getattr(r.recording,name);originals[name]=original
  def wrapper(*args,_name=name,_original=original,**kw):
   start=time.monotonic()
   try:return _original(*args,**kw)
   finally:
    if len(trace)<5000:trace.append(dict(operation=_name,at=start,seconds=time.monotonic()-start,queue_depth=r.recording._queue.qsize()))
  setattr(r.recording,name,wrapper)
 def sample(label):
  stat=pathlib.Path('/proc/self/stat').read_text().split(') ',1)[1].split()
  temperature=float(pathlib.Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
  flags=int(cmd(['vcgencmd','get_throttled'])['out'].split('=')[1],16)
  s=dict(phase=label,at=time.monotonic(),temperature=temperature,flags=flags,cpu_seconds=(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'),rss=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),host_cpu=pathlib.Path('/proc/stat').read_text().splitlines()[0],status=r.status())
  samples.append(s)
  if temperature>=78 or flags&15:raise RuntimeError('thermal_guard')
 def phase(label,seconds,actions=()):
  print('PHASE',label,flush=True);begin=time.monotonic();pending=list(actions);next_sample=0
  while time.monotonic()-begin<seconds:
   if stop.is_set():raise RuntimeError('stop_requested')
   elapsed=time.monotonic()-begin
   while pending and elapsed>=pending[0][0]:pending.pop(0)[1]()
   if elapsed>=next_sample:sample(label);next_sample=elapsed+2
   time.sleep(.05)
  sample(label)
 def manual(action):
  response=app.test_client().post('/api/recording/'+action,headers={'X-Recording-Token':app.config['COLLECTOR_RECORDING_TOKEN']})
  assert response.status_code==200,response.data
  return response.json
 def stop_manual():
  s=manual('stop');assert s['automatic_active'] and s['active'] and s['manual_remaining_seconds']==0
 def drain():
  for _ in range(15):
   if r.recording.status(compact=True)['pending_sessions']==0:return
   phase('drain',2)
  raise RuntimeError('finalization_timeout')
 signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
 try:
  r.start();http.start();phase('outside_detector',8)
  assert not r.recording.status()['active'] and r.observer.status()['ignored_outside_count']>0
  # One shared presentation consumer, no additional camera.
  preview=r.camera.mjpeg_frames(maximum_fps=2,annotated_only=False)
  (out/'preview-before.jpg').write_bytes(next(preview).split(b'\r\n\r\n',1)[1].rstrip(b'\r\n'))
  phase('outside_preview',6)
  stage[0]='inside'
  phase('automatic_rotation',66,[(5,lambda:manual('start')),(8,stop_manual)])
  state=r.recording.status();assert state['qualifying_observation_count']>100,state
  stage[0]='outside';phase('automatic_tail',11);drain()
  manual('start');phase('manual30',31);drain()
  listing=app.test_client().get('/api/recordings').json
  assert len(listing['recordings'])==2,listing
  assert all(x['clips'] for x in listing['recordings'])
  (out/'listing.json').write_text(json.dumps(listing,indent=2))
  preview.close()
  passed=True
 finally:
  r.stop()
  if http.is_alive():server.shutdown();http.join(3)
  server.server_close()
  devices=[]
  for fd in pathlib.Path('/proc/self/fd').iterdir():
   try:
    target=os.readlink(fd)
    if target.startswith('/dev/'):devices.append(target)
   except OSError:pass
  audit=dict(devices=devices,modules=[m for m in sys.modules if any(m==n or m.startswith(n+'.') for n in FORBIDDEN)],threads=[t.name for t in threading.enumerate()],owners=cmd(['sudo','-n','fuser','-v','/dev/video0']))
  passed=passed and not r.recording.status()['shutdown_timed_out'] and audit['owners']['code']==1
  for name,data in [('samples',samples),('trace',trace),('audit',audit),('result',dict(passed=passed,status=r.status(),pid=os.getpid()))]:
   (out/(name+'.json')).write_text(json.dumps(data,indent=2))
  print('STOPPED',passed,flush=True)
 assert passed
if __name__=='__main__':main()
