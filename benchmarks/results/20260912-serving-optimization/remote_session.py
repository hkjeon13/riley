import argparse, json, os, signal, socket, subprocess, time
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912')
PIDS=[2857155,2857156,2857157]
PORTS=[9876,9911,9887]
SNAPSHOT=ROOT/'blender-session.json'
def identity(pid):
 p=Path('/proc')/str(pid)
 return {'pid':pid,'start':(p/'stat').read_text().split(') ',1)[1].split()[19], 'argv':[(x.decode()) for x in (p/'cmdline').read_bytes().split(b'\0') if x], 'cwd':str((p/'cwd').resolve()),'env':{k.decode():v.decode() for raw in (p/'environ').read_bytes().split(b'\0') if b'=' in raw for k,v in [raw.split(b'=',1)] if k in {b'DISPLAY',b'XAUTHORITY',b'XDG_RUNTIME_DIR',b'DBUS_SESSION_BUS_ADDRESS',b'WAYLAND_DISPLAY'}}}
def live(pid):
 try: return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def stop():
 rows=[identity(pid) for pid in PIDS]
 for row in rows:
  if 'blender' not in Path(row['argv'][0]).name:raise RuntimeError('PID changed')
 with SNAPSHOT.open('x') as f:json.dump(rows,f,indent=2)
 os.chmod(SNAPSHOT,0o600)
 for row in rows:
  if identity(row['pid'])!=row:raise RuntimeError('process identity changed')
  os.kill(row['pid'],signal.SIGTERM)
 deadline=time.monotonic()+30
 while any(live(pid) for pid in PIDS):
  if time.monotonic()>deadline:raise RuntimeError('Blender did not exit; no SIGKILL sent')
  time.sleep(.2)
 (ROOT/'blender-restore-deadline').write_text(str(time.time()+1800))
 subprocess.Popen(['python3',str(Path(__file__).resolve()),'watchdog'],stdin=subprocess.DEVNULL,stdout=(ROOT/'watchdog.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps({'stopped':PIDS,'automatic_restore_after_seconds':1800}))
def restore():
 if (ROOT/'blender-restored.json').exists():return
 rows=json.loads(SNAPSHOT.read_text());restored=[]
 for row,port in zip(rows,PORTS):
  if live(row['pid']):
   if identity(row['pid'])['start']==row['start']:restored.append({'original_pid':row['pid'],'new_pid':row['pid'],'already_alive':True});continue
   raise RuntimeError('old PID reused; review needed')
  with socket.socket() as s:
   if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('restore port already occupied')
  env=os.environ.copy();env.update(row['env'])
  log=ROOT/f"blender-resume-{row['pid']}.log"
  with log.open('a') as f:proc=subprocess.Popen(row['argv'],cwd=row['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
  restored.append({'original_pid':row['pid'],'new_pid':proc.pid,'port':port,'log':str(log)})
 (ROOT/'blender-restored.json').write_text(json.dumps(restored,indent=2))
 deadline=time.monotonic()+60
 for row in restored:
  port=row.get('port')
  if port is None:continue
  while True:
   if not live(row['new_pid']):raise RuntimeError('restored Blender exited')
   with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',port))==0:break
   if time.monotonic()>deadline:raise RuntimeError('restored Blender port timeout')
   time.sleep(.5)
 (ROOT/'blender-restore-verified.json').write_text(json.dumps({'alive_and_listening':True,'processes':restored},indent=2))
 print(json.dumps({'restored':restored,'verified':True}))
def watchdog():
 while not (ROOT/'blender-restored.json').exists():
  if time.time()>float((ROOT/'blender-restore-deadline').read_text()):restore();return
  time.sleep(5)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['stop','restore','watchdog','extend']);a=p.parse_args()
 if a.action=='extend':(ROOT/'blender-restore-deadline').write_text(str(time.time()+1800))
 else:globals()[a.action]()
