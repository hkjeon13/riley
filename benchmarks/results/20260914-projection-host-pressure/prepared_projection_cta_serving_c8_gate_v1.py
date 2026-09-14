import os,json,signal,socket,subprocess,time,shutil
from pathlib import Path
root=Path('/data/riley-serving-260913-recovery');out=root/'projection-cta-serving-c8-lifecycle-v1';out.mkdir();restore=root/'blender-restoration-260914'
rows=json.loads((restore/'launch.json').read_text());stopped=[];fds=[];results=[]
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def ready(port):
 try:
  with socket.create_connection(('127.0.0.1',port),timeout=2) as s:
   s.settimeout(5);s.sendall(b'{"type":"get_scene_info","params":{}}');data=b''
   while True:
    chunk=s.recv(65536)
    if not chunk:return False
    data+=chunk
    try:return json.loads(data).get('status')=='success'
    except json.JSONDecodeError:pass
 except OSError:return False
try:
 for row in rows:
  pid=row['pid'];fd=os.pidfd_open(pid);fds.append(fd)
  argv=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
  assert argv==row['argv'] and ready(row['port']), 'Blender identity/readiness differs'
 for row,fd in zip(rows,fds):signal.pidfd_send_signal(fd,signal.SIGTERM);stopped.append(row)
 deadline=time.monotonic()+30
 while any(live(r['pid']) for r in stopped):
  assert time.monotonic()<deadline,'Blender stop timeout';time.sleep(.2)
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root/'toolchain130/nvidia/cu13/lib');env['RILEY_REAL_CHECKPOINT']='/data/riley-benchmark/20260827T051948Z-d7ad713a/model'
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
 assert not compute.strip(),'Competing compute processes'
 (out/'memory-preflight.json').write_text(json.dumps({'meminfo':Path('/proc/meminfo').read_text(),'timestamp_ns':time.time_ns()},indent=2)+'\n')
 argv=['python3',str(root/'source/benchmarks/analysis/projection_cta_serving_screen.py'),str(root),'/dev/shm/riley-projection-cta-serving-c8-v1','--concurrency','8','--warmup','64','--retained','2048','--host-quiet-timeout-seconds','120','--cooldown-timeout-seconds','600']
 with (out/'serving.log').open('w') as log:
  p=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
 results.append({'name':'serving','exit_code':p.returncode})
 (out/'execution.json').write_text(json.dumps(results,indent=2)+'\n');print(results,flush=True)
 assert p.returncode==0,'serving gate failed'

finally:
 for fd in fds:os.close(fd)
 env=os.environ.copy();env.update(DISPLAY=':99',XAUTHORITY=str(restore/'Xauthority'),BLENDER_MCP_DISABLE_TELEMETRY='1');env.pop('WAYLAND_DISPLAY',None)
 for row in stopped:
  if live(row['pid']):continue
  with (restore/f"{row['port']}.log").open('ab') as f:p=subprocess.Popen(row['argv'],cwd=row['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
  row['pid']=p.pid
  (restore/'launch.json').write_text(json.dumps(rows,indent=2)+'\n')
 deadline=time.monotonic()+60
 pending=list(rows)
 while pending and time.monotonic()<deadline:
  pending=[r for r in pending if not ready(r['port'])]
  if pending:time.sleep(1)
 receipt={'restored':not pending,'processes':[{'pid':r['pid'],'port':r['port']} for r in rows]}
 (out/'blender-restored.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
 assert not pending,'Restoration not ready'
assert len(results)==1,'incomplete evidence collection'

(out/'complete.txt').write_text('complete\n')
