import os,json,signal,socket,subprocess,time,shutil
from pathlib import Path
root=Path('/data/riley-serving-260913-recovery');out=root/'projection-cta-model-lifecycle-v1';out.mkdir();restore=root/'blender-restoration-260914'
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
 binary=json.loads((root/'projection-cta-model-build-v1/test-binary.json').read_text())['executable']
 test=[binary,'projection_ctas_match_serial_model_and_reclaim_pages','--ignored','--nocapture','--test-threads=1']
 env['RILEY_EXPERIMENT_PROJECTION_CTAS']='1'
 commands=[('model',test)]
 for name,argv in commands:
  with (out/(name+'.log')).open('w') as f:p=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
  results.append({'name':name,'exit_code':p.returncode});print(name,p.returncode,flush=True)
  (out/'execution.json').write_text(json.dumps(results,indent=2)+'\n')
  if p.returncode:break
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
assert results and all(r['exit_code']==0 for r in results) and len(results)==1,'GPU gate failed'

(out/'complete.txt').write_text('complete\n')
