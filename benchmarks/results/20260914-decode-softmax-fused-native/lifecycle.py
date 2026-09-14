import os,json,signal,socket,subprocess,time,shutil
from pathlib import Path
root=Path('/data/riley-serving-260913-recovery');out=root/'decode-softmax-fused-native-v1';out.mkdir();restore=root/'blender-restoration-260914'
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
 jobs=[]
 for width in (2,4,8):
  jobs.append(('memcheck-w'+str(width),['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(root/'decode-softmax-fused-probe-v1'),'--small',str(width)]))
  jobs.append(('native-w'+str(width),[str(root/'decode-softmax-fused-probe-v1'),'--full',str(width)]))

 for name,argv in jobs:
  deadline=time.monotonic()+600
  while True:
   temperature=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
   if temperature<=48:break
   assert time.monotonic()<deadline,'cooldown timeout'
   time.sleep(1)
  with (out/(name+'.log')).open('w') as log:
   p=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
  results.append({'name':name,'exit_code':p.returncode})
  (out/'execution.json').write_text(json.dumps(results,indent=2)+'\n');print(results,flush=True)
  assert p.returncode==0,name+' gate failed'

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
assert len(results)==6,'incomplete evidence collection'

(out/'complete.txt').write_text('complete\n')
