from pathlib import Path
import argparse,subprocess,json,os,signal,time,threading,hashlib
from holder import admitted_process
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--timeout',type=int,default=900);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:]if a.command[:1]==['--']else a.command;assert cmd
n=Path(a.out);gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu in ['GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7','GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0'];index=6 if gpu.endswith('55ee1e6a7')else 7;lease=Path(os.environ['RILEY_GPU_HOLDER_LEASE_DIR']);holder=int(os.environ['RILEY_GPU_HOLDER_PID']);ticks=os.environ['RILEY_GPU_HOLDER_START_TICKS'];wanted=Path(cmd[0]).resolve();output=os.environ['RILEY_DIVERGENCE_NATIVE_OUTPUT'];verified={}
def tick(pid):return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
def rows():
 raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 return [dict(pid=int(v.split(',')[0]),memory_mib=float(v.split(',')[1]))for v in raw.splitlines()if v.strip()]
def support(row):
 pid=row['pid']
 if pid==holder:
  s=json.loads((lease/f'gpu{index}.status.json').read_text());return tick(pid)==ticks==s['start_ticks']and s['phase']=='active'and time.time_ns()-s['updated_ns']<10**10
 return admitted_process(lease,index,pid,row['memory_mib'])
def own(row):
 pid=row['pid'];proc=Path('/proc')/str(pid)
 try:
  t=tick(pid)
  if pid in verified:return verified[pid]==t
  if proc.stat().st_uid!=os.getuid()or(proc/'exe').resolve()!=wanted:return False
  if (b'RILEY_DIVERGENCE_NATIVE_OUTPUT='+output.encode())not in(proc/'environ').read_bytes().split(b'\0'):return False
  verified[pid]=t;return True
 except FileNotFoundError:return pid in verified and row["memory_mib"]==0
# NVML may report a CUDA worker TID, and may retain its row briefly after
# /proc disappears. Bind the direct executable and its thread group while live;
# never classify an unverified vanished PID as ours or signal another process.
def bind_live_thread(pid,root_pid):
 proc=Path('/proc')/str(pid)
 try:
  if proc.stat().st_uid!=os.getuid()or(proc/'exe').resolve()!=wanted:return False
  fields=dict(line.split(':',1)for line in(proc/'status').read_text().splitlines()if ':'in line)
  if int(fields['Tgid'])!=root_pid:return False
  if (b'RILEY_DIVERGENCE_NATIVE_OUTPUT='+output.encode())not in(proc/'environ').read_bytes().split(b'\0'):return False
  verified[pid]=tick(pid);return True
 except FileNotFoundError:return False

def exited_verified(row):
 return row['pid']in verified and not Path('/proc',str(row['pid'])).exists()

initial=rows();assert all(support(x)for x in initial),('occupied; preserved',initial)
state=dict(status='running',pid=os.getpid(),start_ticks=tick(os.getpid()),argv=cmd,gpu=gpu,initial=initial,started_ns=time.time_ns(),observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());receipt=n.with_suffix('.json');assert not receipt.exists();receipt.write_text(json.dumps(state,indent=2));start=time.monotonic();failure=None
with n.with_suffix('.stdout').open('x')as stdout,n.with_suffix('.stderr').open('x')as stderr,n.with_suffix('.samples.jsonl').open('x')as samples:
 child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True);child_tick=tick(child.pid)
 initial_child_bound=bind_live_thread(child.pid,child.pid)
 stop_tracking=threading.Event()
 def track_threads():
  while not stop_tracking.is_set()and child.poll()is None:
   try:
    if tick(child.pid)!=child_tick:break
    for task in Path('/proc',str(child.pid),'task').iterdir():bind_live_thread(int(task.name),child.pid)
   except FileNotFoundError:pass
   stop_tracking.wait(.005)
 tracker=threading.Thread(target=track_threads,daemon=True)
 state.update(child_pid=child.pid,child_start_ticks=child_tick,expected_executable=str(wanted),early_verified_owned_start_ticks=dict(verified));receipt.write_text(json.dumps(state,indent=2))
 try:
  binding_attempts=0;binding_deadline=time.monotonic()+2
  while not initial_child_bound and child.poll()is None and time.monotonic()<binding_deadline:
   time.sleep(.005);binding_attempts+=1;initial_child_bound=bind_live_thread(child.pid,child.pid)
  state['initial_binding_retries']=binding_attempts
  if not initial_child_bound:
   proc=Path('/proc')/str(child.pid);details=dict(pid=child.pid,wanted=str(wanted),expected_uid=os.getuid())
   try:
    details.update(uid=proc.stat().st_uid,exe=str((proc/'exe').resolve()),status_tgid=next(x for x in(proc/'status').read_text().splitlines()if x.startswith('Tgid:')),environment_marker=(b'RILEY_DIVERGENCE_NATIVE_OUTPUT='+output.encode())in(proc/'environ').read_bytes().split(b'\0'))
   except FileNotFoundError:details['exited']=True
   raise AssertionError(('direct child executable/UID/Tgid/environment binding failed',details))
  tracker.start()
  while child.poll()is None:
   current=rows();device_mib=float(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert device_mib<=22528,'22GiB whole-device cap';samples.write(json.dumps(dict(time_ns=time.time_ns(),device_memory_mib=device_mib,processes=current,verified_owned_start_ticks=dict(verified)))+'\n');samples.flush();unknown=[x for x in current if not own(x)and not support(x)and not exited_verified(x)];assert not unknown,('foreign preserved',unknown);assert sum(x['memory_mib']for x in current if own(x)or x['pid']==holder)<=22528,'22GiB own+holder budget';assert time.monotonic()-start<a.timeout,'timeout';time.sleep(.5)
 except BaseException as e:failure=repr(e)
 finally:
  if child.poll()is None:
   assert tick(child.pid)==child_tick;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:assert tick(child.pid)==child_tick;os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
  rc=child.wait();deadline=time.monotonic()+30
  while any(own(x)or exited_verified(x)for x in rows()):
   if time.monotonic()>deadline:failure=failure or'own GPU teardown timeout';break
   time.sleep(.25)
stop_tracking.set()
if tracker.ident is not None:tracker.join(timeout=2)
assert not tracker.is_alive()
state.update(status='complete'if rc==0 and failure is None else'failed',returncode=rc,failure=failure,verified_owned_start_ticks=verified,finished_ns=time.time_ns());receipt.write_text(json.dumps(state,indent=2));print(json.dumps(state));raise SystemExit(0 if state['status']=='complete'else 1)
