"""Resume identical matrix only after classified premeasurement IO gate failure."""
import pathlib,json,time,hashlib,subprocess,shlex
R=pathlib.Path(__file__).resolve().parent;O=R/'kernel-batch06-quiet-resume-queue-attempt02';O.mkdir();failure=None
C=R/'kernel-batch06-terminal-collection-attempt01/completion.json';V=R/'kernel-batch06-independent-replay-attempt01/completion.json';CODE=R/'kernel-batch06-controller';pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.iterdir() if p.is_file()}
(O/'preparation.json').write_text(json.dumps({'controller_input_sha256':pins,'wait_for':[str(C),str(V)],'retry_only':'attempt01 exact c1-fixed first admission timeout; zero measured lanes','orders':[['v52','candidate','vllm'],['vllm','candidate','v52']]*2,'admission_policy':'unchanged CPU_some<=5,IO_full<=5,memory_full<=0.5;3 consecutive2s; zero GPU','foreign_process_mutation':False,'serving_performance':'미실행'},indent=2)+'\n')
def remote(code):return subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(code)],text=True)
try:
 deadline=time.monotonic()+21600
 while not C.exists() or not V.exists():
  assert time.monotonic()<deadline,'prior evidence still incomplete; do not restart live work'
  time.sleep(20)
 collected=json.loads(C.read_bytes());verified=json.loads(V.read_bytes());assert collected['failure'] is None and verified['failure'] is None
 prior=collected['records'][0]['completion'];assert prior['records']==[] and prior['failure']=={'type':'RuntimeError','message':'whole attempt aborted: host start gate timeout c1-fixed-r0-v52-start'},'not the classified zero-lane failure; no retry'
 assert json.loads((R/'kernel-batch06-independent-replay-attempt01/raw-replay.json').read_bytes())['lanes']==[]
 policy=json.loads((CODE/'batch-plan.json').read_bytes())['host_start_gate'];streak=0
 script="import json,subprocess,time;from pathlib import Path;print(json.dumps({'time_ns':time.time_ns(),'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']},'GPU':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'attempt02_exists':Path('/data/riley-serving-261007/kernel-batch06-quiet-attempt02').exists()}))"
 while True:
  state=json.loads(remote(script));assert not state['attempt02_exists'],'existing attempt02; inspect original, never restart'
  def avg(resource,kind):return float(next(row for row in state['pressure'][resource].splitlines() if row.startswith(kind+' ')).split()[1].split('=')[1])
  quiet=not state['GPU'] and avg('cpu','some')<=policy['cpu_some_max'] and avg('io','full')<=policy['io_full_max'] and avg('memory','full')<=policy['memory_full_max'];streak=streak+1 if quiet else 0
  state.update(quiet=quiet,streak=streak,required_streak=3)
  (O/'wait-state.json').write_text(json.dumps(state,indent=2)+'\n')
  with (O/'all-host-observations.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
  if streak>=3:break
  assert time.monotonic()<deadline,'host still above frozen admission limits; no measured HTTP work or actor mutation'
  time.sleep(2 if quiet else 30)
 assert all(hashlib.sha256((CODE/n).read_bytes()).hexdigest()==h for n,h in pins.items())
 check="import json,hashlib;from pathlib import Path;r=Path('/data/riley-serving-261007/kernel-batch06-controller');print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.iterdir() if p.is_file()}))";assert json.loads(remote(check))==pins
 (O/'launch-state.json').write_text(json.dumps(state,indent=2)+'\n')
 with (O/'remote-controller.log').open('x') as f:p=subprocess.run(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant','/data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/kernel-batch06-controller/run.py --attempt 2 --mode quiet'],stdout=f,stderr=subprocess.STDOUT)
 (O/'SSH-exit.json').write_text(json.dumps({'exit':p.returncode,'actual_terminal_inferred':False})+'\n');assert p.returncode==0,'inspect actual terminal; observation error never restarts a live job'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'adopted':False,'goal_achieved':False},indent=2)+'\n')
