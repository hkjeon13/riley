"""Run predeclared experimental Batch03 profiling after actual Qwen terminal."""
import hashlib,json,shlex,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch03-profile-local-dispatch-attempt01';O.mkdir()
REMOTE='/data/riley-serving-261007';Q=R/'qwen-m2-independent-collection-attempt04/completion.json'
FILES={R/'run_kernel_batch03_profile.py':'run_kernel_batch03_profile.py',R/'kernel-batch03-profile-plan.json':'kernel-batch03-profile-plan.json',R/'batch02-batch03-independent-replay-attempt02/kernel-batch03-quiet-attempt01-independent-replay.json':'kernel-batch03-quiet-attempt01-independent-replay.json'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
pins={str(p):sha(p) for p in FILES};failure=None;steps=[]
probe="""import json,subprocess
from pathlib import Path
root=Path('/data/riley-serving-261007');rows=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
live=[line for line in rows if 'run_qwen_m2_bounded_attempt03.py' in line and 'python' in line and 'python3 -c' not in line]
print(json.dumps({'native_controller_live':live,'native_terminal':(root/'qwen-m2-bounded-validation-attempt03/completion.json').exists(),'GPU_compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()}))
"""
write('preparation.json',{'pins':pins,'wait_for':str(Q),'scope':'32 diagnostic profiles, no serving timing qualification; actual native terminal and independent collection terminal required; no restart on observation failure'})
try:
 deadline=time.monotonic()+14400
 while not Q.exists():
  write('wait-state.json',{'time_ns':time.time_ns(),'Qwen_independent_collection_terminal':False,'profile_executed':False})
  if time.monotonic()>deadline:raise RuntimeError('Qwen independent collection still incomplete at deadline; no profile started')
  time.sleep(30)
 receipt=json.loads(Q.read_text());write('Qwen-independent-collection-snapshot.json',receipt);assert receipt['failure'] is None,'Qwen evidence collection failed; inspect actual failure before profiling'
 while True:
  try:state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True))
  except (subprocess.SubprocessError,OSError) as error:
   with (O/'observation-errors.jsonl').open('a') as f:f.write(json.dumps({'time_ns':time.time_ns(),'error':str(error),'terminal_inferred':False})+'\n')
   if time.monotonic()>deadline:raise RuntimeError('native terminal state unavailable; no restart or profile started')
   time.sleep(30);continue
  write('remote-wait-state.json',state)
  if not state['native_controller_live'] and state['native_terminal'] and not state['GPU_compute']:break
  if time.monotonic()>deadline:raise RuntimeError('actual native/controller/GPU still occupied at deadline; left untouched')
  time.sleep(30)
 assert all(sha(Path(p))==digest for p,digest in pins.items()),'predeclared local profile inputs changed'
 for file,name in FILES.items():
  subprocess.run(['scp',str(file),'ai-assistant:'+REMOTE+'/'+name],check=True)
  steps.append({'uploaded':name,'sha256':pins[str(file)]});write('progress.json',steps)
 argv=[REMOTE+'/vllm0271-venv/bin/python',REMOTE+'/run_kernel_batch03_profile.py','--attempt','1','--plan',REMOTE+'/kernel-batch03-profile-plan.json','--verified-baseline',REMOTE+'/kernel-batch03-quiet-attempt01-independent-replay.json']
 write('launch.json',{'remote_argv':argv,'scope':'diagnostic-only; no adoption'})
 with (O/'remote-controller.log').open('x') as log:code=subprocess.run(['ssh','ai-assistant',shlex.join(argv)],stdout=log,stderr=subprocess.STDOUT).returncode
 steps.append({'SSH_observation_exit':code});write('progress.json',steps)
 assert code==0,'profile SSH observation nonzero; recheck actual process/terminal before any restart'
except BaseException as error:failure={'type':type(error).__name__,'message':str(error)};raise
finally:write('completion.json',{'failure':failure,'steps':steps,'serving_performance':'미실행; profiling is diagnostic only','goal_achieved':False})
