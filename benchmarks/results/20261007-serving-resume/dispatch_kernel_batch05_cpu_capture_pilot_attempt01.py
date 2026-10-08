import hashlib,json,time,subprocess,shlex
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch05-cpu-capture-pilot-local-dispatch-attempt01';O.mkdir();failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
items=[(R/'run_kernel_batch05_cpu_capture_pilot_attempt01.py','run_kernel_batch05_cpu_capture_pilot_attempt01.py'),(R/'kernel-batch05-cpu-capture-pilot-plan-attempt01.json','kernel-batch05-cpu-capture-pilot-plan-attempt01.json'),(R/'kernel-batch05-independent-replay-attempt02/kernel-batch05-quiet-attempt01-independent-replay.json','kernel-batch05-cpu-capture-pilot-independent-serving-screen.json')];pins={str(p):sha(p) for p,n in items}
wait=R/'qwen-free-running-dispatch-attempt02/completion.json'
(O/'preparation.json').write_text(json.dumps({'pins':pins,'wait_for':str(wait),'scope':'pilot8 CPU process-tree root with private profiler dirs after validated full-lifetime GPU pilot; no serving scores or host setting mutations','orders':[['v52','candidate'],['candidate','v52']],'selected_cells':['c1-fixed','c32-natural'],'expected_lanes':8},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not wait.exists():
  if time.monotonic()>deadline:raise RuntimeError('Qwen actual paired archive incomplete; no profiler launched')
  time.sleep(30)
 assert json.loads(wait.read_bytes())['failure'] is None
 qwen=json.loads((R/'qwen-free-running-dispatch-attempt02/independent-proof.json').read_bytes());assert qwen['passed'] and qwen['actual_exact_logit_rows']==128 and qwen['actual_exact_selected_tokens']==128
 assert all(sha(p)==h for p,h in pins.items())
 pilot=json.loads((R/'kernel-batch05-capture-pilot-independent-collection-attempt01/independent8-profile-verification.json').read_bytes());assert pilot['complete_profiles']==8 and pilot['HTTP_requests_replayed']==1536
 assert json.loads((R/'kernel-batch05-capture-pilot-phase-analysis-attempt01/completion.json').read_bytes())['failure'] is None
 probe="import json,subprocess;from pathlib import Path;print(json.dumps({'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'destinations_exist':[n for n in %r if (Path('/data/riley-serving-261007')/n).exists()]}))"%[n for p,n in items]
 state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True));assert not state['compute'] and not state['destinations_exist'],'GPU occupied or create-only destination exists; leave all existing actors/evidence unchanged'
 for p,name in items:subprocess.run(['scp',str(p),'ai-assistant:/data/riley-serving-261007/'+name],check=True)
 cmd='sudo -n /data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/run_kernel_batch05_cpu_capture_pilot_attempt01.py --attempt 1 --plan /data/riley-serving-261007/kernel-batch05-cpu-capture-pilot-plan-attempt01.json --verified-baseline /data/riley-serving-261007/kernel-batch05-cpu-capture-pilot-independent-serving-screen.json'
 with (O/'remote-controller.log').open('x') as log:p=subprocess.run(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant',cmd],stdout=log,stderr=subprocess.STDOUT)
 (O/'SSH-exit.json').write_text(json.dumps({'exit':p.returncode,'actual_terminal_inferred':False},indent=2)+'\n')
 assert p.returncode==0,'SSH observation nonzero; inspect actual profiling process/terminal, do not restart from observation failure'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행; diagnostic only','goal_achieved':False},indent=2)+'\n')
