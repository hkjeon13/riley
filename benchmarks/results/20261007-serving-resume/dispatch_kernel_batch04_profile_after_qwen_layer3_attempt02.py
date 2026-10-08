import hashlib,json,time,subprocess,shlex
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch04-profile-local-dispatch-attempt02';O.mkdir();failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
items=[(R/'run_kernel_batch04_profile.py','run_kernel_batch04_profile.py'),(R/'kernel-batch04-profile-plan.json','kernel-batch04-profile-plan.json'),(R/'kernel-batch04-independent-replay-attempt01/kernel-batch04-quiet-attempt01-independent-replay.json','kernel-batch04-independent-serving-screen.json')];pins={str(p):sha(p) for p,n in items}
wait=R/'qwen-step109-layer3-localization-collection-attempt02/completion.json'
(O/'preparation.json').write_text(json.dumps({'pins':pins,'wait_for':str(wait),'scope':'full32 diagnostic after actual Batch04 failed full96 and selected3 Qwen paired archive; no serving claim','orders':[['v52','candidate'],['candidate','v52']],'all8':True},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not wait.exists():
  if time.monotonic()>deadline:raise RuntimeError('Qwen actual paired archive incomplete; no profiler launched')
  time.sleep(30)
 assert json.loads(wait.read_bytes())['failure'] is None
 assert all(sha(p)==h for p,h in pins.items())
 probe="import json,subprocess;from pathlib import Path;print(json.dumps({'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'destinations_exist':[n for n in %r if (Path('/data/riley-serving-261007')/n).exists()]}))"%[n for p,n in items]
 state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True));assert not state['compute'] and not state['destinations_exist'],'GPU occupied or create-only destination exists; leave all existing actors/evidence unchanged'
 for p,name in items:subprocess.run(['scp',str(p),'ai-assistant:/data/riley-serving-261007/'+name],check=True)
 cmd='/data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/run_kernel_batch04_profile.py --attempt 1 --plan /data/riley-serving-261007/kernel-batch04-profile-plan.json --verified-baseline /data/riley-serving-261007/kernel-batch04-independent-serving-screen.json'
 with (O/'remote-controller.log').open('x') as log:p=subprocess.run(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant',cmd],stdout=log,stderr=subprocess.STDOUT)
 (O/'SSH-exit.json').write_text(json.dumps({'exit':p.returncode,'actual_terminal_inferred':False},indent=2)+'\n')
 assert p.returncode==0,'SSH observation nonzero; inspect actual profiling process/terminal, do not restart from observation failure'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행; diagnostic only','goal_achieved':False},indent=2)+'\n')
