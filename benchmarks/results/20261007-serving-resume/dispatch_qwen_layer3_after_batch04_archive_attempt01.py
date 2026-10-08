import json,hashlib,subprocess,time,shlex
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'qwen-step109-layer3-local-dispatch-attempt01';O.mkdir();C=R/'kernel-batch04-terminal-collection-attempt01/completion.json';failure=None
names=['qwen-full128-source-attempt03.tar','qwen-full128-commit-attempt03.pack','qwen-full128-source-receipt-attempt03.json','qwen_step109_layer3_hf_stage_probe_attempt01.py','run_qwen_step109_layer3_hf_stage_attempt01.py','run_qwen_full128_attempt03.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={n:sha(R/n) for n in names}
(O/'preparation.json').write_text(json.dumps({'pins':pins,'wait_for':str(C),'scope':'selected layer3 step108/109 teacher-forced diagnostic; no serving or free-running claim'},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not C.exists():
  if time.monotonic()>deadline:raise RuntimeError('Batch04 archive terminal absent; no source upload or GPU execution started')
  time.sleep(30)
 assert json.loads(C.read_text())['failure'] is None, 'Batch04 collection failed; serving may require evidence recovery; leave all actors unchanged'
 assert all(sha(R/n)==h for n,h in pins.items()),'predeclared observer source changed'
 probe="import json,subprocess;print(json.dumps({'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()}))"
 state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True));assert not state['compute'],'GPU occupied; leave all actors unchanged'
 for name in names:subprocess.run(['scp',str(R/name),'ai-assistant:/data/riley-serving-261007/'+name],check=True)
 for script in ['run_qwen_step109_layer3_hf_stage_attempt01.py','run_qwen_full128_attempt03.py']:
  with (O/(script+'.log')).open('x') as f:p=subprocess.run(['ssh','ai-assistant','/data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/'+script],stdout=f,stderr=subprocess.STDOUT)
  (O/(script+'-SSH-exit.json')).write_text(json.dumps({'exit':p.returncode,'terminal_inferred':False},indent=2)+'\n')
  if script.startswith('run_qwen_step109_layer3_hf'):
   assert p.returncode==0,'HF observer or SSH observation failed; inspect actual controller before any restart'
  # Native exact128 gate remains expected to fail: this is an observer, not a numerical fix.
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','goal_achieved':False},indent=2)+'\n')
