import json,hashlib,subprocess,time,shlex
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch04-matched-local-dispatch-attempt01';O.mkdir();C=R/'qwen-step109-localization-collection-attempt01/completion.json';code=R/'kernel-batch04-controller';failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={p.name:sha(p) for p in code.iterdir() if p.is_file()};(O/'preparation.json').write_text(json.dumps({'pins':pins,'wait_for':str(C),'scope':'all8 quiet matched96; no exclusion; original vLLM0.27.1 benchmark distinct from restored HF image'},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not C.exists():
  if time.monotonic()>deadline:raise RuntimeError('paired Qwen source collection terminal absent; no serving benchmark started')
  time.sleep(20)
 assert json.loads(C.read_text())['failure'] is None,'paired archive collection failed; no serving benchmark started'
 assert all(sha(code/n)==h for n,h in pins.items()),'predeclared controller input changed'
 probe="import json,subprocess;print(json.dumps({'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()}))"
 state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True));assert not state['compute'],'GPU occupied; no serving benchmark started'
 subprocess.run(['ssh','ai-assistant','mkdir','/data/riley-serving-261007/kernel-batch04-controller'],check=True)
 for n in pins:subprocess.run(['scp',str(code/n),'ai-assistant:/data/riley-serving-261007/kernel-batch04-controller/'+n],check=True)
 with (O/'controller.log').open('x') as f:p=subprocess.run(['ssh','ai-assistant','/data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/kernel-batch04-controller/run.py --attempt 1 --mode quiet'],stdout=f,stderr=subprocess.STDOUT)
 (O/'SSH-observation-exit.json').write_text(json.dumps({'exit':p.returncode,'terminal_inferred':False},indent=2)+'\n')
 assert p.returncode==0,'SSH observation or benchmark failed; inspect actual remote process/terminal before any restart'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'goal_achieved':False},indent=2)+'\n')
