import json,hashlib,time,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch04-full-goal-gap-audit-attempt01';O.mkdir();wait=R/'kernel-batch04-independent-replay-attempt01/completion.json';failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();names=['audit_serving_full_goal_gaps.py','summarize_kernel_batch01.py','summarize_baseline.py'];pins={n:sha(R/n) for n in names}
(O/'preparation.json').write_text(json.dumps({'pins':pins,'wait_for':str(wait),'scope':'all8 actual mean gaps and all per-repeat regressions; no candidate adoption'},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not wait.exists():
  if time.monotonic()>deadline:raise RuntimeError('actual independent replay not terminal')
  time.sleep(30)
 terminal=json.loads(wait.read_bytes());assert terminal['failure'] is None and len(terminal['records'])==1 and terminal['records'][0]['failure'] is None
 assert all(sha(R/n)==h for n,h in pins.items())
 base=wait.parent;scope='kernel-batch04-quiet-attempt01';inputs=[base/(scope+'-independent-replay.json'),base/(scope+'-summary.json')]
 with (O/'audit.log').open('x') as log:subprocess.run([sys.executable,str(R/'audit_serving_full_goal_gaps.py'),*[str(x) for x in inputs],'--output',str(O/'all8-goal-gap-audit.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
 (O/'source-binding.json').write_text(json.dumps({'pins':pins,'inputs':{str(x):sha(x) for x in inputs},'independent_replay_terminal_sha256':sha(wait)},indent=2)+'\n')
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'goal_achieved':False},indent=2)+'\n')
