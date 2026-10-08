import json,time,hashlib,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch05-host-independent-analysis-attempt02';O.mkdir();failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
wait=R/'kernel-batch05-independent-replay-attempt02/completion.json'
inputs=['analyze_batch04_phase_host_samples.py','analyze_serving_host_samples.py','summarize_baseline.py'];pins={n:sha(R/n) for n in inputs}
(O/'preparation.json').write_text(json.dumps({'wait_for':str(wait),'source_pins':pins,'scope':'host observations only; no exclusions or causal attribution'},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not wait.exists():
  if time.monotonic()>deadline:raise RuntimeError('full serving independent replay not complete')
  time.sleep(30)
 replay=json.loads(wait.read_bytes());assert replay['failure'] is None and len(replay['records'])==1 and replay['records'][0]['failure'] is None
 assert all(step.get('exit')==0 for step in replay['records'][0]['steps']), 'independent replay/launch audit incomplete'
 assert all(sha(R/n)==h for n,h in pins.items())
 collection=json.loads((R/'kernel-batch05-terminal-collection-attempt01/completion.json').read_bytes());assert collection['failure'] is None and len(collection['records'])==1
 record=collection['records'][0];archive=Path(record['archive']);assert sha(archive)==record['archive_sha256']
 for script,name in [('analyze_serving_host_samples.py','all-source-host-observations.json'),('analyze_batch04_phase_host_samples.py','phase-bounded-host-observations.json')]:
  with (O/(script+'.log')).open('x') as log:subprocess.run([sys.executable,str(R/script),str(archive),'--output',str(O/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((O/'phase-bounded-host-observations.json').read_bytes());assert result['lanes_with_actual_host_samples']==96 and len(result['reports'])==192
 (O/'archive-binding.json').write_text(json.dumps({'source_archive_sha256':record['archive_sha256'],'source_pins':pins,'goal_achieved':False},indent=2)+'\n')
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'goal_achieved':False},indent=2)+'\n')
