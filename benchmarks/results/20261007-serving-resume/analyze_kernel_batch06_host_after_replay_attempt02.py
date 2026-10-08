import json,time,hashlib,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch06-host-independent-analysis-attempt02';O.mkdir();failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
wait=R/'kernel-batch06-independent-replay-attempt02/completion.json';inputs=['analyze_batch04_phase_host_samples.py','analyze_serving_host_samples.py','analyze_serving_host_samples_v2.py','summarize_baseline.py'];pins={n:sha(R/n) for n in inputs}
(O/'preparation.json').write_text(json.dumps({'wait_for':str(wait),'source_pins':pins,'scope':'all source host observations and conservative phase-bounded samples; unclassified samples retained; no performance exclusions or causal attribution'},indent=2)+'\n')
try:
 deadline=time.monotonic()+43200
 while not wait.exists():
  assert time.monotonic()<deadline,'original independent replay not terminal; do not restart live benchmark'
  time.sleep(30)
 replay=json.loads(wait.read_bytes());assert replay['failure'] is None and all(step.get('exit',0)==0 for step in replay['steps']),'independent replay incomplete or failed'
 raw=json.loads(wait.with_name('raw-replay.json').read_bytes());assert raw['raw_replay_passed'] and raw['source_commits']['candidate']=='e42cc345a08913b2dc1f9a0ea618dd73e3390f8e'
 assert all(sha(R/n)==h for n,h in pins.items())
 collection=json.loads((R/'kernel-batch06-terminal-collection-attempt02/completion.json').read_bytes());assert collection['failure'] is None and len(collection['records'])==1
 record=collection['records'][0];archive=Path(record['archive']);assert sha(archive)==record['archive_sha256']
 for script,name in [('analyze_serving_host_samples_v2.py','all-source-host-observations.json'),('analyze_batch04_phase_host_samples.py','phase-bounded-host-observations.json')]:
  with (O/(script+'.log')).open('x') as log:subprocess.run([sys.executable,str(R/script),str(archive),'--output',str(O/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((O/'phase-bounded-host-observations.json').read_bytes())
 if raw['matrix_complete']:assert result['lanes_with_actual_host_samples']==96 and len(result['reports'])==192
 (O/'archive-binding.json').write_text(json.dumps({'source_archive_sha256':record['archive_sha256'],'source_pins':pins,'serving_matrix_complete':raw['matrix_complete'],'lanes_with_actual_host_samples':result['lanes_with_actual_host_samples'],'goal_achieved':False},indent=2)+'\n')
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'adopted':False,'goal_achieved':False},indent=2)+'\n')
