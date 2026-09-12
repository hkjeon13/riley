import json,subprocess
from pathlib import Path
import run_serving_optimization as shared
r=Path('/tmp/riley-opt-260912');base=Path('/tmp/riley-g04-vllm-profile-260911');plan=json.loads((base/'measurement-plan.json').read_text());binding=json.loads((base/'native-binding.json').read_text())
for projection in ['off','q','gate','down']:
 for label in ['diagnostic-baseline','diagnostic-candidate']:
  output=label+('-'+projection if projection!='off' else '')
  preflight=r/(output+'-preflight');preflight.mkdir()
  shared.preflight(plan,binding,preflight)
  subprocess.run(['python3',str(r/'profile_trial.py'),'--label',label,'--projection',projection],check=True)
  with (r/(output+'-summary.json')).open('x') as f:subprocess.run(['python3',str(r/'profile_owned_graph.py'),'summarize','--log',str(r/(output+'-trial.log'))],stdout=f,check=True)
  print(json.dumps({'completed':output}),flush=True)
