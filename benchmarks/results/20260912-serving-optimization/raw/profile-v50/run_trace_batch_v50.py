from pathlib import Path
import subprocess,json,os
r=Path('/tmp/riley-opt-260912')
assert len(json.loads((r/'variable-serving-screen-round57/completion.json').read_text())['records'])==24
py='/data/riley-vllm-interim.CfrT9T/venv/bin/python'
for kind,script in [('natural','run_native_trace_v50.py'),('fixed','run_fixed_trace_v50.py')]:
 with (r/f'v50-profile-{kind}.log').open('w') as log:subprocess.run([py,str(r/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS profile',kind,flush=True)
 d=r/('native-trace-v50' if kind=='natural' else 'fixed-trace-v50')
 for cap in ([16,32] if kind=='natural' else [32]):
  with (d/f'c{cap}-export.log').open('w') as log:subprocess.run(['/data/cuda-12.8.1/bin/nsys','export','--type','sqlite','--output',str(d/f'c{cap}.sqlite'),str(d/f'c{cap}.nsys-rep')],stdout=log,stderr=subprocess.STDOUT,check=True)
 with (r/f'v50-profile-{kind}-analysis.log').open('w') as log:subprocess.run([py,str(r/('analyze_trace_v50.py' if kind=='natural' else 'analyze_fixed_trace_v50.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS analysis',kind,flush=True)
