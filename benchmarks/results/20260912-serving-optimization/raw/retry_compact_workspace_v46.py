from pathlib import Path
import os,subprocess
r=Path("/tmp/riley-opt-260912")
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'))
jobs=[('workspace-build',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley']),('http-retry2',['python3',str(r/'run_v4_http_v46_c32.py')])]
for name,args in jobs:
 with (r/f'compact-v46-{name}.log').open('w') as log:subprocess.run(args,cwd=r/'prefill-shapes-source-v11',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
for backend in ['cpu','gpu-greedy']:
 env['V46_BACKEND']=backend
 with (r/f'compact-v46-fallback-{backend}.log').open('w') as log:subprocess.run(['python3',str(r/'run_compact_fallback_lane_v46.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS fallback',backend,flush=True)
import json
cpu=json.loads((r/'v4-fallback-v46-cpu/responses.json').read_text());gpu=json.loads((r/'v4-fallback-v46-gpu-greedy/responses.json').read_text());assert cpu==gpu
(r/'compact-v46-fallback-comparison.json').write_text(json.dumps({'responses_per_backend':len(cpu),'exact_match':True,'sequential':6,'concurrent':16,'temperatures':[0,0.7],'top_p':0.9,'seeds':'1234 + request index','mask_and_penalty':'not exposed by HTTP; existing typed eligibility unit test required'},indent=2))
print('PASS fallback exact parity',flush=True)
