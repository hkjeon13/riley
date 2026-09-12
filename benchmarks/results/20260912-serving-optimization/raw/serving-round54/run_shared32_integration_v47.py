from pathlib import Path
import os,subprocess,json
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip(),'GPU is occupied'
jobs=[('memcheck',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'loaded_v5','--ignored','--test-threads=1','--nocapture']),('http',['python3',str(r/'run_v5_http_v47.py')])]
for name,args in jobs:
 with (r/f'shared32-v47-{name}-integration.log').open('w') as log:subprocess.run(args,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
for backend in ['cpu','gpu-greedy']:
 env['V46_BACKEND']=backend
 with (r/f'shared32-v47-fallback-{backend}.log').open('w') as log:subprocess.run(['python3',str(r/'run_v5_fallback_v47.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS fallback',backend,flush=True)
a=json.loads((r/'v5-fallback-ordered-v47-cpu/responses.json').read_text());b=json.loads((r/'v5-fallback-ordered-v47-gpu-greedy/responses.json').read_text());assert a==b
(r/'shared32-v47-fallback-comparison.json').write_text(json.dumps({'exact_match':True,'responses_per_backend':len(a),'request_ids_exact':True},indent=2))
print('PASS fallback parity',flush=True)
