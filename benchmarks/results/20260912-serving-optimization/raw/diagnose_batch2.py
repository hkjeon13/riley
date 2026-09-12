"""Isolated event diagnosis, only after all retained performance trials finish."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import run_serving_optimization as shared
r=Path('/tmp/riley-opt-260912');src=r/'diagnostic-batch2-source';target=r/'diagnostic-batch2-target'
while not (r/'batch2-engine/summary.json').exists():
 for name in ['batch2-http','batch2-engine']:
  status=r/name/'completion.json'
  if status.exists() and json.loads(status.read_text()).get('completed') is False:raise RuntimeError('performance failed; inspect before diagnosis')
 time.sleep(5)
assert json.loads((r/'batch2-engine/summary.json').read_text())['completed']
subprocess.run(['python3',str(r/'remote_session.py'),'extend'],check=True)
subprocess.run(['git','clone','--no-hardlinks','--quiet',str(r/'batch2-source'),str(src)],check=True)
(target/'release').mkdir(parents=True)
subprocess.run(['rsync','-a','--exclude=riley-cuda-*',str(r/'batch2-target/release')+'/',str(target/'release')+'/'],check=True)
with (r/'diagnostic-batch2-instrumentation.json').open('x') as log:
 subprocess.run(['python3',str(r/'profile_owned_graph_batch2.py'),'instrument','--source-root',str(src),'--apply'],stdout=log,check=True)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(target),LD_LIBRARY_PATH='/data/riley-g04-cuda13/lib')
with (r/'diagnostic-batch2-build.log').open('x') as log:
 subprocess.run(['cargo','build','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley'],cwd=src,env=env,stdout=log,stderr=log,check=True)
plan=json.loads((r/'batch2-http-plan.json').read_text());binding=json.loads((r/'batch2-binding.json').read_text())
preflight=r/'diagnostic-batch2-preflight';preflight.mkdir();shared.preflight(plan,binding,preflight)
subprocess.run(['python3',str(r/'profile_batch2_trial.py')],check=True)
with (r/'diagnostic-batch2-summary.json').open('x') as log:
 subprocess.run(['python3',str(r/'profile_owned_graph_batch2.py'),'summarize','--log',str(r/'diagnostic-batch2-trial.log')],stdout=log,check=True)
print(json.dumps({'diagnosed':True,'instrumented':True,'performance_claim_eligible':False,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binary_sha256':hashlib.sha256((target/'release/riley').read_bytes()).hexdigest()}))
