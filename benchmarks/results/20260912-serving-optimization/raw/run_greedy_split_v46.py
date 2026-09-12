from pathlib import Path
import subprocess,os,time
r=Path('/tmp/riley-opt-260912');o=r/'greedy-split-v46';src=r/'prefill-shapes-source-v11'
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()

jobs=[('build',['nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(src/'kernels/include'),str(o/'probe.cu'),'-o',str(o/'probe')])]
jobs += [(tool,['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(o/'probe')]) for tool in ['memcheck','racecheck']]
for name,args in jobs:
 with (o/f'{name}.log').open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
deadline=time.monotonic()+300
while int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],env=env,text=True).strip())>48:
 assert time.monotonic()<deadline;time.sleep(2)
with (o/'timing.jsonl').open('w') as log:subprocess.run([str(o/'probe'),'time'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PASS timing',flush=True)
