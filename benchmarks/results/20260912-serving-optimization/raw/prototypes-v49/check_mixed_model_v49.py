from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');d=r/'mixed-model-v49';src=r/'prefill-shapes-source-v11'
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
jobs=[('build',['nvcc','-std=c++17','-arch=sm_89','-O3','--fmad=false','-I'+str(d),'-I'+str(src/'kernels/src'),str(d/'probe.cu'),'-o',str(d/'probe')]),('correctness',[str(d/'probe'),str(r/'loaded-rope-fixture-v11')]),('memcheck',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(d/'probe'),str(r/'loaded-rope-fixture-v11')])]
for name,args in jobs:
 with (d/(name+'.log')).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS model',name,flush=True)
