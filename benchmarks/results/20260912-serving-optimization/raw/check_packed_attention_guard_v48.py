from pathlib import Path
import os,subprocess,shutil
r=Path('/tmp/riley-opt-260912');out=r/'packed-prefill-v48';src=r/'prefill-shapes-source-v11';shutil.copy2(r/'packed_attention_guard_v48.cu',out/'attention-guard.cu')
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
jobs=[('attention-build',['nvcc','-std=c++17','-arch=sm_89','-O3','--fmad=false','-I'+str(out),'-I'+str(src/'kernels/src'),str(out/'attention-guard.cu'),'-o',str(out/'attention-guard')])]
jobs += [('attention-'+tool,['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(out/'attention-guard')]) for tool in ['memcheck','racecheck']]
for name,args in jobs:
 with (out/(name+'.log')).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
