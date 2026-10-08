from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');d=r/'gate-workingset-v56'
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
with (d/'build.log').open('w') as log:subprocess.run(['/data/riley-g04-cuda13/bin/nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(d/'probe.cu'),'-o',str(d/'probe')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
with (d/'timing.jsonl').open('w') as out,(d/'checks.log').open('w') as err:subprocess.run([str(d/'probe')],env=env,stdout=out,stderr=err,check=True)
print('PASS working-set timing and output guards',flush=True)
