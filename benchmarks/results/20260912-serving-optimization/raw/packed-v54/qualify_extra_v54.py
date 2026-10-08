from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');d=r/'packed-value-v54'
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
def run(name,args):
 with (d/name).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
for name in ['writer','decode']:
 run(name+'-build.log',['/data/riley-g04-cuda13/bin/nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(d/(name+'.cu')),'-o',str(d/name)])
 run(name+'-correctness.log',[str(d/name)])
 for sanitizer in ['memcheck','racecheck']:
  run(name+'-'+sanitizer+'.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',sanitizer,'--error-exitcode','99',str(d/name)])
with (d/'decode-timing.jsonl').open('w') as out,(d/'decode-timing-check.log').open('w') as err:subprocess.run([str(d/'decode'),'--timing'],env=env,stdout=out,stderr=err,check=True)
print('PASS decode timing',flush=True)
