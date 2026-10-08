from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');d=r/'packed-value-v54'
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
def run(name,cmd):
 with (d/name).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
run('boundary-build.log',['/data/riley-g04-cuda13/bin/nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(d/'boundary.cu'),'-o',str(d/'boundary')])
run('boundary-correctness.log',[str(d/'boundary')])
run('boundary-memcheck.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(d/'boundary')])
