from pathlib import Path
import os,subprocess,time
r=Path('/tmp/riley-opt-260912');o=r/'paired-values-v45';src=r/'prefill-shapes-source-v11'
for name,needle in [('memcheck','ERROR SUMMARY: 0 errors'),('racecheck','RACECHECK SUMMARY: 0 hazards')]:assert needle in (o/f'{name}.log').read_text()
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()

with (o/'timing-build.log').open('w') as log:subprocess.run(['nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(o),'-I'+str(src/'kernels/src'),str(o/'timing.cu'),'-o',str(o/'timing')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PASS timing build',flush=True)
deadline=time.monotonic()+300
while int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],env=env,text=True).strip())>48:
 assert time.monotonic()<deadline;time.sleep(2)
with (o/'timing.jsonl').open('w') as log:subprocess.run([str(o/'timing')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PASS timing',flush=True)
