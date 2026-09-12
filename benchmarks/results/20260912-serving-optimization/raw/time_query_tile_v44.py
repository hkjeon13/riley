from pathlib import Path
import os,subprocess,time,json
r=Path('/tmp/riley-opt-260912');out=r/'query-tile-v44';src=r/'prefill-shapes-source-v11'
for width in [8,16]:
 for tool in ['memcheck','racecheck']:
  txt=(out/f'{width}-{tool}.log').read_text();assert ('ERROR SUMMARY: 0 errors' if tool=='memcheck' else 'RACECHECK SUMMARY: 0 hazards') in txt,(width,tool)
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
subprocess.run(['python3',str(r/'prepare_query_timing_v44.py')],check=True)
for width in [8,16]:
 with (out/f'{width}-timing-build.log').open('w') as log:subprocess.run(['nvcc','-std=c++17','-arch=sm_89','-O3','-DRILEY_QUERY_TILE='+str(width),'-I'+str(out),'-I'+str(src/'kernels/src'),str(out/'timing.cu'),'-o',str(out/f'timing{width}')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS timing build',width,flush=True)
for width in [8,16]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
 deadline=time.monotonic()+180
 while int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],env=env,text=True).strip())>48:
  assert time.monotonic()<deadline;time.sleep(2)
 with (out/f'{width}-timing.jsonl').open('w') as log:subprocess.run([str(out/f'timing{width}')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS timing',width,flush=True)
