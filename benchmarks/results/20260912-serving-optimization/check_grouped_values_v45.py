from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');out=r/'grouped-values-v45';src=r/'prefill-shapes-source-v11'
assert subprocess.run(['pgrep','-f','^/data/riley-vllm-interim.CfrT9T/venv/bin/python /tmp/riley-opt-260912/serving_screen_round52.py$'],stdout=subprocess.DEVNULL).returncode==1,'Round52 is still running'
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
jobs=[('build',['nvcc','-std=c++17','-arch=sm_89','-O3','-Xptxas=-v','-I'+str(out),'-I'+str(src/'kernels/src'),str(out/'probe.cu'),'-o',str(out/'probe')])]
jobs += [(tool,['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(out/'probe')]) for tool in ['memcheck','racecheck']]
for name,args in jobs:
 with (out/f'{name}.log').open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
