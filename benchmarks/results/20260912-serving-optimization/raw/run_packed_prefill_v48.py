from pathlib import Path
import os,subprocess,shutil
r=Path('/tmp/riley-opt-260912');out=r/'packed-prefill-v48';src=r/'prefill-shapes-source-v11';shutil.copy2(r/'packed_prefill_probe_v48.cu',out/'probe.cu')
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
# Match graph_numerics_precise.cu arithmetic options.
jobs=[(tool,['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(out/'probe'),str(r/'prefill-full-model-v11')]) for tool in ['memcheck','racecheck']]
jobs += [('timing',[str(out/'probe'),str(r/'prefill-full-model-v11'),'timing'])]
for name,args in jobs:
 with (out/(name+'.log')).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
