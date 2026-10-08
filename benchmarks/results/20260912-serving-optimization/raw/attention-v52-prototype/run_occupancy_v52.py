from pathlib import Path
import os,subprocess,json,hashlib
r=Path('/tmp/riley-opt-260912');d=r/'mapped-occupancy-v52';d.mkdir()
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
cmd=['/data/riley-g04-cuda13/bin/nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(r/'mapped_occupancy_v52.cu'),'-o',str(d/'probe')]
with (d/'build.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
with (d/'attributes.log').open('w') as log:subprocess.run([str(d/'probe')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
cmd=['/data/cuda-12.8.1/bin/ncu','--section','LaunchStats','--section','Occupancy','--section','SpeedOfLight','--section','SchedulerStats','--section','WarpStateStats','--kernel-name','regex:mapped_attention','--launch-count','1','--export',str(d/'baseline'),str(d/'probe')]
with (d/'ncu.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
(d/'receipt.json').write_text(json.dumps({'scope':'synthetic P398 plus31decode metadata, production V51 mapped kernel. Diagnostic only, not serving timing.','ncu_exit_code':p.returncode,'ncu_command':cmd,'probe_sha256':hashlib.sha256((d/'probe').read_bytes()).hexdigest()},indent=2)+'\n');print((d/'attributes.log').read_text(),p.returncode)
