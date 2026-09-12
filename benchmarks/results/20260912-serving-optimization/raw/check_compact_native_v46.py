from pathlib import Path
import subprocess,os
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11';env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
for name,args in [('build',['nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(s/'kernels/src'),'-I'+str(s/'kernels/include'),str(r/'compact_result_probe_v46.cu'),'-o',str(r/'compact-result-v46')])]+[(tool,['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(r/'compact-result-v46')]) for tool in ['memcheck','racecheck']]:
 with (r/f'compact-v46-native-{name}.log').open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
