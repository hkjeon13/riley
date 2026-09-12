from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11';env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'))
for name,args in [('native-build',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley']),('cli-tests',['cargo','test','--release','-p','riley-server','--features','cuda,server','--bin','riley','variable_v'])]:
 with (r/f'shared32-v47-final-{name}.log').open('w') as log:subprocess.run(args,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
