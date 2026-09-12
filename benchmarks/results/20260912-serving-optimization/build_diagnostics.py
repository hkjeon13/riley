import os,shutil,subprocess,json
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912')
for label,base in [('diagnostic-baseline','/tmp/riley-g04-vllm-profile-source-260911'),('diagnostic-candidate',str(ROOT/'candidate-source'))]:
 src=ROOT/(label+'-source');target=ROOT/(label+'-target')
 subprocess.run(['git','clone','--no-hardlinks','--quiet',base,str(src)],check=True)
 (target/'release').mkdir(parents=True)
 subprocess.run(['rsync','-a','--exclude=riley-cuda-*','/tmp/riley-g04-vllm-profile-target/release/',str(target/'release')+'/'],check=True)
 with (ROOT/(label+'-instrumentation.json')).open('x') as f:subprocess.run(['python3',str(ROOT/'profile_owned_graph.py'),'instrument','--source-root',str(src),'--apply','--projection-events'],stdout=f,check=True)
 env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(target),LD_LIBRARY_PATH='/data/riley-g04-cuda13/lib')
 with (ROOT/(label+'-build.log')).open('x') as log:subprocess.run(['cargo','build','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley'],cwd=src,env=env,stdout=log,stderr=log,check=True)
 print(json.dumps({'built':label}),flush=True)
