import os,pathlib,subprocess
r=pathlib.Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11';env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model')
env['RILEY_V3_MODEL_FIXTURE']=str(r/'prefill-full-model-v11')
import re
log=(r/'v3-recorder-checkpoint-v11.log').read_text();match=re.search(r'Running tests/v3_recorder_checkpoint_gpu.rs \(([^)]+)\)',log);assert match
with (r/'v3-recorder-checkpoint-memcheck-v11.log').open('w') as log:subprocess.run(['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','9',match.group(1),'--ignored','--nocapture'],env=env,cwd=s,stdout=log,stderr=subprocess.STDOUT,check=True)
print('checkpoint memcheck passed',flush=True)
