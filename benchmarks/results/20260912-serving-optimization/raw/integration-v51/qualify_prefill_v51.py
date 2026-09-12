from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
def run(name,args):
 with (r/name).open('w') as log:subprocess.run(args,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
run('prefill-v51-build.log',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley'])
run('prefill-v51-test-build.log',['cargo','test','--release','-p','riley-scheduler','--features','cuda','--test','v3_shared_owned_gpu','--no-run'])
run('prefill-v51-owned.log',[str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'--ignored','--nocapture','--test-threads=1'])

run('prefill-v51-serving-qualification-controller.log',['python3',str(r/'qualify_serving_v51.py')])
