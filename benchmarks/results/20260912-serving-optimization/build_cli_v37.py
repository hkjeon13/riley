import os,subprocess,pathlib
r=pathlib.Path('/tmp/riley-opt-260912');env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
def run(name,args):
 with (r/name).open('w') as log:subprocess.run(args,cwd=r/'prefill-shapes-source-v11',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
os.umask(0o077)
run('cpu-v37-cli.log',['cargo','test','-p','riley-server','--features','cuda,server','--bin','riley'])
run('cpu-v37-release.log',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley'])
