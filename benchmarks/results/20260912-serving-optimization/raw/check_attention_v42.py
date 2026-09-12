from pathlib import Path
import subprocess,os,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';inc=r/'attention-v42-decode-reference';inc.mkdir()
for p in (src/'kernels/src').glob('*.cuh'):shutil.copy2(p,inc/p.name)
shutil.copy2(r/'attention-v42-baseline/prefill_shape_attention.cuh',inc/'prefill_shape_attention.cuh')
probe=(src/'kernels/tests/prefill_shape_attention_probe.cu').read_text().replace('n>160','n>4096').replace('end<=160','end<=4096');(r/'prefill-attention-v42.cu').write_text(probe)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0')
def run(name,args):
 with (r/name).open('w') as log:subprocess.run(args,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
run('attention-v42-decode-build.log',['nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(inc),str(src/'kernels/tests/shared16_primitive_probe.cu'),'-o',str(r/'attention-v42-decode')])
run('attention-v42-prefill-build.log',['nvcc','-std=c++17','-arch=sm_89','-O3','-I'+str(src/'kernels/src'),str(r/'prefill-attention-v42.cu'),'-o',str(r/'attention-v42-prefill')])
for name in ['decode','prefill']:run('attention-v42-'+name+'-memcheck.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(r/('attention-v42-'+name))])
run('attention-v42-build.log',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley'])
run('attention-v42-gpu-build.log',['cargo','test','--release','-p','riley-scheduler','--features','cuda','--test','v3_shared_owned_gpu','--no-run'])
env.update(RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
run('attention-v42-owned.log',[str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'--ignored','--nocapture','--test-threads=1'])
