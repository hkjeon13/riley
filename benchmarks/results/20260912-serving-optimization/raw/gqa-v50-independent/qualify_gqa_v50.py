from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
p=s/'crates/riley-runtime/src/llama/graph_decode_full.rs';x=p.read_text();line='include_bytes!("../../../../kernels/src/decode_shared32_attention.cuh").as_slice(),';assert line in x;x=x.replace(line,line+'\ninclude_bytes!("../../../../kernels/src/decode_gqa_attention_v50.cuh").as_slice(),');p.write_text(x)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
def run(name,args):
 with (r/name).open('w') as log:subprocess.run(args,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
for kind in ['memcheck','racecheck']:run('gqa-v50-independent-'+kind+'.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',kind,'--error-exitcode','99',str(r/'gqa-attention-v50-independent/expanded')])
run('gqa-v50-build.log',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley'])
run('gqa-v50-test-build.log',['cargo','test','--release','-p','riley-scheduler','--features','cuda','--test','v3_shared_owned_gpu','--no-run'])
run('gqa-v50-owned.log',[str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'--ignored','--nocapture','--test-threads=1'])
