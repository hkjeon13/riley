from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
p=src/'crates/riley-scheduler/tests/v3_shared_owned_gpu.rs';s=p.read_text();sig='fn run_shared<const ROWS:usize>(active:usize,request_count:usize,physical:usize)->Result<(),Box<dyn std::error::Error>>{'
s=s.replace(sig,'''#[test]
#[ignore="requires pinned checkpoint and GPU; all prefill graph buckets"]
fn loaded_prefill_buckets_eight_rows()->Result<(),Box<dyn std::error::Error>>{run_shared_profile::<8>(4,3,64,512,129)}
#[test]
#[ignore="requires pinned checkpoint and GPU; all prefill graph buckets"]
fn loaded_prefill_buckets_sixteen_rows()->Result<(),Box<dyn std::error::Error>>{run_shared_profile::<16>(4,3,64,512,129)}
'''+sig+'''
run_shared_profile::<ROWS>(active,request_count,physical,128,73)
}
fn run_shared_profile<const ROWS:usize>(active:usize,request_count:usize,physical:usize,capacity:usize,chunk:usize)->Result<(),Box<dyn std::error::Error>>{''')
s=s.replace('::<ROWS>(&context,128)?','::<ROWS>(&context,capacity)?').replace('iteration_token_budget:73,max_prefill_chunk_tokens:73','iteration_token_budget:chunk,max_prefill_chunk_tokens:chunk').replace('vec![17;128]','vec![17;chunk+1]');p.write_text(s)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
for name,args in [('build',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley']),('gpu-build',['cargo','test','--release','-p','riley-scheduler','--features','cuda','--test','v3_shared_owned_gpu','--no-run']),('owned',[str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'--ignored','--nocapture','--test-threads=1'])]:
 with (r/f'buckets-v43-{name}.log').open('w') as log:subprocess.run(args,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
