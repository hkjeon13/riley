from pathlib import Path
import subprocess,os
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';proto=r/'query-dispatch-v44'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
assert (proto/'8-timing.jsonl').exists()
for name in ['memcheck','racecheck']:
 assert ('ERROR SUMMARY: 0 errors' if name=='memcheck' else 'RACECHECK SUMMARY: 0 hazards') in (proto/f'8-{name}.log').read_text()
header=(proto/'prefill_query_tile_v44.cuh').read_text();header=header[:header.index('#ifndef RILEY_QUERY_TILE')]
(src/'kernels/src/prefill_query_tile_attention.cuh').write_text(header)
p=src/'kernels/src/prefill_shape_model.cuh';s=p.read_text();s='#include "prefill_query_tile_attention.cuh"\n'+s
old='riley_prefill_shape::attention_shape<<<dim3(capacity,3),96,0,stream>>>(b(3),lk,lv,b(4),capacity,0,reinterpret_cast<const int*>(shape+1),pages,shape+2);'
new='riley_prefill_query_tile::attention<8><<<dim3(max((capacity+7)/8,min(capacity,31U)),9),32,0,stream>>>(b(3),lk,lv,b(4),capacity,0,reinterpret_cast<const int*>(shape+1),pages,shape+2);'
assert old in s;s=s.replace(old,new);p.write_text(s)
p=src/'crates/riley-cuda/build.rs';s=p.read_text();old='        kernels_dir.join("src/prefill_shape_attention.cuh"),';assert old in s;s=s.replace(old,old+'\n        kernels_dir.join("src/prefill_query_tile_attention.cuh"),');p.write_text(s)
p=src/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();old='            include_bytes!("../../../../kernels/src/prefill_shape_attention.cuh").as_slice(),';assert old in s;s=s.replace(old,old+'\n            include_bytes!("../../../../kernels/src/prefill_query_tile_attention.cuh").as_slice(),');p.write_text(s)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',CUDA_VISIBLE_DEVICES='0',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
for name,args in [('build',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley']),('gpu-build',['cargo','test','--release','-p','riley-scheduler','--features','cuda','--test','v3_shared_owned_gpu','--no-run']),('owned',[str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'--ignored','--nocapture','--test-threads=1'])]:
 with (r/f'query-v44-{name}.log').open('w') as log:subprocess.run(args,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
