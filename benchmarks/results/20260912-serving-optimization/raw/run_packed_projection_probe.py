"""Numerical-only packed projection experiment; existing Blender stays running."""
import hashlib,json,os,subprocess
from pathlib import Path
r=Path('/tmp/riley-opt-260912');src=r/'projection-probe-source';target=r/'projection-probe-target'
name='crates/riley-runtime/src/llama/graph_decode_packed_projection_probe_gpu.rs'
subprocess.run(['git','clone','--no-hardlinks','--quiet',str(r/'batch2-source'),str(src)],check=True)
probe=r/'graph_decode_packed_projection_probe_gpu.rs'
assert hashlib.sha256(probe.read_bytes()).hexdigest()=='5b306e1b455039cab22874490e570043f3fc57b165b889740fc6aba50e730e59'
(src/name).write_bytes(probe.read_bytes())
p=src/'crates/riley-runtime/src/llama/graph_decode_full.rs';p.write_text(p.read_text()+'\n#[cfg(all(test, feature = "cuda"))]\n#[path = "graph_decode_packed_projection_probe_gpu.rs"]\nmod packed_projection_probe_gpu;\n')
subprocess.run(['git','add',name,'crates/riley-runtime/src/llama/graph_decode_full.rs'],cwd=src,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','--quiet','-m','snapshot: packed projection arithmetic experiment'],cwd=src,check=True)
(target/'release').mkdir(parents=True)
subprocess.run(['rsync','-a','--exclude=riley-cuda-*',str(r/'batch2-target/release')+'/',str(target/'release')+'/'],check=True)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(target),LD_LIBRARY_PATH='/data/riley-g04-cuda13/lib',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_PACKED_PROJECTION_PROBE_OUTPUT=str(r/'packed-projection-probe.json'))
with (r/'packed-projection-probe.log').open('x') as log:
 subprocess.run(['cargo','test','--release','-p','riley-runtime','--features','cuda','--lib','packed_projection_numerical_feasibility','--','--ignored','--nocapture','--test-threads=1'],cwd=src,env=env,stdout=log,stderr=log,check=True)
report=json.loads((r/'packed-projection-probe.json').read_text());print(json.dumps({'experiment_completed':report['experiment_completed'],'equal':report['equal'],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'foreign_blender_retained':True,'performance_claim':False}))
