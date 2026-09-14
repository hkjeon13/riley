import os,json,subprocess,hashlib,shutil
from pathlib import Path
root=Path('/data/riley-serving-260913-recovery');out=root/'projection-cta-serving-build-v1';out.mkdir()
env=os.environ.copy();env.update(PATH=str(root/'toolchain130/nvidia/cu13/bin')+':/data/cmake-3.31.12/bin:/home/psyche/.cargo/bin:/usr/bin:/bin',LD_LIBRARY_PATH=str(root/'toolchain130/nvidia/cu13/lib'),CARGO_TARGET_DIR=str(root/'target'),RILEY_FA3_SOURCE=str(root/'deps/flash-attention'),CARGO_BUILD_JOBS='4')
for name,argv in [('build',['cargo','build','-p','riley-server','--features','server,cuda','--bin','riley','--release','--offline']),('runtime-ticket-tests',['cargo','test','-p','riley-runtime','--features','cuda','--lib','staged_window_tests','--offline'])]:
 with (out/(name+'.log')).open('w') as log:p=subprocess.run(argv,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
 (out/(name+'-exit.json')).write_text(json.dumps({'exit_code':p.returncode,'argv':argv})+'\n');print(name,p.returncode,flush=True)
 assert p.returncode==0,name
binary=root/'target/release/riley';data=binary.read_bytes();h=hashlib.sha256(data).hexdigest()
assert h not in ['a525729d037b519e9c796b7574f960820fb6cbeb1e0d60e4a8a504c4cd616403','e95bd576863b7a7605c7491f1d3a18b9561e5a77d36f89466f3f8a360aaa85af']
dst=root/'riley-projection-cta-serving-v1';assert not dst.exists();shutil.copy2(binary,dst)
(out/'binary.json').write_text(json.dumps({'sha256':h,'path':str(dst)})+'\n')
files=['crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-runtime/src/llama/variable_session.rs','benchmarks/analysis/projection_cta_serving_screen.py','kernels/src/decode_shared32_model.cuh','kernels/src/graph_numerics_precise.cu','kernels/optional/decode_projection_split_rows.cuh']
(out/'sources.json').write_text(json.dumps({f:hashlib.sha256((root/'source'/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n');print(h,flush=True)
