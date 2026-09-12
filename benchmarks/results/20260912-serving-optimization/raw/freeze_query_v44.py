from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 5 passed' in (r/'query-v44-owned.log').read_text()
assert 'SERVER_EXIT 0' in (r/'query-v44-http.log').read_text()
header=(r/'query-dispatch-v44/prefill_query_tile_v44.cuh').read_text()
wrapper='namespace riley_prefill_query_tile {'+header.split('#endif\nnamespace riley_prefill_query_tile {')[1]
probe=(r/'query-dispatch-v44/probe.cu').read_text().replace('#include "prefill_query_tile_v44.cuh"','#include "prefill_query_tile_attention.cuh"\n#define RILEY_QUERY_TILE 8\n'+wrapper)
(src/'kernels/tests/prefill_query_tile_probe.cu').write_text(probe)
subprocess.run(['git','diff','--check'],cwd=src,check=True)
paths=['kernels/src/prefill_query_tile_attention.cuh','kernels/src/prefill_shape_model.cuh','crates/riley-cuda/build.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','kernels/tests/prefill_query_tile_probe.cu']
subprocess.run(['git','add',*paths],cwd=src,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Share prefill attention across queries with short-input and nonfinite fallback'],cwd=src,check=True)
out=r/'variable-candidate-v44';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'query-v44-build.log')};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');print(json.dumps(build))
for name in ['serving_screen','analyze_serving']:
 s=(r/f'{name}_round51.py').read_text().replace('round51','round52').replace('variable-candidate-v43','variable-candidate-v44').replace('V42 versus V43 prefill graph buckets','V42 versus V44 shared-query attention');(r/f'{name}_round52.py').write_text(s)
(r/'query-v44.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
