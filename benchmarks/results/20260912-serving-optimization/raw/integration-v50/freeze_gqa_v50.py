from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
assert 'test result: ok. 16 passed' in (r/'gqa-v50-owned.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'gqa-v50-memcheck.log').read_text()
assert '0 errors, 0 warnings' in (r/'gqa-v50-independent-racecheck.log').read_text()
assert json.loads((r/'gqa-v50-fallback-comparison.json').read_text())['exact_match']
h=json.loads((r/'v7-http-v50-c32-final/process.json').read_text());assert h['returncode']==0 and h['responses']==37
subprocess.run(['git','diff','--check'],cwd=s,check=True)
paths=['crates/riley-cuda/build.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','kernels/src/decode_shared32_model.cuh','kernels/src/ffi_internal.hpp','kernels/src/graph_numerics_precise.cu','kernels/src/graph_resources.cu','kernels/src/decode_gqa_attention_v50.cuh']
assert set(subprocess.check_output(['git','diff','--name-only'],cwd=s,text=True).splitlines())==set(paths)-{'kernels/src/decode_gqa_attention_v50.cuh'}
subprocess.run(['git','add',*paths],cwd=s,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Group decode QK heads and pack value warps for mixed serving'],cwd=s,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=s)
out=r/'variable-candidate-v50';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'gqa-v50-build.log'),'sampling_backend':'gpu-greedy; V7 mixed rows with grouped decode QK and independent three-warp values; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'gqa-v50.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=s));print(json.dumps(build),flush=True)
