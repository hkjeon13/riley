from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
assert 'test result: ok. 16 passed' in (r/'prefill-v51-owned.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'prefill-v51-memcheck.log').read_text()
assert json.loads((r/'prefill-v51-fallback-comparison.json').read_text())['exact_match']
h=json.loads((r/'v7-http-v51-c32-final/process.json').read_text());assert h['returncode']==0 and h['responses']==37
subprocess.run(['git','diff','--check'],cwd=s,check=True)
paths=['crates/riley-cuda/build.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','kernels/src/mixed_model_v49.cuh','kernels/src/prefill_fused_gate_v51.cuh']
assert set(subprocess.check_output(['git','diff','--name-only'],cwd=s,text=True).splitlines())==set(paths)-{'kernels/src/prefill_fused_gate_v51.cuh'}
subprocess.run(['git','add',*paths],cwd=s,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Fuse prefill gate up projections and SwiGLU for mixed serving'],cwd=s,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=s)
out=r/'variable-candidate-v51';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'prefill-v51-build.log'),'sampling_backend':'gpu-greedy; V7 mixed rows with fused prefill gate up SwiGLU and grouped decode attention; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'prefill-v51.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=s));print(json.dumps(build),flush=True)
