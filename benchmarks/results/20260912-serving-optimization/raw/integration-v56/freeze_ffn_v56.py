from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
assert 'test result: ok. 16 passed' in (r/'ffn-v56-owned.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'ffn-v56-memcheck.log').read_text()
assert json.loads((r/'ffn-v56-fallback-comparison.json').read_text())['exact_match']
h=json.loads((r/'v7-http-v56-c32-final/process.json').read_text());assert h['returncode']==0 and h['responses']==37
for directory,cases in [('decode-gate-layout-v56',1632),('merge-norm-v56',816)]:
 assert f'cases={cases}' in (r/directory/'correctness.log').read_text()
 assert 'ERROR SUMMARY: 0 errors' in (r/directory/'memcheck.log').read_text()
 assert '0 errors, 0 warnings' in (r/directory/'racecheck.log').read_text()
subprocess.run(['git','diff','--check'],cwd=s,check=True)
paths=['kernels/src/decode_gate_v56.cuh','kernels/src/decode_merge_norm_v56.cuh','kernels/src/decode_shared32_model.cuh','crates/riley-cuda/build.rs','crates/riley-runtime/src/llama/graph_decode_full.rs']
assert {line[3:] for line in subprocess.check_output(['git','status','--porcelain'],cwd=s,text=True).splitlines()}==set(paths)
subprocess.run(['git','add',*paths],cwd=s,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Fuse decode gate rows and projection residual normalization'],cwd=s,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=s)
out=r/'variable-candidate-v56';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'ffn-v56-build.log'),'sampling_backend':'gpu-greedy; V7 token-major V, fused gate/up row warps and projection merge residual normalization; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'ffn-v56.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=s));print(json.dumps(build),flush=True)
