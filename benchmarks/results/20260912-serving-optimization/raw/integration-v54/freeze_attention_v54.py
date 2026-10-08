from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
assert 'test result: ok. 16 passed' in (r/'attention-v54-owned.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'attention-v54-memcheck.log').read_text()
assert json.loads((r/'attention-v54-fallback-comparison.json').read_text())['exact_match']
h=json.loads((r/'v7-http-v54-c32-final/process.json').read_text());assert h['returncode']==0 and h['responses']==37
for name in ['memcheck','boundary-memcheck','writer-memcheck','decode-memcheck']:
 assert 'ERROR SUMMARY: 0 errors' in (r/'packed-value-v54'/f'{name}.log').read_text()
for name in ['racecheck','writer-racecheck','decode-racecheck']:
 assert '0 errors, 0 warnings' in (r/'packed-value-v54'/f'{name}.log').read_text()
assert 'writer_cases=84' in (r/'packed-value-v54/writer-correctness.log').read_text()
assert 'cases=168' in (r/'packed-value-v54/decode-correctness.log').read_text()
subprocess.run(['git','diff','--check'],cwd=s,check=True)
paths=['crates/riley-cuda/build.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','kernels/src/decode_gqa_attention_v50.cuh','kernels/src/decode_shared32_model.cuh','kernels/src/mixed_attention_v49.cuh','kernels/src/mixed_model_v49.cuh','kernels/src/mixed_rope_v49.cuh','kernels/src/packed_value_v54.cuh']
assert {line[3:] for line in subprocess.check_output(['git','status','--porcelain'],cwd=s,text=True).splitlines()}==set(paths)
subprocess.run(['git','add',*paths],cwd=s,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Pack V7 value cache for paired attention loads across mixed and decode'],cwd=s,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=s)
out=r/'variable-candidate-v54';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'attention-v54-build.log'),'sampling_backend':'gpu-greedy; V7 packed V direct writes and paired reads, compact shared attention state, fused prefill and grouped decode; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'attention-v54.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=s));print(json.dumps(build),flush=True)
