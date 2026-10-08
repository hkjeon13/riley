from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
assert 'test result: ok. 16 passed' in (r/'attention-v55-owned.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'attention-v55-memcheck.log').read_text()
assert json.loads((r/'attention-v55-fallback-comparison.json').read_text())['exact_match']
h=json.loads((r/'v7-http-v55-c32-final/process.json').read_text());assert h['returncode']==0 and h['responses']==37
assert 'cases=2688' in (r/'decode-mask-v55/correctness.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'decode-mask-v55/memcheck.log').read_text()
assert '0 errors, 0 warnings' in (r/'decode-mask-v55/racecheck.log').read_text()
subprocess.run(['git','diff','--check'],cwd=s,check=True)
paths=['kernels/src/decode_gqa_attention_v50.cuh']
assert {line[3:] for line in subprocess.check_output(['git','status','--porcelain'],cwd=s,text=True).splitlines()}==set(paths)
subprocess.run(['git','add',*paths],cwd=s,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Separate full K16 decode value tiles from masked partial tails'],cwd=s,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=s)
out=r/'variable-candidate-v55';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'attention-v55-build.log'),'sampling_backend':'gpu-greedy; V7 packed V direct writes and paired reads with full-tile decode fast path and branchless tail mask, compact shared attention state, fused prefill and grouped decode; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'attention-v55.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=s));print(json.dumps(build),flush=True)
