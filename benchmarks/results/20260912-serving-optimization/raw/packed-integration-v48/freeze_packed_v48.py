from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 13 passed' in (r/'packed-v48-owned.log').read_text()
assert 'test result: ok. 16 passed' in (r/'wire-v48-rust.log').read_text()
assert 'test result: ok. 36 passed' in (r/'wire-v48-scheduler.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'packed-v48-memcheck.log').read_text()
assert 'test result: ok. 1 passed' in (r/'packed-v48-cli.log').read_text()
assert json.loads((r/'packed-v48-fallback-comparison.json').read_text())['exact_match']
http=json.loads((r/'v6-http-v48-c32-final/process.json').read_text());assert http['returncode']==0 and http['responses']==37
subprocess.run(['git','diff','--check'],cwd=src,check=True)
paths=['crates/riley-cuda/build.rs','crates/riley-cuda/src/ffi.rs','crates/riley-cuda/src/graph_resources.rs','crates/riley-runtime/src/llama/executor/config.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-runtime/src/llama/variable_session.rs','crates/riley-scheduler/src/authority.rs','crates/riley-scheduler/src/config.rs','crates/riley-scheduler/src/execution.rs','crates/riley-scheduler/src/scheduler.rs','crates/riley-scheduler/tests/v3_shared_owned_gpu.rs','crates/riley-server/src/engine.rs','crates/riley-server/src/main.rs','kernels/include/riley_cuda.h','kernels/src/compact_shared_result.cuh','kernels/src/ffi_internal.hpp','kernels/src/graph_numerics_precise.cu','kernels/src/graph_resources.cu','kernels/src/prefill_shape_packet.hpp','kernels/src/packed_prefill_attention_v48.cuh','kernels/src/packed_prefill_rope_v48.cuh','kernels/src/packed_prefill_model_v48.cuh','kernels/tests/packed_wire_v48_test.cpp']
subprocess.run(['git','add',*paths],cwd=src,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Pack up to four prefill owners in versioned V6 serving graphs'],cwd=src,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
out=r/'variable-candidate-v48';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'packed-v48-final-build.log'),'sampling_backend':'gpu-greedy; V6 packed prefill and32 decode rows; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'packed-v48.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
s=(r/'serving_screen_round54.py').read_text().replace('round54','round55').replace('variable-candidate-v47','variable-candidate-v48')
s=s.replace("'variable-smol-v5' if name=='new' and concurrency==32 else 'variable-smol-v4'","'variable-smol-v6' if name=='new' else 'variable-smol-v4'")
s=s.replace('V47 GPU16 atC16 and GPU32 atC32','V48 packed GPU32 atC16/C32').replace('V47 GPU16 atC16/GPU32 atC32','V48 packed GPU32 atC16/C32').replace('V46 versus V47 thirty-two-row decode','V46 versus V48 packed prefill')
(r/'serving_screen_round55.py').write_text(s)
s=(r/'analyze_serving_round54.py').read_text().replace('round54','round55').replace('V47 GPU16 atC16/GPU32 atC32','V48 packed GPU32 atC16/C32').replace('Assess wider decode against V46','Assess packed prefill against V46');(r/'analyze_serving_round55.py').write_text(s)
print(json.dumps(build),flush=True)
