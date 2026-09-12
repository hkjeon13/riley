from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 16 passed' in (r/'mixed-v49-owned.log').read_text()
assert 'test result: ok. 18 passed' in (r/'wire-v49-rust.log').read_text()
assert 'test result: ok. 37 passed' in (r/'wire-v49-scheduler.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'mixed-v49-memcheck.log').read_text()
assert 'test result: ok. 1 passed' in (r/'mixed-v49-cli.log').read_text()
assert json.loads((r/'mixed-v49-fallback-comparison.json').read_text())['exact_match']
http=json.loads((r/'v7-http-v49-c32-final/process.json').read_text());assert http['returncode']==0 and http['responses']==37
subprocess.run(['git','diff','--check'],cwd=src,check=True)
paths=['crates/riley-cuda/build.rs','crates/riley-cuda/src/ffi.rs','crates/riley-cuda/src/graph_resources.rs','crates/riley-runtime/src/llama/executor/config.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-runtime/src/llama/variable_session.rs','crates/riley-scheduler/src/authority.rs','crates/riley-scheduler/src/config.rs','crates/riley-scheduler/src/execution.rs','crates/riley-scheduler/src/scheduler.rs','crates/riley-scheduler/tests/v3_shared_owned_gpu.rs','crates/riley-server/src/engine.rs','crates/riley-server/src/main.rs','kernels/include/riley_cuda.h','kernels/src/compact_shared_result.cuh','kernels/src/ffi_internal.hpp','kernels/src/graph_numerics_precise.cu','kernels/src/graph_resources.cu','kernels/src/prefill_shape_packet.hpp','kernels/src/mixed_attention_v49.cuh','kernels/src/mixed_rope_v49.cuh','kernels/src/mixed_model_v49.cuh','kernels/src/decode_shared32_result.cuh','kernels/tests/mixed_wire_v49_test.cpp']
subprocess.run(['git','add',*paths],cwd=src,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Execute mixed prefill and decode owners in versioned V7 serving graphs'],cwd=src,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
out=r/'variable-candidate-v49';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'mixed-v49-final-build.log'),'sampling_backend':'gpu-greedy; V7 mixed prefill/decode rows; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'mixed-v49.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))

s=(r/'serving_screen_round55.py').read_text().replace('round55','round56').replace('variable-candidate-v48','variable-candidate-v49').replace('variable-candidate-v46','variable-candidate-v48')
s=s.replace("'variable-smol-v6' if name=='new' else 'variable-smol-v4'","'variable-smol-v7' if name=='new' else 'variable-smol-v6'")
s=s.replace('V48','V49').replace('V46','V48').replace('V48 GPU16','V48 packed GPU32').replace('V49 packed GPU32','V49 mixed GPU32').replace('V48 versus V49 packed prefill','V48 versus V49 mixed execution')
(r/'serving_screen_round56.py').write_text(s)
s=(r/'analyze_serving_round55.py').read_text().replace('round55','round56').replace('V48','V49').replace('V46','V48').replace('V48 GPU16','V48 packed GPU32').replace('V49 packed GPU32','V49 mixed GPU32').replace('Assess packed prefill','Assess mixed execution');(r/'analyze_serving_round56.py').write_text(s)
print(json.dumps(build),flush=True)
