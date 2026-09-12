from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 10 passed' in (r/'shared32-v47-owned-gpu.log').read_text()
assert 'test result: ok. 14 passed' in (r/'shared32-v47-wire-regression.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'shared32-v47-memcheck-integration.log').read_text()
assert 'test result: ok. 2 passed' in (r/'shared32-v47-final-cli-tests.log').read_text()
assert json.loads((r/'shared32-v47-fallback-comparison.json').read_text())['exact_match']
p=r/'run_v5_http_v47.py';p.write_text(p.read_text().replace('v5-http-v47-c32','v5-http-v47-c32-final'))
with (r/'shared32-v47-http-final.log').open('w') as log:subprocess.run(['python3',str(p)],stdout=log,stderr=subprocess.STDOUT,check=True)
s=(r/'shared32-qkv-v47/probe.cu').read_text().replace('decode_shared32_v47.cuh','decode_shared32.cuh').replace('decode_shared32_attention_v47.cuh','decode_shared32_attention.cuh');(src/'kernels/tests/shared32_primitive_probe.cu').write_text(s)
subprocess.run(['git','diff','--check'],cwd=src,check=True)
paths=['crates/riley-cuda/build.rs','crates/riley-cuda/src/ffi.rs','crates/riley-cuda/src/graph_resources.rs','crates/riley-runtime/src/llama/executor/config.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-runtime/src/llama/variable_session.rs','crates/riley-scheduler/src/config.rs','crates/riley-scheduler/src/execution.rs','crates/riley-scheduler/src/scheduler.rs','crates/riley-scheduler/tests/v3_shared_owned_gpu.rs','crates/riley-server/src/engine.rs','crates/riley-server/src/main.rs','kernels/include/riley_cuda.h','kernels/src/compact_shared_result.cuh','kernels/src/ffi_internal.hpp','kernels/src/gemm.cu','kernels/src/graph_numerics_precise.cu','kernels/src/graph_resources.cu','kernels/src/prefill_shape_model.cuh','kernels/src/prefill_shape_packet.hpp','kernels/src/decode_shared32.cuh','kernels/src/decode_shared32_attention.cuh','kernels/src/decode_shared32_model.cuh','kernels/src/decode_shared32_result.cuh','kernels/tests/shared32_primitive_probe.cu']
subprocess.run(['git','add',*paths],cwd=src,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Add versioned thirty-two-row decode serving with shared projection weights'],cwd=src,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
out=r/'variable-candidate-v47';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'shared32-v47-final-native-build.log'),'sampling_backend':'gpu-greedy; V4 preserved, V5 permits32 rows; full-logit fallback supported'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');(r/'shared32-v47.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
s=(r/'serving_screen_round53.py').read_text().replace('round53','round54').replace('variable-candidate-v46','variable-candidate-v47').replace('variable-candidate-v44','variable-candidate-v46')
s=s.replace("'variable-smol-v3' if concurrency<=8 else 'variable-smol-v4'","'variable-smol-v5' if name=='new' and concurrency==32 else 'variable-smol-v4'").replace("'gpu-greedy' if name=='new' else 'cpu'","'gpu-greedy'")
s=s.replace('both Riley GPU16 atC16/C32','V46 GPU16 atC16/C32, V47 GPU16 atC16 and GPU32 atC32').replace('both Riley active capacity equals client and execution width16; V44 CPU normative, V46 compact GPU greedy','both Riley active capacity equals client; V46 GPU16, V47 GPU16 atC16 and GPU32 atC32; both compact GPU greedy').replace('V44 versus V46 compact greedy results','V46 versus V47 thirty-two-row decode')
(r/'serving_screen_round54.py').write_text(s)
s=(r/'analyze_serving_round53.py').read_text().replace('round53','round54').replace('V44 CPU versus V46 compact GPU greedy','V46 GPU16 versus V47 GPU16 atC16/GPU32 atC32').replace('both Riley GPU16','both Riley compact GPU greedy').replace('Assess compact results against V44','Assess wider decode against V46');(r/'analyze_serving_round54.py').write_text(s)
print(json.dumps(build),flush=True)
