from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 7 passed' in (r/'compact-v46-owned-gpu.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (r/'compact-v46-memcheck-integration.log').read_text()
assert json.loads((r/'compact-v46-fallback-comparison.json').read_text())['exact_match']
p=r/'run_v4_http_v46_c32.py';p.write_text(p.read_text().replace('v4-http-v46-c32-retry2','v4-http-v46-c32-final-verified'))
with (r/'compact-v46-http-final.log').open('w') as log:subprocess.run(['python3',str(p)],stdout=log,stderr=subprocess.STDOUT,check=True)
assert 'SERVER_EXIT 0' in (r/'compact-v46-http-final.log').read_text()
subprocess.run(['git','diff','--check'],cwd=src,check=True)
paths=['crates/riley-cuda/build.rs','crates/riley-cuda/src/ffi.rs','crates/riley-cuda/src/graph_resources.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-runtime/src/llama/variable_session.rs','crates/riley-scheduler/src/execution.rs','crates/riley-scheduler/tests/v3_shared_owned_gpu.rs','crates/riley-server/src/engine.rs','crates/riley-server/src/main.rs','kernels/include/riley_cuda.h','kernels/src/graph_resources.cu','kernels/src/compact_shared_result.cuh','kernels/tests/compact_shared_result_probe.cu']
subprocess.run(['git','add',*paths],cwd=src,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Add compact sixteen-row greedy graph completions with full-logit fallback'],cwd=src,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
out=r/'variable-candidate-v46';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'compact-v46-release-final.log'),'sampling_backend':'gpu-greedy; V4 only; per-iteration full-logit fallback for ineligible sampling'};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n')
(r/'compact-v46.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
s=(r/'serving_screen_round52.py').read_text().replace('round52','round53').replace('variable-candidate-v44','variable-candidate-v46').replace('variable-candidate-v42','variable-candidate-v44').replace('(4,8,16,32)','(16,32)').replace('[4,8,16,32]','[16,32]').replace("option(argv,'--sampling-backend','cpu')","option(argv,'--sampling-backend','gpu-greedy' if name=='new' else 'cpu')").replace('both Riley GPU8 atC4/C8, GPU16 atC16/C32','both Riley GPU16 atC16/C32').replace('both Riley active capacity equals client, same execution width8 atC4/C8 and16 atC16/C32','both Riley active capacity equals client and execution width16; V44 CPU normative, V46 compact GPU greedy').replace('C4/C8/C16/C32; V42 versus V44 shared-query attention','C16/C32; V44 versus V46 compact greedy results')
(r/'serving_screen_round53.py').write_text(s)
s=(r/'analyze_serving_round52.py').read_text().replace('round52','round53').replace('len(raw)==48','len(raw)==24').replace('[4,8,16,32]','[16,32]').replace('C4/C8/C16/C32 matched serving screen; V42 versus V44 shared-query attention versus vLLM; both Riley GPU8 atC4/C8 and16 atC16/C32','C16/C32 matched serving screen; V44 CPU versus V46 compact GPU greedy versus vLLM; both Riley GPU16').replace('Assess dedicated decode batch against previous V3 and vLLM; high concurrency remains unqualified','Assess compact results against V44 and vLLM; broader concurrency and long-term stability remain unqualified');(r/'analyze_serving_round53.py').write_text(s)
print(json.dumps(build),flush=True)
