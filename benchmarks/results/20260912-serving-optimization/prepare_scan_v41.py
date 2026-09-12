from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 3 passed' in (r/'scan-v41-owned-gpu.log').read_text()
files=['crates/riley-runtime/src/llama/multi_descriptor/mod.rs','crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-runtime/src/llama/multi_descriptor/result_scan.rs','crates/riley-runtime/src/llama/graph_decode_full.rs']
subprocess.run(['git','diff','--check'],cwd=src,check=True);subprocess.run(['git','add','--',*files],cwd=src,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Vectorize complete BF16 result and inactive-buffer validation'],cwd=src,check=True)
out=r/'variable-candidate-v41';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'scan-v41-build.log'),'change':'AVX2 full validation with scalar fallback, four independent reductions and vectorized inactive-byte check'}
(out/'build.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
s=(r/'run_v4_http_v40_c32.py').read_text().replace('v4-http-v40-c32-final','v4-http-v41-c32-final');(r/'run_v4_http_v41_c32.py').write_text(s)
s=(r/'serving_screen_round48.py').read_text().replace('round48','round49').replace('variable-candidate-v40','variable-candidate-v41').replace('variable-candidate-v37','variable-candidate-v40')
s=s.replace("'variable-smol-v4' if name=='new' else 'variable-smol-v3'", "'variable-smol-v3' if concurrency<=8 else 'variable-smol-v4'")
s=s.replace('all admission equals client; previous GPU8, new GPU16','all admission equals client; both Riley GPU8 atC4/C8, GPU16 atC16/C32')
s=s.replace('both Riley active capacity equals client, previous GPU8 versus new GPU16','both Riley active capacity equals client, same execution width8 atC4/C8 and16 atC16/C32')
s=s.replace('V37 GPU8 versus V40 GPU16 versus vLLM','V40 versus V41 SIMD validation versus vLLM, Riley matched profile by concurrency')
(r/'serving_screen_round49.py').write_text(s)
s=(r/'analyze_serving_round48.py').read_text().replace('round48','round49').replace('V37 GPU8 versus V40 GPU16 versus vLLM','V40 versus V41 SIMD validation versus vLLM; both Riley GPU8 atC4/C8 and16 atC16/C32');(r/'analyze_serving_round49.py').write_text(s)
