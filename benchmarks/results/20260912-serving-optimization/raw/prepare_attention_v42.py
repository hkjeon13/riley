from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 3 passed' in (r/'attention-v42-owned.log').read_text()
shutil.copy2(r/'prefill-attention-v42.cu',src/'kernels/tests/prefill_shape_attention_probe.cu')
subprocess.run(['git','diff','--check'],cwd=src,check=True)
subprocess.run(['git','add','kernels/src/decode_shared16_attention.cuh','kernels/src/prefill_shape_attention.cuh','kernels/tests/prefill_shape_attention_probe.cu'],cwd=src,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Group attention fragment loads while preserving ordered MMA'],cwd=src,check=True)
out=r/'variable-candidate-v42';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'attention-v42-build.log')};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');print(json.dumps(build))
s=(r/'serving_screen_round49.py').read_text().replace('round49','round50').replace('variable-candidate-v41','variable-candidate-v42').replace('variable-candidate-v40','variable-candidate-v41').replace('V40 versus V41 SIMD validation','V41 versus V42 attention loading');(r/'serving_screen_round50.py').write_text(s)
s=(r/'analyze_serving_round49.py').read_text().replace('round49','round50').replace('V40 versus V41 SIMD validation','V41 versus V42 attention loading');(r/'analyze_serving_round50.py').write_text(s)
s=(r/'run_v4_http_v41_c32.py').read_text().replace('v4-http-v41-c32-final','v4-http-v42-c32-final');(r/'run_v4_http_v42_c32.py').write_text(s)
