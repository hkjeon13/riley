from pathlib import Path
import subprocess,json,hashlib,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
assert 'test result: ok. 5 passed' in (r/'buckets-v43-owned.log').read_text()
assert (r/'v4-http-v43-c32-final/shutdown.json').exists()
subprocess.run(['git','diff','--check'],cwd=src,check=True)
subprocess.run(['git','add','kernels/src/graph_resources.cu','crates/riley-scheduler/tests/v3_shared_owned_gpu.rs'],cwd=src,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Capture and select smaller prefill graphs with shared resource ownership'],cwd=src,check=True)
out=r/'variable-candidate-v43';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'buckets-v43-build.log')};(out/'build.json').write_text(json.dumps(build,indent=2)+'\n');print(json.dumps(build))
for name in ['serving_screen','analyze_serving']:
 s=(r/f'{name}_round50.py').read_text().replace('round50','round51').replace('variable-candidate-v42','variable-candidate-v43').replace('variable-candidate-v41','variable-candidate-v42').replace('V41 versus V42 attention loading','V42 versus V43 prefill graph buckets');(r/f'{name}_round51.py').write_text(s)
(r/'buckets-v43.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
