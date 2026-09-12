from pathlib import Path
import subprocess,hashlib,json,shutil
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';p=src/'crates/riley-runtime/src/llama/variable_session.rs';saved=p.read_bytes()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip();assert commit=='214c8ed7309d00ea908c7e29e3c2ed962f8f945d'
try:
 subprocess.run(['python3',str(r/'profile_fill_v29.py')],check=True)
 patch=subprocess.check_output(['git','diff'],cwd=src);assert b'BATCH stage=' in patch;(r/'fill-v34-instrumentation.patch').write_bytes(patch)
 subprocess.run(['python3',str(r/'build_fill_v34.py')],check=True)
 out=r/'fill-diagnostic-v34';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley')
 (out/'build.json').write_text(json.dumps({'base_commit':commit,'binary_sha256':hashlib.sha256((out/'riley').read_bytes()).hexdigest(),'instrumentation_patch_sha256':hashlib.sha256(patch).hexdigest()},indent=2)+'\n')
finally:
 p.write_bytes(saved);shutil.copy2(r/'variable-candidate-v33/riley',r/'prefill-shapes-target-v11/release/riley')
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
 assert hashlib.sha256((r/'prefill-shapes-target-v11/release/riley').read_bytes()).hexdigest()=='5ffceea0d2494f951cf3b92115ad04063bbadb8e5c45c3c5ffdf67a3cd8ec7a3'
subprocess.run(['/data/riley-vllm-interim.CfrT9T/venv/bin/python',str(r/'run_fill_v34.py')],check=True)
