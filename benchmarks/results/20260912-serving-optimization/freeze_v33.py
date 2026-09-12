import pathlib,subprocess,hashlib,json,shutil
r=pathlib.Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
files=['kernels/src/decode_shared.cuh']
subprocess.run(['git','diff','--check'],cwd=src,check=True)
subprocess.run(['git','add','--',*files],cwd=src,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Pipeline shared projection loads and bound MMA loop code size'],cwd=src,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip();out=r/'variable-candidate-v33';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'source_commit':commit,'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'shared-v33-release.log')};(out/'build.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
