import pathlib,subprocess,hashlib,json,shutil
r=pathlib.Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
files=['crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-scheduler/src/execution.rs','crates/riley-server/src/engine.rs']
subprocess.run(['git','diff','--check'],cwd=src,check=True)
subprocess.run(['git','add','--',*files],cwd=src,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Reuse validated V3 argmax for eligible greedy sampling and parallelize logit validation'],cwd=src,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip();out=r/'variable-candidate-v30';out.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',out/'riley')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'source_commit':commit,'binaries':{str(out/'riley'):sha(out/'riley')},'build_log_sha256':sha(r/'cpu-v30-release.log')};(out/'build.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
