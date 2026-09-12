import pathlib,subprocess,json,hashlib,tarfile,shutil
r=pathlib.Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
log=(r/'prefill-v11-prefill-owner.log').read_text()
assert 'P128_PREFILL_PARITY' in log and 'P128_PREFILL_REUSE' in log and '2 passed; 0 failed' in log
shutil.copy2(r/'prefill-v11-prefill-owner.log',r/'prefill-v3-dynamic-owner-regression.log')
paths=['kernels/src/prefill_shape_projection.cuh','kernels/src/prefill_shape_rope_kv.cuh','kernels/src/prefill_shape_attention.cuh','kernels/tests/prefill_shape_graph_replay.cu']
subprocess.run(['git','add',*paths],cwd=s,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Read V3 live geometry on retained CUDA graph replay'],cwd=s,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip()
p=r/'prefill-v3-dynamic.patch';p.write_bytes(subprocess.check_output(['git','diff','ffbff8132fbf53e6862a18475b79e1f214f95ab1','HEAD'],cwd=s))
files=[p,r/'prefill-v3-graph-build.log',r/'prefill-v3-graph-build-r2.log',r/'prefill-v3-graph-memcheck.log',r/'prefill-v3-dynamic-owner-regression.log']
m=r/'prefill-v3-dynamic-manifest.json';m.write_text(json.dumps({'source_commit':commit,'full_variable_model_owner':False,'files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
with tarfile.open(r/'prefill-v3-dynamic.tar.gz','w:gz') as t:
 for p in files+[m]:t.add(p,arcname=str(p.relative_to(r)))
print(commit)
