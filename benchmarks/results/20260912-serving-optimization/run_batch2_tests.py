"""Correctness gate for the immutable P128 snapshot, not a timing campaign."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912');src=ROOT/'batch2-source';target=ROOT/'batch2-target'
build=json.loads((ROOT/'batch2-build.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==build['source_commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src,text=True)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(target),LD_LIBRARY_PATH='/data/riley-g04-cuda13/lib',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model')
checks=[]
for name in ['batched_prefill_retained_reuse_cancel_and_rejected_output_invalidation','batched_prefill_exact_logits_status_and_every_decode_kv','owned_graph_smol_p128_o32_reuses_scheduler_block_mappings','owned_graph_vllm_smol_p128_o32_reuses_scheduler_block_mappings']:
 subprocess.run(['python3',str(ROOT/'remote_session.py'),'extend'],check=True)
 path=ROOT/('batch2-'+name+'.log')
 with path.open('x') as log:
  subprocess.run(['cargo','test','--release','-p','riley-runtime','--features','cuda','--lib',name,'--','--ignored','--nocapture','--test-threads=1'],cwd=src,env=env,stdout=log,stderr=log,check=True)
 content=path.read_text();assert '1 passed; 0 failed' in content
 checks.append({'name':name,'passed':True,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 print(json.dumps(checks[-1]),flush=True)
with (ROOT/'batch2-profile-unit-tests.log').open('x') as log:
 subprocess.run(['cargo','test','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley-profile'],cwd=src,env=env,stdout=log,stderr=log,check=True)
result={'passed':True,'source_commit':build['source_commit'],'checks':checks,'binaries':build['binaries'],'reference_binding_sha256':hashlib.sha256(Path('/tmp/riley-g04-vllm-profile-260911/native-binding.json').read_bytes()).hexdigest(),'vllm_reference_tokens_exact':True,'full_logits_and_kv_exact':True,'performance_claim_eligible':False}
(ROOT/'batch2-gpu-tests.json').write_text(json.dumps(result,indent=2)+'\n')
subprocess.run(['/data/riley-vllm-interim.CfrT9T/venv/bin/python',str(ROOT/'batch2_http_check.py')],check=True,env=env)
print(json.dumps(result))
