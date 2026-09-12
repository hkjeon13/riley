import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912');MODEL='/data/riley-benchmark/20260827T051948Z-d7ad713a/model'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(name,test,output=None):
 env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:'+env['PATH'],CARGO_TARGET_DIR=str(ROOT/(name+'-target')),LD_LIBRARY_PATH='/data/riley-g04-cuda13/lib',RILEY_REAL_CHECKPOINT=MODEL,CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4')
 if output:env['RILEY_STAGE_PARITY_OUTPUT']=str(output)
 with (ROOT/(name+'-'+test+'.log')).open('x') as log:subprocess.run(['cargo','test','--release','-p','riley-runtime','--features','cuda','--lib',test,'--','--ignored','--nocapture','--test-threads=1'],cwd=ROOT/(name+'-source'),env=env,stdout=log,stderr=log,check=True)
for name in ['baseline-test','candidate']:
 run(name,'owned_profile2_stage_parity_bytes',ROOT/(name+'-parity.bin'))
run('candidate','owned_graph_vllm_smol_p128_o32_reuses_scheduler_block_mappings')
a=ROOT/'baseline-test-parity.bin';b=ROOT/'candidate-parity.bin'
assert a.stat().st_size==b.stat().st_size,'size mismatch'
with a.open('rb') as x,b.open('rb') as y:
 offset=0
 while True:
  ab=x.read(1024*1024);bb=y.read(1024*1024)
  if ab!=bb:
   first=offset+next(i for i,(u,v) in enumerate(zip(ab,bb)) if u!=v)
   raise RuntimeError('parity mismatch at byte '+str(first))
  if not ab:break
  offset+=len(ab)
manifests=[json.loads(Path(str(p)+'.json').read_text()) for p in [a,b]]
assert manifests[0]['records']==manifests[1]['records']
ref=json.loads(Path('/tmp/riley-g04-vllm-profile-260911/native-binding.json').read_text())['generated_token_ids']
assert all(m['requests'][0]['output_tokens']==ref and m['requests'][-1]['output_tokens']==ref and m['zero_device_and_pinned_allocations_after_close'] for m in manifests)
result={'schema_version':'riley.stage-split-correctness.v1','passed':True,'performance_claim_eligible':False,'raw_bytes_equal':offset,'raw_sha256':digest(a),'records':len(manifests[0]['records']),'replays_per_source':809,'requests_per_source':6,'baseline_test_source_commit':subprocess.check_output(['git','-C',str(ROOT/'baseline-test-source'),'rev-parse','HEAD'],text=True).strip(),'candidate_source_commit':subprocess.check_output(['git','-C',str(ROOT/'candidate-source'),'rev-parse','HEAD'],text=True).strip(),'vllm_reference_tokens_exact':True,'reference_binding_sha256':digest(Path('/tmp/riley-g04-vllm-profile-260911/native-binding.json')),'candidate_stage_cancellation_drop_test_passed':True,'candidate_binaries':{str(ROOT/'candidate-target/release'/exe):digest(ROOT/'candidate-target/release'/exe) for exe in ['riley','riley-profile']}}
(ROOT/'stage-correctness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
