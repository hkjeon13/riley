import json,hashlib,subprocess
from pathlib import Path
R=Path('/Users/psyche/PycharmProjects/riley/benchmarks/results/20261007-serving-resume');C=R/'qwen-isolated-norm-independent-collection-attempt01';I=R/'qwen-layer3-isolated-norm-inputs-attempt01';H=C/'qwen-layer3-isolated-norm-hf-validation-attempt01';N=C/'qwen-isolated-norm-native-validation-attempt01'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((I/'manifest.json').read_text());hf=json.loads((H/'result.json').read_text());source=json.loads((R/'qwen-isolated-norm-source-receipt-attempt01.json').read_text())
archives=[]
for folder,terminal in [(H,'controller-completion.json'),(N,'completion.json')]:
 receipt=json.loads((C/(folder.name+'-receipt.json')).read_text());assert receipt['completion']['failure'] is None and json.loads((folder/terminal).read_text())['failure'] is None
 assert receipt['source_before']==receipt['source_after'];assert sha(C/(folder.name+'.tar.gz'))==receipt['archive_sha256']
 for name,item in receipt['source_before'].items():assert sha(folder/name)==item['sha256'] and (folder/name).stat().st_size==item['bytes']
 archives.append({'name':folder.name,'archive_sha256':receipt['archive_sha256'],'bytes':receipt['archive_bytes'],'files_verified':receipt['files_verified']})
 prep=json.loads((folder/'preparation.json').read_text());after=json.loads((folder/'pins-after.json').read_text());assert prep['pins']==after
for k,v in hf['producer'].items():assert v==manifest['HF_producer'][k]
assert hf['Qwen2_source_sha256']==manifest['HF_producer']['transformers_qwen2_source']['sha256'];assert hf['input_manifest_sha256']==sha(I/'manifest.json');assert hf['weight_sha256']==sha(H/'norm-weight.bf16');assert (H/'norm-weight.bf16').stat().st_size==4096
assert json.loads((N/'source-git-receipt.json').read_text())['source_commit']==source['source_commit'];integrity=json.loads((N/'source-integrity-after.json').read_text());assert integrity['files']==source['files'] and integrity['code_worktree_clean']
assert json.loads((N/'build-process.json').read_text())['exit']==0
launch=json.loads((N/'native-launch.json').read_text());process=json.loads((N/'native-process.json').read_text());assert process['exit']==0 and launch['binary_sha256']==process['binary_sha256_after'] and process['log_sha256']==sha(N/'native.log')
w=Path('/Users/psyche/.codex/worktrees/qwen-cache-full128/riley');changed=subprocess.check_output(['git','-C',str(w),'diff','--name-only','de64579cdcbeb78cbe638fa76f3c2e0b944cc5cf',source['source_commit']],text=True).splitlines();assert changed==['crates/riley-cuda/tests/primitives_gpu.rs']
rows=[]
for item in manifest['rows']:
 files={f['role']:f for f in item['files']}
 for f in files.values():assert sha(I/f['name'])==f['sha256']
 expected_hf=(I/files['HF_output']['name']).read_bytes();expected_native=(I/files['native_output']['name']).read_bytes();step=item['step'];shapes=next(x['shapes'] for x in hf['rows'] if x['step']==step)
 for shape in shapes:
  b=(H/shape['output_file']).read_bytes();assert len(b)==4096 and sha(H/shape['output_file'])==shape['output_sha256'];assert shape['captured_HF_exact']==(b==expected_hf) and shape['captured_native_exact']==(b==expected_native)
  if shape['rows']==1:assert b==expected_hf
 for count in [1,2048]:assert (N/f'step{step}-rows{count}-native-output.bf16').read_bytes()==expected_native
 diff=sum(expected_hf[i:i+2]!=expected_native[i:i+2] for i in range(0,4096,2));assert diff==(0 if step==108 else 16)
 rows.append({'step':step,'native_fresh_weight_replay_exact':True,'native_HF_unequal_BF16_elements':diff,'HF_actual_shapes':shapes})
assert hf['rows'][1]['shapes'][0]['variance_FP32_bits']=='0x3e66144e';assert hf['rows'][1]['shapes'][1]['variance_FP32_bits']=='0x3e66144c';assert hf['rows'][1]['shapes'][2]['captured_native_exact']
result={'independent_raw_replay':'PASS','archives':archives,'source_commit':source['source_commit'],'verified_source_files':len(source['files']),'native_binary_sha256':launch['binary_sha256'],'weight_sha256':hf['weight_sha256'],'rows':rows,'finding':'step109 singleton Torch reduction produces variance0x3e66144e; multirow Torch produces0x3e66144c and exact captured native output; unchanged native RMSNorm with fresh checkpoint weight reproduces16 differing BF16 elements at singleton and2048 rows','limits':'shape-dependent RMSNorm mismatch isolated; full128 correction, bounded M1/M2 regression, free-running and serving remain unverified','serving_performance':'미실행','goal_achieved':False}
p=C/'independent-replay.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
