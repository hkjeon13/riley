"""Independently reconcile every native BF16 logits row with the frozen HF oracle."""
import argparse,hashlib,json,struct
from pathlib import Path
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(path.read_bytes())
def verify(directory,teacher_path,receipt):
 result=load(directory/'qwen3b-p2048-cache-on-full128-native-result.json');native=load(directory/'native-process.json');launch=load(directory/'native-launch.json');log=(directory/'native.log').read_bytes()
 assert sha(log)==native['log_sha256'] and native['result']==result
 assert launch['binary_sha256']==native['binary_sha256_after']
 assert launch['source_revision']==result['source_commit']==receipt['source_commit']=='de64579cdcbeb78cbe638fa76f3c2e0b944cc5cf'
 before=load(directory/'source-git-receipt.json');after=load(directory/'source-integrity-after.json')
 assert before['source_commit']==after['source_commit']==receipt['source_commit'] and before['code_worktree_clean'] and after['code_worktree_clean']
 assert before['verified_source_files']==len(receipt['files'])==433 and after['files']==receipt['files']
 for key,name in [('rust_decode','crates/riley-runtime/src/llama/decode.rs'),('native_decode_attention','kernels/src/decode_attention.cu'),('test','crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs')]:assert result['source_sha256'][key]==receipt['files'][name]
 contract=result['contract'];assert contract['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
 assert contract['prompt_tokens']==2048 and contract['output_logit_rows']==128 and contract['decode_steps']==127 and contract['maximum_logical_length']==2175
 assert contract['dtype']=='BF16' and contract['comparison']=='byte-exact; no tolerance'
 teacher=teacher_path.read_bytes();assert sha(teacher)==contract['teacher_sidecar_sha256']=='d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
 hlen=struct.unpack('<Q',teacher[:8])[0];assert 0<hlen<len(teacher)-8
 header=json.loads(teacher[8:8+hlen]);expected=teacher[8+hlen:];tensor=header['teacher_forced/logits']
 assert tensor['dtype']=='BF16' and tensor['shape']==[128,151936] and tensor['data_offsets']==[0,len(expected)]
 raw=(directory/Path(result['candidate_raw_sidecar']['path']).name).read_bytes();assert sha(raw)==result['candidate_raw_sidecar']['sha256'] and len(raw)==len(expected)==result['candidate_raw_sidecar']['bytes']==128*303872
 assert len(result['rows'])==128;rows=[]
 for i,row in enumerate(result['rows']):
  a=raw[i*303872:(i+1)*303872];e=expected[i*303872:(i+1)*303872]
  assert row=={'row':i,'bf16_exact':a==e,'actual_sha256':sha(a),'expected_sha256':sha(e)}
  mismatch=sum(x!=y for x,y in zip(struct.iter_unpack('<H',a),struct.iter_unpack('<H',e)))
  rows.append({'row':i,'bf16_exact':a==e,'unequal_elements':mismatch,'actual_sha256':sha(a),'expected_sha256':sha(e)})
 first=next((r['row'] for r in rows if not r['bf16_exact']),None);qualified=first is None
 assert first==result['first_non_exact_row'] and result['quality_gate']['full128_teacher_forced_logits_exact'] is qualified
 assert all(result['repeat_execution'].values()) and all(result['invalid_position_guards'].values())
 assert result['quality_gate']['serving_selector_eligible'] is False and result['quality_gate']['free_running_generation_verified'] is False
 assert result['serving_performance']=='미실행' and result['performance_claim_eligible'] is False
 if qualified:
  assert native['exit']==0 and b'test result: ok. 1 passed; 0 failed; 0 ignored' in log and load(directory/'completion.json')['failure'] is None
 else:
  assert native['exit']!=0 and b'test result: FAILED. 0 passed; 1 failed' in log and load(directory/'completion.json')['failure'] is not None
 return {'independent_native_logit_rows':128,'exact_rows':sum(r['bf16_exact'] for r in rows),'first_non_exact_row':first,'rows':rows,'full128_teacher_forced_logits_qualified':qualified,'source_commit':result['source_commit'],'native_binary_sha256':launch['binary_sha256'],'lifecycle_scope':'two same-owner runs, invalid-position guards and zero-allocation close bound to actual source/log; not HTTP cancellation proof','free_running_generation':'unverified','serving_selector':'unverified','serving_performance':'미실행','adopted':False,'goal_achieved':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--teacher-sidecar',required=True,type=Path);p.add_argument('--source-receipt',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=verify(a.directory,a.teacher_sidecar,load(a.source_receipt));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
