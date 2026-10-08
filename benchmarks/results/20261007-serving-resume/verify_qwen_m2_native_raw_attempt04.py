"""Reconcile actual native BF16 bytes with immutable HF M1/M2 and teacher rows."""
import argparse,hashlib,json,math,struct
from pathlib import Path
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(path.read_bytes())
def safetensors(raw):
    length=struct.unpack('<Q',raw[:8])[0];assert 0<length<len(raw)-8
    return json.loads(raw[8:8+length]),memoryview(raw)[8+length:]
def verify(directory,hf_directory,teacher_path,source_receipt):
    result_path=directory/'qwen3b-p2048-cache-on-m1-m2-native-result.json'
    result=load(result_path);native=load(directory/'native-process.json');launch=load(directory/'native-launch.json')
    log=(directory/'native.log').read_bytes();assert native['log_sha256']==sha(log) and native['result']==result
    assert launch['binary_sha256']==native['binary_sha256_after']
    assert launch['source_revision']==result['source_commit']==source_receipt['source_commit']=='2612208b9ac02ffa79b563b87f073c239b35da18'
    source=load(directory/'source-git-receipt.json');assert source['source_commit']==result['source_commit']
    assert source['code_worktree_clean'] is True and source['verified_source_files']==len(source_receipt['files'])==433
    after=load(directory/'source-integrity-after.json')
    assert after['source_commit']==result['source_commit'] and after['code_worktree_clean'] is True and after['files']==source_receipt['files']
    assert result['source_sha256']['rust_decode']==source_receipt['files']['crates/riley-runtime/src/llama/decode.rs']
    assert result['source_sha256']['native_decode_attention']==source_receipt['files']['kernels/src/decode_attention.cu']
    assert result['source_sha256']['test']==source_receipt['files']['crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs']
    contract=result['contract'];assert contract['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert contract['prompt_tokens']==2048 and contract['teacher_token_ids']==[304,279] and contract['logical_lengths']==[2049,2050]
    assert contract['dtype']=='BF16' and contract['comparison']=='byte-exact; no tolerance'
    m_raw=(hf_directory/'qwen3b-p2048-cache-on-layer-stage.json').read_bytes();h_raw=(hf_directory/'qwen3b-p2048-cache-on-layer-stage.safetensors').read_bytes()
    assert sha(m_raw)==result['hf_stage_artifact']['manifest_sha256']=='b2fc7301636ba618b904b554cd914ac35dd88c2f83e7c7736571353a4fd1fa4d'
    assert sha(h_raw)==result['hf_stage_artifact']['sidecar_sha256']=='5c12fc6f34cddfc73ef0ebc33b4903550ef7f632c722e53e158f1c715ffd70c9'
    m=json.loads(m_raw)
    for entry in m['provenance']['source_repository']['sources'].values():
        if entry['path'].startswith('tools/'):
            assert source_receipt['files'][entry['path']]==entry['sha256']
    header,data=safetensors(h_raw);hf_bytes={};hf_spans=[]
    assert set(header)-{'__metadata__'}=={v['key'] for v in m['tensors'].values()}
    for name,t in m['tensors'].items():
        tensor=header[t['key']];a,b=tensor['data_offsets'];assert tensor['dtype']=='BF16' and tensor['shape']==t['shape']
        assert 0<=a<b<=len(data) and b-a==t['bf16_le_bytes']==2*math.prod(t['shape'])
        hf_bytes[name]=bytes(data[a:b]);assert sha(hf_bytes[name])==t['bf16_le_sha256'];hf_spans.append((a,b))
    cursor=0
    for a,b in sorted(hf_spans):assert a==cursor;cursor=b
    assert cursor==len(data) and len(hf_bytes)==156
    raw_path=directory/Path(result['candidate_raw_sidecar']['path']).name;raw=raw_path.read_bytes()
    assert sha(raw)==result['candidate_raw_sidecar']['sha256'] and len(raw)==result['candidate_raw_sidecar']['bytes']
    stages=result['stages'];expected={name for name in hf_bytes if name.startswith(('decode_step_1.','decode_step_2.'))}
    assert set(stages)==expected and len(stages)==104
    spans=[];exact=[];mismatches=[]
    for name,s in stages.items():
        a,b=s['raw_data_offsets'];tensor=m['tensors'][name];assert s['shape']==tensor['shape']
        assert 0<=a<b<=len(raw) and b-a==len(hf_bytes[name])==s['element_count']*2
        actual=raw[a:b];expected_bytes=hf_bytes[name]
        assert sha(actual)==s['riley_bf16_le_sha256'] and sha(expected_bytes)==s['hf_bf16_le_sha256']
        unequal=sum(x!=y for x,y in zip(struct.iter_unpack('<H',actual),struct.iter_unpack('<H',expected_bytes)))
        assert unequal==s['unequal_element_count'] and s['bf16_exact'] is (unequal==0)
        if unequal: mismatches.append({'stage':name,'offset':a,'unequal_elements':unequal})
        else:exact.append(name)
        spans.append((a,b))
    cursor=0
    for a,b in sorted(spans):assert a==cursor;cursor=b
    assert cursor==len(raw)
    first=min(mismatches,key=lambda x:x['offset'])['stage'] if mismatches else None
    assert result['summary']=={'stage_count':104,'bf16_exact_stage_count':len(exact),'first_non_exact_stage':first}
    teacher_raw=teacher_path.read_bytes();assert sha(teacher_raw)==contract['teacher_sidecar_sha256']=='d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
    th,td=safetensors(teacher_raw);assert th['teacher_forced/logits']['shape']==[128,151936] and th['teacher_forced/logits']['dtype']=='BF16'
    assert th['teacher_forced/logits']['data_offsets']==[0,len(td)] and len(td)==128*303872
    matches=[]
    for step in [1,2]:
        name=f'decode_step_{step}.last_logits';a,b=stages[name]['raw_data_offsets']
        golden=bytes(td[step*303872:(step+1)*303872]);assert hf_bytes[name]==golden
        matches.append(raw[a:b]==golden)
    gate=result['quality_gate'];assert gate['teacher_rows1_2_exact']==matches
    qualified=len(exact)==104 and all(matches)
    assert gate['M1_M2_bf16_exact'] is qualified and gate['serving_selector_eligible'] is False and gate['full128_generation_eligible'] is False
    assert result['performance_claim_eligible'] is False and result['serving_performance']=='미실행'
    assert all(result['repeat_execution'].values()) and all(result['invalid_position_guards'].values())
    if qualified:
        assert native['exit']==0 and b'test result: ok. 1 passed; 0 failed; 0 ignored' in log
        assert load(directory/'completion.json')['failure'] is None
    return {'independent_HF_raw_tensors':156,'independent_native_raw_tensors':104,'native_exact_stages':len(exact),
        'first_non_exact_stage':first,'mismatches':mismatches,'teacher_rows1_2_exact':matches,'M1_M2_qualified':qualified,
        'source_commit':result['source_commit'],'native_binary_sha256':launch['binary_sha256'],
        'scope':'native BF16 bytes compared independently; owner-repeat/position guards/zero-allocation close bound to source and actual native log',
        'full128_generation':'unverified','serving_performance':'미실행','adopted':False,'goal_achieved':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--hf-directory',required=True,type=Path);p.add_argument('--teacher-sidecar',required=True,type=Path);p.add_argument('--source-receipt',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=verify(a.directory,a.hf_directory,a.teacher_sidecar,load(a.source_receipt));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
