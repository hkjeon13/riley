"""Hash all HF M1 stage bytes and reconcile native comparison without tolerances."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from verify_qwen_kv import member,decode,require

def verify(archive):
    manifest_raw=member(archive,'/qwen3b-p2048-cache-on-m1-layer14-detail.json')
    m=decode(manifest_raw);raw=member(archive,'/qwen3b-p2048-cache-on-m1-layer14-detail.safetensors')
    r=decode(member(archive,'/qwen3b-p2048-cache-on-m1-layer14-qkav-rust-result.json'))
    require(m['contract']['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','wrong model')
    require(m['trace_profile']['detailed_layer_index']==14,'wrong layer')
    require(m['contract']['input']['teacher_decode_token_ids']==[304,279],'wrong teacher tokens')
    require(m['contract']['input']['prefill_prompt_token_count']==2048,'wrong prefill count')
    require(len(m['tensors'])==m['sidecar']['tensor_count']==17,'wrong stage count')
    require(hashlib.sha256(raw).hexdigest()==m['sidecar']['sha256']==r['hf_stage_artifact']['sidecar_sha256'],'sidecar mismatch')
    require(hashlib.sha256(manifest_raw).hexdigest()==r['hf_stage_artifact']['manifest_sha256'],'native manifest mismatch')
    size=struct.unpack('<Q',raw[:8])[0];require(0<size<len(raw)-8,'invalid header')
    h=decode(raw[8:8+size]);data=memoryview(raw)[8+size:];spans=[];hashes={};exact=[]
    require(set(h)-{'__metadata__'}=={v['key'] for v in m['tensors'].values()},'inventory differs')
    stages=r['candidate']['stages'];require(set(stages)==set(m['tensors']),'native inventory differs')
    for name,t in m['tensors'].items():
        tensor=h[t['key']];a,b=tensor['data_offsets'];elements=math.prod(t['shape'])
        require(tensor['dtype']=='BF16' and tensor['shape']==t['shape'],'layout differs')
        require(0<=a<b<=len(data) and b-a==elements*2==t['bf16_le_bytes'],'extent differs')
        sha=hashlib.sha256(data[a:b]).hexdigest();hashes[name]=sha;spans.append((a,b))
        require(sha==t['bf16_le_sha256']==stages[name]['hf_bf16_le_sha256'],'raw/native HF digest differs')
        require(stages[name]['element_count']==elements,'native element count differs')
        match=sha==stages[name]['riley_bf16_le_sha256']
        require(stages[name]['bf16_exact'] is match and ((stages[name]['unequal_element_count']==0) is match),'forged equality flag')
        if match:exact.append(name)
    cursor=0
    for a,b in sorted(spans):require(a==cursor,'overlap/hole');cursor=b
    require(cursor==len(data),'unbound bytes')
    require(r['candidate']['summary']['bf16_exact_stage_count']==len(exact),'native summary differs')
    order=['input_norm','q_proj','k_proj','v_proj','q_rope','k_rope','attention_scores',
           'attention_probabilities','attention_context','after_attention_residual','post_attention_norm',
           'gate_proj','up_proj','gated','down_proj','output']
    causal_order=[f'layer14.{name}.last' for name in order]+['last_logits']
    require(set(causal_order)==set(stages),'unexpected causal-stage inventory')
    first=next((name for name in causal_order if name not in exact),None)
    require(r['candidate']['summary']['first_non_exact_stage']==first,'native first divergence differs')
    require(r['candidate']['repeat_execution']['all_m1_layer_detail_tensors_bf16_identical'] is True,'owner repeat failed')
    require(r['source_compatibility']=={'current_source_differences':[],'mode':'exact-trace-source'},'source mismatch')
    require(hashes['last_logits']==m['contract']['source_logit_bindings']['decode_step_1']['bf16_le_sha256'],'M1 logit binding mismatch')
    teacher=member(archive,'/teacher-row1.bf16');receipt=decode(member(archive,'/teacher-row1-receipt.json'))
    require(receipt['source_sha256']=='d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39' and receipt['row']==1,'wrong immutable teacher row')
    require(len(teacher)==303872 and hashlib.sha256(teacher).hexdigest()==receipt['row_sha256']==hashes['last_logits'],'raw immutable teacher binding differs')
    require(r['quality_gate']['cache_on_m1_decode_bf16_exact'] is False and r['performance_claim_eligible'] is False,'incorrect admission')
    return {'HF_raw_stages_verified':17,'native_exact_stages':len(exact),'exact_stages':exact,
        'first_non_exact_stage':r['candidate']['summary']['first_non_exact_stage'],
        'probabilities':stages['layer14.attention_probabilities.last'],'layer_output':stages['layer14.output.last'],
        'repeat_execution':r['candidate']['repeat_execution'],'quality_gate':r['quality_gate'],
        'scope':'HF raw independently hashed; native equality and owner-repeat supplied by source-bound diagnostic; no native raw byte dump',
        'serving_performance':'미실행','goal_achieved':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=verify(a.archive);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['HF_raw_stages_verified','native_exact_stages','first_non_exact_stage']}))
