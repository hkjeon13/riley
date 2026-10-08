"""Independently bind immutable HF bytes to source-bound native M1 claims."""
import hashlib,json,math,struct
from pathlib import Path
from verify_qwen_kv import member

ROOT=Path(__file__).resolve().parent
D=ROOT/'verified-stages-attempt02'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def verify():
    raw_manifest=(D/'hf-original/qwen3b-p2048-cache-on-layer-stage.json').read_bytes()
    raw=(D/'hf-original/qwen3b-p2048-cache-on-layer-stage.safetensors').read_bytes()
    assert sha(raw_manifest)=='b2fc7301636ba618b904b554cd914ac35dd88c2f83e7c7736571353a4fd1fa4d'
    assert sha(raw)=='5c12fc6f34cddfc73ef0ebc33b4903550ef7f632c722e53e158f1c715ffd70c9'
    m=json.loads(raw_manifest);r=read(D/'qwen-m1-regular-full52-attempt02/qwen3b-p2048-cache-on-m1-full52-rust-result.json')
    assert m['contract']['model_revision']==r['contract']['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert m['contract']['input']['teacher_decode_token_ids']==r['contract']['teacher_decode_token_ids']==[304,279]
    assert m['contract']['input']['prefill_prompt_token_count']==r['contract']['prefill_input_token_count']==2048
    assert r['hf_stage_artifact']['manifest_sha256']==sha(raw_manifest)
    assert r['hf_stage_artifact']['sidecar_sha256']==m['sidecar']['sha256']==sha(raw)
    n=struct.unpack('<Q',raw[:8])[0];assert 0<n<len(raw)-8
    header=json.loads(raw[8:8+n]);data=memoryview(raw)[8+n:];spans=[];hashes={}
    assert set(header)-{'__metadata__'}=={t['key'] for t in m['tensors'].values()}
    assert len(m['tensors'])==m['sidecar']['tensor_count']==156
    for name,t in m['tensors'].items():
        h=header[t['key']];a,b=h['data_offsets']
        assert h['dtype']=='BF16' and h['shape']==t['shape']
        assert 0<=a<b<=len(data) and b-a==math.prod(t['shape'])*2==t['bf16_le_bytes']
        hashes[name]=sha(data[a:b]);assert hashes[name]==t['bf16_le_sha256'];spans.append((a,b))
    cursor=0
    for a,b in sorted(spans):assert a==cursor;cursor=b
    assert cursor==len(data)
    stages=r['candidate']['stages']
    expected={name.removeprefix('decode_step_1.') for name in hashes if name.startswith('decode_step_1.')}
    assert set(stages)==expected and len(expected)==52
    for name,s in stages.items():
        t=m['tensors']['decode_step_1.'+name]
        assert s['hf_bf16_le_sha256']==s['riley_bf16_le_sha256']==hashes['decode_step_1.'+name]
        assert s['element_count']==math.prod(t['shape']) and s['bf16_exact'] is True
        assert s['unequal_element_count']==0 and s['max_abs_bf16_as_f32']==s['mean_abs_bf16_as_f32']==0
    assert r['candidate']['summary']=={'bf16_exact_stage_count':52,'first_non_exact_stage':None,'stage_count':52}
    repeat=r['candidate']['repeat_execution']
    assert all(repeat[k] is True for k in ['all_m1_trace_tensors_bf16_identical','reused_prepared_owner','reused_trace_storage'])
    gate=r['quality_gate'];assert gate['required_m1_stage_count']==gate['candidate_exact_m1_stage_count']==52
    assert all(gate[k] is True for k in ['cache_on_full_forward_bf16_exact','cache_on_m1_decode_bf16_exact','corrected_cache_on_eligible'])
    assert gate['serving_selector_eligible'] is False and gate['performance_claim_eligible'] is False and r['performance_claim_eligible'] is False
    teacher=member(ROOT/'qwen-m1-layer14-qkav-attempt01-with-teacher-row1.tar.gz','/teacher-row1.bf16')
    assert len(teacher)==303872 and sha(teacher)==hashes['decode_step_1.last_logits']=='d4bb3effb2125dde0f10ccd5d8e9e2d97b56dd01cdf2d1179d02afbe5a138b05'
    git=read(D/'qwen-m1-regular-full52-attempt02/source-git-receipt.json')
    launch=read(D/'qwen-m1-regular-full52-attempt02/native-launch.json')
    assert git['revision']==launch['source_revision']=='4da970a7153ff615cc9f6b0e679d3bd0d7ce2064'
    assert git['strict_test_source_sha256']=='c5710039103d0a383b0b24ad219c22a72c6e7a9b9bd1b187a8bcee95cf4abbf0'
    assert sha((ROOT/'qwen-full52-reviewed-attempt02.rs').read_bytes())==git['strict_test_source_sha256']
    assert sha((ROOT/'qwen-full52-reviewed-attempt02.pack').read_bytes())==git['pack_sha256']
    differences={x['path']:x for x in r['source_compatibility']['current_source_differences']}
    assert differences['crates/riley-runtime/src/llama/decode.rs']['candidate_sha256']=='7fd5764896d964bd478e7df4c08a29cef14fbd9f8ee6587bdc79912459d0a609'
    assert differences['crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs']['candidate_sha256']==git['strict_test_source_sha256']
    native=read(D/'qwen-m1-regular-full52-attempt02/native-process.json')
    log=(D/'qwen-m1-regular-full52-attempt02/native.log').read_bytes()
    assert native['exit']==0 and native['log_sha256']==sha(log) and native['result']==r
    assert b'test result: ok. 1 passed; 0 failed; 0 ignored' in log
    assert read(D/'qwen-m1-regular-full52-attempt02/completion.json')['failure'] is None
    return {'independent_HF_raw_tensor_count':156,'native_M1_exact_stages':52,'all_raw_HF_hashes_bound':True,
        'teacher_row1_sha256':sha(teacher),'source_commit':git['revision'],'quality_gate':gate,'repeat_execution':repeat,
        'scope':'Immutable HF bytes independently hashed; native candidate byte digests and repeat claims bound to actual source/test log. No independent native raw tensor dump.',
        'M2_full128_generation':'unverified','serving_performance':'미실행','adopted':False,'goal_achieved':False}
if __name__=='__main__':
    result=verify();(ROOT/'qwen-full52-attempt02-independent-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
