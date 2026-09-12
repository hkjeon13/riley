"""Exact BF16 prefix comparison, independent of torch and token sampling."""
import hashlib
import json
import math
from pathlib import Path
import struct

root=Path(__file__).resolve().parent
cases=json.loads((root/'prefixes.json').read_text())
vllm=json.loads((root/'vllm-prefix.json').read_text())
assert vllm['model_outputs_unchanged'] and vllm['before_tokens']==vllm['after_tokens']
assert all(w['exposed_vs_opaque_qkv_exact'] for w in vllm['worker_results'])

def unpack(path):
    raw=path.read_bytes()
    assert len(raw)%2==0
    bits=struct.unpack('<'+'H'*(len(raw)//2),raw)
    values=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in bits]
    assert all(math.isfinite(v) for v in values)
    return raw,values

def compare(a,b,width):
    aa,x=unpack(a);bb,y=unpack(b)
    assert len(x)==len(y)
    different=[i for i,(u,v) in enumerate(zip(x,y)) if u!=v]
    err=[abs(u-v) for u,v in zip(x,y)]
    first=different[0] if different else None
    return {'elements':len(x),'unequal':len(different),'bytes_equal':aa==bb,
            'max_abs':max(err),'mean_abs':sum(err)/len(err),
            'first_difference':None if first is None else {
                'row':first//width,'column':first%width,'riley':x[first],'vllm':y[first]},
            'riley_sha256':hashlib.sha256(aa).hexdigest(),'vllm_sha256':hashlib.sha256(bb).hexdigest()}

points=[('embedding',576),('layer0.input_norm',576),('layer0.q_proj',576),
        ('layer0.k_proj',192),('layer0.v_proj',192)]
results={}
for case,ids in cases.items():
    result={'input_tokens':ids,'modes':{}}
    for mode in ['eager','compiled','eager_fp32','split','split_fp32']:
        comparisons={label:compare(root/'riley'/f'{case}-{label}.bf16',
            root/f'{case}-{mode}-{label}.bf16',width) for label,width in points}
        result['modes'][mode]={'points':comparisons,
            'first_different_point':next((label for label,_ in points
                                         if not comparisons[label]['bytes_equal']),None)}
    assert result['modes']['compiled']['points']['embedding']['bytes_equal']
    result['isolated_norm_variant_vs_compiled']={label:compare(
        root/'riley_norm_variant'/f'{case}-{label}.bf16',
        root/f'{case}-compiled-{label}.bf16',width) for label,width in points}
    results[case]=result
out={'performance_trials':0,'competitive_qualification_passed':False,
     'scope':'actual loaded-model isolated prefix replay vs Riley dense forward trace, before first attention',
     'not_proven':'This is not a tap of the full vLLM compiled serving graph, nor a decode layer-by-layer trace.',
     'generation_unchanged_by_diagnostic':True,'prefix_exposed_vs_opaque_qkv_exact':True,'cases':results}
(root/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({case:result['modes']['compiled']['first_different_point'] for case,result in results.items()}))
