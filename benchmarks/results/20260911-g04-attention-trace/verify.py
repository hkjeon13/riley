"""CPU verification of raw GPU evidence; no performance sampling."""
import hashlib,json,struct,math
from pathlib import Path
root=Path(__file__).resolve().parent
v=json.loads((root/'vllm-attention.json').read_text())
assert v['model_outputs_unchanged'] and v['before_tokens']==v['after_tokens']
d=json.loads((root/'comparison.json').read_text())
def values(raw):
    return [struct.unpack('<f',struct.pack('<I',b[0]<<16))[0] for b in struct.iter_unpack('<H',raw)]
def compare(a,b):
    assert len(a)==len(b)
    x,y=values(a),values(b)
    assert all(math.isfinite(z) for z in x+y)
    return {'elements':len(x),'unequal':sum(u!=v for u,v in zip(x,y)),
            'max_abs':max(abs(u-v) for u,v in zip(x,y)),
            'sha256':hashlib.sha256(a).hexdigest(),'reference_sha256':hashlib.sha256(b).hexdigest()}
for key,record in d['records'].items():
    case,name=key.split('/')
    assert record==compare((root/f'{case}-{name}.bf16').read_bytes(),
                           (root/f'{case}-{name}.reference.bf16').read_bytes()),key
extra={}
for case in ['m1','p128','common136']:
    for axis in ['q','k']:
        reference=(root/'riley'/f'{case}-layer0.{axis}_rope.bf16').read_bytes()
        for mode,suffix in [('compiled','bf16'),('selected','reference.bf16')]:
            raw=(root/f'{case}-rope_compiled_{axis}_vs_selected.{suffix}').read_bytes()
            extra[f'{case}/rope_{axis}_{mode}_vs_riley']=compare(raw,reference)
    a=(root/f'{case}-fused_norm_compiled.bf16').read_bytes()
    b=(root/f'{case}-fused_norm_fp32_formula.bf16').read_bytes()
    assert a==b,case
(root/'direct-rope-comparison.json').write_text(json.dumps(extra,indent=2)+'\n')
print(json.dumps({'verified_comparisons':len(d['records'])+len(extra),
                  'compiled_fused_norm_matches_fp32_formula':True,'performance_trials':0}))
print(json.dumps({k:v['unequal'] for k,v in extra.items()}))
