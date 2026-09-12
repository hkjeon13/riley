"""Independently verify captured values and diagnostic invariants on CPU."""
import hashlib
import json
from pathlib import Path
import struct
root=Path(__file__).resolve().parent
d=json.loads((root/'comparison.json').read_text())
v=json.loads((root/'vllm-tail.json').read_text())
assert v['model_outputs_unchanged'] and v['before_tokens']==v['after_tokens']
assert d['original_reduced_precision_flag']==d['restored_reduced_precision_flag']
assert d['performance_trials']==0

def unpack(raw):
    return [struct.unpack('<f',struct.pack('<I',b[0]<<16))[0] for b in struct.iter_unpack('<H',raw)]

for key,record in d['records'].items():
    case,name=key.split('/')
    point=('post_attention_norm' if name.startswith('norm_') else
           'gate_proj' if name.startswith('gate_') else
           'up_proj' if name.startswith('up_') else
           'gated' if name.startswith('swiglu_') else
           'down_proj' if name.startswith('down_') else 'output')
    raw=(root/record['file']).read_bytes()
    ref=(root/'riley'/f'{case}-layer0.{point}.bf16').read_bytes()
    assert len(raw)==len(ref)==2*record['elements']
    assert hashlib.sha256(raw).hexdigest()==record['sha256']
    x,y=unpack(raw),unpack(ref)
    assert sum(a!=b for a,b in zip(x,y))==record['unequal'],key
    assert max(abs(a-b) for a,b in zip(x,y))==record['max_abs'],key
for case in ['m1','p128','common136']:
    compiled=(root/f'{case}-swiglu_compiled.bf16').read_bytes()
    assert compiled==(root/f'{case}-swiglu_fp32_formula.bf16').read_bytes()
    assert compiled==(root/f'{case}-swiglu_eager.bf16').read_bytes()
    assert (root/f'{case}-swiglu_staged.bf16').read_bytes()==(root/'riley'/f'{case}-layer0.gated.bf16').read_bytes()
print(json.dumps({'verified_comparisons':len(d['records']), 'generation_unchanged':True,
                  'performance_trials':0}))
