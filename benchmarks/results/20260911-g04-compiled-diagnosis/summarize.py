"""Summarize only recorded correctness observations; never waive a gate."""
import ast
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
old=Path('/tmp/riley-g04-readiness-260911')
followup=Path('/tmp/riley-g04-followup-260911')
def read(path): return json.loads(path.read_text())
def first(a,b):
    assert len(a)==len(b)==32
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None)
base=read(old/'parity.json')
default=read(root/'default.json')['requests'][0]['tokens']
assert default==base['vllm_output_tokens']
eager=read(followup/'vllm-eager-probe.json')['tokens']
modes={}
for name in ['default','no-graph','custom-ops','silu','rope','norm']:
    data=read(root/(name+'.json'))
    assert len(data['requests'])==2
    a,b=[r['tokens'] for r in data['requests']]
    assert a==b
    modes[name]={'tokens':a,'logprobs_invariant':True,
                 'first_mismatch_vs_default':first(a,default),
                 'matches_eager':a==eager}
assert modes['no-graph']['tokens']==default
assert modes['rope']['tokens']==default
assert modes['norm']['tokens']==default
assert modes['silu']['tokens']==eager
assert modes['custom-ops']['tokens']==eager
variants={'canonical':base['riley_output_tokens'],
          'norm_only':read(followup/'diagnosis.json')['diagnostic_norm_only_tokens']}
for name,log in [('norm_and_swiglu','riley-norm-swiglu.log'),('swiglu_only','riley-swiglu-only.log')]:
    text=(root/log).read_text()
    assert 'test result: ok. 1 passed; 0 failed;' in text
    variants[name]=ast.literal_eval(re.search(r'output_tokens=(\[[^\n]+\])',text)[1])
tensor_norm=read(root/'norm-tensor.json')
tensor_swiglu=read(root/'swiglu-tensor.json')
assert tensor_norm['compiled_vs_native']['unequal']==0
assert tensor_swiglu['compiled_vs_single_round']['unequal']==0
assert tensor_swiglu['native_vs_eager']['unequal']==0
result={'performance_trials':0,'competitive_qualification_passed':False,
        'production_candidate_modified':False,'vllm_correctness_requests':12,
        'mode_ablation':modes,'riley_diagnostic_variants':{
            name:{'tokens':tokens,'first_mismatch_vs_default':first(tokens,default)}
            for name,tokens in variants.items()},
        'norm_tensor':tensor_norm,'swiglu_tensor':tensor_swiglu,
        'conclusions':[
            'CUDA graph replay and requesting logprobs do not explain the observed output difference.',
            'Selecting only native SiluAndMul reproduces the eager token sequence with compilation still enabled.',
            'On sampled actual layer0 projections, compiled SwiGLU equals single-round FP32 composition; native equals BF16-staged eager.',
            'On sampled embedding inputs, compiled RMSNorm equals native RMSNorm and differs from HF eager.',
            'None of the isolated Riley rounding variants matches default vLLM for all 32 tokens.'
        ],
        'limitations':[
            'One fixed prompt; causal flag ablation is not complete cross-engine layer/operator tracing.',
            'The native diagnostic changes the SiLU/product composition; it is not a qualified standalone SiLU implementation.',
            'Graph/eager equality inside an altered diagnostic source does not preserve the canonical HF contract.',
            'GPU exclusivity remains blocked by unrelated Blender sessions.'
        ]}
result['evidence']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.iterdir()) if p.is_file() and p.name!='summary.json'}
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'default_reproduced':True,'silu_ablation_reproduced_eager':True,
                  'performance_trials':0,'competitive_qualification_passed':False}))
