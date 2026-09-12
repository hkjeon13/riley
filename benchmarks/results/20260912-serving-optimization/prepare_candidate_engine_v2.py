import hashlib,json
from pathlib import Path
r=Path('/tmp/riley-opt-260912')
b=json.loads((r/'candidate-binding.json').read_text());b['source']['correctness_gate_id']='g04-vllm-smol-p128-v1'
bp=r/'candidate-engine-binding.json'
with bp.open('x') as f:json.dump(b,f,indent=2);f.write('\n')
p=json.loads((r/'candidate-engine-plan.json').read_text());argv=p['engine_lanes']['riley']['argv'];argv[argv.index('--correctness-gate-id')+1]='g04-vllm-smol-p128-v1';p['immutable_files'][str(bp)]=hashlib.sha256(bp.read_bytes()).hexdigest();p['binding_note']='Gate ID identifies unchanged arithmetic contract required by closed v1 schema; new implementation/source/binary/correctness-report remain separately bound.'
with (r/'candidate-engine-v2-plan.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
