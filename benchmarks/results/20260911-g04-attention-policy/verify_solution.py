"""Validate the bounded numerical solution without running performance trials."""
from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent
s=r.parent/'20260911-g04-serving-trace'
inv=json.loads((s/'invariance.json').read_text())
assert inv['before']==inv['observed']==inv['after']
assert inv['matches_default_graph_reference']
t=json.loads((s/'serving-tensors.json').read_text())
assert len(t['records'])==960 and t['restored']
assert all(value==32 for value in t['calls'].values())
m=json.loads((r/'mma-comparison-final.json').read_text())
assert len(m['records'])==960 and m['total_unequal']==0
assert all(row['unequal']==0 and row['max_abs']==0 for row in m['records'])
g=json.loads((r/'graph-replay.json').read_text())
assert g['captures']==1 and g['replays']==930 and g['all_exact']
assert g['invalid_lengths_rejected']==[0,161] and g['valid_replay_recovers']
assert all(row['replay_allocated_bytes_delta']==0 for row in g['records'])
q=json.loads((r/'random-comparison.json').read_text())
assert len(q['records'])==12
assert sum(row.get('unsupported_rejected',False) for row in q['records'])==2
assert all(row.get('unsupported_rejected') or row['unequal']==0 for row in q['records'])
p=json.loads((r/'independent-mma-profile-final.json').read_text())
assert p['exact'] and len(p['tokens'])==32 and p['tokens']==p['reference']==inv['before']
for d in [inv,m,g,q,p]: assert d['performance_trials']==0
result={'numerical_prototype_passed':True,'actual_serving_attention_cases':960,
        'dynamic_graph_replays':930,'full_generation_tokens_exact':32,
        'production_riley_candidate_integrated':False,'performance_trials':0}
(r/'solution-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
