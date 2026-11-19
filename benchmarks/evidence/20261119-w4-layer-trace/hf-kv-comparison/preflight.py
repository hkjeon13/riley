import numpy as np
from hf_kv_comparison import kv_stats, replay_cell
from replay_attention_mask import attention

rng = np.random.default_rng(0)
q, k, v = (rng.normal(size=shape) for shape in ((7, 128), (16, 128), (16, 128)))
alt = v + 0.5
base = attention(q, k, v)
changed = attention(q, k, alt)
result = replay_cell({'riley': q, 'hf': q}, {'riley': k, 'hf': k}, {'riley': v, 'hf': alt}, {'riley': base, 'hf': changed})
assert len(result) == 8 and result['q_riley_k_riley_v_riley']['riley']['rms'] == 0
assert result['q_hf_k_hf_v_hf']['hf']['rms'] == 0
assert abs(result['q_riley_k_riley_v_hf']['riley']['rms'] - 0.5) < 1e-12
assert kv_stats(v, v)['different_elements'] == 0
assert kv_stats(v, alt)['tokens_with_difference'] == 16
print('KV byte and attention substitution fixtures passed')
