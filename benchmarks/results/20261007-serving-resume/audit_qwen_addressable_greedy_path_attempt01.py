#!/usr/bin/env python3
"""CPU-only audit of frozen teacher logits; not native generation evidence."""
import array
import hashlib
import json
import math
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parent
inputs = json.loads((ROOT / 'qwen-original-teacher-greedy-path-inputs-attempt01.json').read_text())
policy = json.loads((ROOT / 'qwen-original-EOS-policy-inputs-attempt01.json').read_text())
raw = (ROOT / 'qwen-full128-independent-collection-attempt01/teacher-cache-on-logits.safetensors').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'd1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
header_length = struct.unpack('<Q', raw[:8])[0]
header = json.loads(raw[8:8 + header_length])
tensors = {k: v for k, v in header.items() if k != '__metadata__'}
assert len(tensors) == 1, tensors.keys()
tensor = next(iter(tensors.values()))
assert tensor['dtype'] == 'BF16' and tensor['shape'] == [128, 151936]
assert tensor['data_offsets'] == [0, 128 * 151936 * 2]
payload = raw[8 + header_length:]
assert len(payload) == 128 * 151936 * 2
words = array.array('H', payload)
if sys.byteorder == 'big':
    words.byteswap()
assert words.itemsize == 2
contract = inputs['contract']['teacher_forced_generation']
assert contract['addressable_token_count'] == 151665
assert contract['fixed_output_token_count'] == 128
assert contract['teacher_token_derivation'] == 'cache-off-addressable-greedy-argmax'
eos_ids = policy['generation_config.json']['eos_token_id']
rows = []
for row in range(128):
    best_value = -math.inf
    best_token = None
    ties = []
    for token in range(151936):
        bits = words[row * 151936 + token]
        assert bits & 0x7F80 != 0x7F80, (row, token, bits)
        if token >= 151665:
            continue
        value = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
        if value > best_value:
            best_value, best_token, ties = value, token, [token]
        elif value == best_value:
            ties.append(token)
    teacher = inputs['teacher_token_ids'][row]
    rows.append({'row': row, 'selected_token_id': best_token, 'teacher_token_id': teacher,
                 'exact_token_match': best_token == teacher, 'maximum_logit': best_value,
                 'tie_token_ids': ties, 'selected_is_configured_eos': best_token in eos_ids})
receipt = {'schema_version': 'riley.qwen-addressable-greedy-path-audit.v1',
           'oracle_sha256': inputs['oracle_sha256'], 'teacher_logits_sha256': hashlib.sha256(raw).hexdigest(),
           'contract': {'addressable_token_count': 151665, 'vocabulary_size': 151936,
                        'tie_rule': 'numeric maximum; lowest token ID; signed zeros equal',
                        'non_finite_policy': 'reject in full vocabulary',
                        'output_token_count': 128, 'original_oracle_eos_policy': 'fixed 128; no early stop'},
           'model_eos_policy_source': policy, 'rows': rows,
           'teacher_greedy_matches': sum(r['exact_token_match'] for r in rows),
           'eos_selected_rows': [r['row'] for r in rows if r['selected_is_configured_eos']],
           'scope': 'frozen HF oracle CPU audit only; native free-running generation unexecuted',
           'serving_performance': '미실행', 'goal_achieved': False}
with (ROOT / 'qwen-addressable-greedy-path-audit-attempt01.json').open('x') as f:
    json.dump(receipt, f, indent=2, ensure_ascii=False)
print(json.dumps({k: receipt[k] for k in ['teacher_greedy_matches', 'eos_selected_rows', 'scope']}, ensure_ascii=False))
assert receipt['teacher_greedy_matches'] == 128
