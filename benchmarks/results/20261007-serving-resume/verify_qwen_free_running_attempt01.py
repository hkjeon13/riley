#!/usr/bin/env python3
"""Independently replay actual native own-greedy raw rows after safe collection."""
import argparse
import hashlib
import json
import pathlib
import struct

parser = argparse.ArgumentParser()
parser.add_argument('collected_native_directory', type=pathlib.Path)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent
native = args.collected_native_directory.resolve()
source = json.loads((root / 'qwen-free-running-source-receipt-attempt01.json').read_text())
oracle_inputs = json.loads((root / 'qwen-original-teacher-greedy-path-inputs-attempt01.json').read_text())
teacher = root / 'qwen-full128-independent-collection-attempt01/teacher-cache-on-logits.safetensors'
teacher_bytes = teacher.read_bytes()
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(teacher_bytes) == 'd1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
header_size = struct.unpack('<Q', teacher_bytes[:8])[0]
header = json.loads(teacher_bytes[8:8 + header_size])
tensors = [v for k, v in header.items() if k != '__metadata__']
assert len(tensors) == 1 and tensors[0]['dtype'] == 'BF16'
assert tensors[0]['shape'] == [128, 151936]
assert tensors[0]['data_offsets'] == [0, 128 * 151936 * 2]
expected = teacher_bytes[8 + header_size:]
completion = json.loads((native / 'completion.json').read_text())
assert completion['failure'] is None, completion
process = json.loads((native / 'native-process.json').read_text())
assert process['exit'] == 0
assert sha((native / 'native.log').read_bytes()) == process['log_sha256']
launch = json.loads((native / 'native-launch.json').read_text())
assert launch['source_revision'] == source['source_commit']
assert launch['binary_sha256'] == process['binary_sha256_after']
assert launch['argv'][1:] == ['--ignored', '--exact', 'qwen3b_p2048_cache_on_free_running128_logits_quality_gate', '--nocapture']
policy = json.loads((native / 'selection-policy-process.json').read_text())
assert policy['exit'] == 0 and policy['binary_sha256'] == launch['binary_sha256']
assert sha((native / 'selection-policy.log').read_bytes()) == policy['log_sha256']
before = json.loads((native / 'source-git-receipt.json').read_text())
after = json.loads((native / 'source-integrity-after.json').read_text())
assert before['source_commit'] == after['source_commit'] == source['source_commit']
assert before['verified_source_files'] == len(source['files']) == 433
assert before['code_worktree_clean'] and after['code_worktree_clean']
assert after['files'] == source['files']
result_path = native / 'qwen3b-p2048-cache-on-free-running128-native-result.json'
result = json.loads(result_path.read_text())
assert result == process['result'] and result['source_commit'] == source['source_commit']
assert result['source_sha256']['test'] == source['files']['crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs']
assert result['source_sha256']['rust_decode'] == source['files']['crates/riley-runtime/src/llama/decode.rs']
assert result['source_sha256']['native_decode_attention'] == source['files']['kernels/src/decode_attention.cu']
assert result['sampling'] == {'addressable_token_count': 151665, 'method': 'greedy numeric argmax',
                             'tie_rule': 'lowest token ID; signed zeros equal',
                             'input_rule': 'own previous selected token; no teacher token inputs',
                             'non_finite_policy': 'reject in full vocabulary'}
row_bytes = 151936 * 2
actual = result_path.with_suffix('.bf16').read_bytes()
assert len(actual) == len(expected) == 128 * row_bytes
assert result['candidate_raw_sidecar']['sha256'] == sha(actual)
assert result['candidate_raw_sidecar']['bytes'] == len(actual)
rows = []
generated = []
for row in range(128):
    raw = actual[row * row_bytes:(row + 1) * row_bytes]
    golden = expected[row * row_bytes:(row + 1) * row_bytes]
    maximum = float('-inf')
    selected = None
    for token, (bits,) in enumerate(struct.iter_unpack('<H', raw)):
        assert bits & 0x7F80 != 0x7F80, (row, token, bits)
        if token >= 151665:
            continue
        value = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
        if value > maximum:
            maximum, selected = value, token
    generated.append(selected)
    record = {'row': row, 'bf16_exact': raw == golden,
              'actual_sha256': sha(raw), 'expected_sha256': sha(golden)}
    assert result['rows'][row] == record
    rows.append(record)
assert generated == result['generated_token_ids'] == oracle_inputs['teacher_token_ids']
assert result['generated_token_ids_sha256'] == sha(struct.pack('<128I', *generated))
assert result['generated_tokens_match_oracle']
assert actual == expected and result['first_non_exact_row'] is None
assert result['repeat_execution'] == {'same_owner': True, 'all128_rows_and_selected_tokens_identical': True}
assert result['invalid_position_guards'] == {'before_prefill': True, 'after_decode_step127': True}
assert result['quality_gate'] == {'free_running128_logits_exact': True, 'serving_selector_eligible': False,
                                  'free_running_generation_verified': True}
assert result['cleanup'] == 'owner closed; CUDA allocation accounting zero; stream/context closed'
eos_rows = [row for row, token in enumerate(generated) if token in [151645, 151643]]
assert result['eos']['selected_rows'] == eos_rows == []
assert result['eos']['token_ids'] == [151645, 151643]
assert result['serving_performance'] == '미실행' and not result['goal_achieved']
receipt = {'schema_version': 'riley.qwen-own-greedy-independent-proof.v1', 'passed': True,
           'source_commit': source['source_commit'], 'binary_sha256': launch['binary_sha256'],
           'source_blob_count': 433, 'actual_raw_sha256': sha(actual),
           'actual_exact_logit_rows': sum(row['bf16_exact'] for row in rows),
           'actual_exact_selected_tokens': len(generated), 'generated_token_ids': generated,
           'repeat_and_lifecycle_scope': 'actual pinned native test checks both executions; raw replay of first execution',
           'eos_selected_rows': eos_rows,
           'qualification_scope': 'fixed128 diagnostic owner; production selector, HTTP cancellation/re-request unverified',
           'serving_performance': '미실행', 'goal_achieved': False}
with args.output.open('x') as out:
    json.dump(receipt, out, indent=2, ensure_ascii=False)
print(json.dumps(receipt, ensure_ascii=False))
