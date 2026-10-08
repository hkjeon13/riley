"""Independently hash HF sidecar bytes and reconcile native all-layer KV receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile

def require(ok, message):
    if not ok:
        raise ValueError(message)

def unique(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON field: '+key)
        result[key] = value
    return result

def decode(raw):
    return json.loads(raw, object_pairs_hook=unique)

def member(archive, suffix):
    with tarfile.open(archive) as tar:
        matches = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(suffix)]
        require(len(matches) == 1, 'ambiguous or missing archive member: '+suffix)
        return tar.extractfile(matches[0]).read()

def verify(hf_archive, native_archive):
    manifest_raw = member(hf_archive, '/qwen3b-p2048-cache-on-prefill-kv.json')
    manifest = decode(manifest_raw)
    sidecar = member(hf_archive, '/qwen3b-p2048-cache-on-prefill-kv.safetensors')
    require(manifest['schema_version'].endswith('kv-trace.v2'), 'wrong schema')
    require(manifest['contract']['model_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'wrong model')
    require(manifest['trace_profile']['selected_layer_indices'] == list(range(36)), 'incomplete layer selection')
    require(manifest['trace_profile']['tensor_count'] == 72, 'wrong inventory size')
    require(hashlib.sha256(sidecar).hexdigest() == manifest['sidecar']['sha256'], 'sidecar digest mismatch')
    size = struct.unpack('<Q', sidecar[:8])[0]
    require(0 < size < len(sidecar)-8, 'invalid sidecar header')
    header = decode(sidecar[8:8+size])
    data = memoryview(sidecar)[8+size:]
    names = [f'prefill.layer{i}.{kind}' for i in range(36) for kind in ('key', 'value')]
    require(set(manifest['tensors']) == set(names), 'incomplete manifest inventory')
    require(set(header)-{'__metadata__'} == {manifest['tensors'][n]['key'] for n in names}, 'sidecar inventory mismatch')
    intervals = []
    hashes = {}
    for name in names:
        metadata = manifest['tensors'][name]
        tensor = header[metadata['key']]
        require(tensor['dtype'] == 'BF16' and tensor['shape'] == metadata['shape'] == [2, 2048, 128], 'tensor layout mismatch')
        start, end = tensor['data_offsets']
        require(0 <= start < end <= len(data) and end-start == metadata['bf16_le_bytes'] == 1048576, 'invalid tensor extent')
        hashes[name] = hashlib.sha256(data[start:end]).hexdigest()
        require(hashes[name] == metadata['bf16_le_sha256'], 'tensor digest mismatch')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        require(start == cursor, 'overlap or hole in sidecar')
        cursor = end
    require(cursor == len(data), 'unbound sidecar bytes')
    result = decode(member(native_archive, '/qwen3b-p2048-cache-on-prefill-kv-rust-result.json'))
    require(result['hf_stage_artifact']['manifest_sha256'] == hashlib.sha256(manifest_raw).hexdigest(), 'native uses another manifest')
    require(result['hf_stage_artifact']['sidecar_sha256'] == manifest['sidecar']['sha256'], 'native uses another sidecar')
    candidate = result['candidate']
    require(candidate['selected_layer_indices'] == list(range(36)), 'incomplete native layer selection')
    require(set(candidate['tensors']) == set(names), 'incomplete native inventory')
    unequal = []
    for name in names:
        observed = candidate['tensors'][name]
        require(observed['hf_bf16_le_sha256'] == hashes[name], 'native HF digest differs')
        exact = observed['riley_bf16_le_sha256'] == hashes[name]
        require(observed['bf16_exact'] is exact, 'native equality claim differs from digests')
        require((observed['unequal_element_count'] == 0) is exact, 'native unequal count contradicts equality')
        if not exact:
            unequal.append(name)
    summary = candidate['summary']
    require(summary['tensor_count'] == 72 and summary['bf16_exact_tensor_count'] == 72-len(unequal), 'native summary mismatch')
    require(summary['first_non_exact_tensor'] == (unequal[0] if unequal else None), 'native first divergence mismatch')
    logits = candidate.get('prefill_last_logits')
    if logits is not None:
        row = member(native_archive, '/teacher-prefill-row0.bf16')
        receipt = decode(member(native_archive, '/teacher-prefill-row0-receipt.json'))
        require(receipt['source_sha256'] == 'd1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39' and receipt['row'] == 0, 'wrong immutable teacher source')
        require(len(row) == 151936*2 and logits['element_count'] == 151936, 'wrong prefill logit size')
        teacher_hash = hashlib.sha256(row).hexdigest()
        require(teacher_hash == receipt['row_sha256'] == manifest['contract']['source_logit_binding']['bf16_le_sha256'] == logits['hf_bf16_le_sha256'], 'teacher row hash binding differs')
        exact = logits['riley_bf16_le_sha256'] == teacher_hash
        require(logits['bf16_exact'] is exact and result['quality_gate']['prefill_last_token_bf16_exact'] is exact, 'forged native prefill logit flag')
        require(candidate['repeat_execution']['prefill_last_logits_bf16_identical'] is True, 'prefill logit reuse not verified')
    return {'HF_raw_hashes_verified': 72, 'native_exact_tensor_count': 72-len(unequal),
            'first_non_exact_tensor': unequal[0] if unequal else None,
            'same_owner_repeat': candidate['repeat_execution'], 'native_prefill_logits': logits,
            'scope': 'HF raw bytes independently hashed; Riley hashes and owner reuse supplied by source-bound native test',
            'serving_performance': '미실행', 'goal_achieved': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hf_archive', type=Path)
    parser.add_argument('native_archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.hf_archive, args.native_archive)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False))
