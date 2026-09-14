#!/usr/bin/env python3
"""Verify the fixed context-split model milestone, preserving its failed gate."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from check_attention_natural_screen import adjudicate

root = Path(__file__).resolve().parents[2]
directory = root / 'benchmarks/results/20260914-attention-split-model'
manifest = json.loads((directory / 'evidence/manifest.json').read_text())
chunks = []
for part in manifest['parts']:
    assert Path(part['name']).name == part['name']
    data = (directory / 'evidence' / part['name']).read_bytes()
    assert len(data) == part['bytes'] and hashlib.sha256(data).hexdigest() == part['sha256']
    chunks.append(data)
archive = b''.join(chunks)
assert len(archive) == manifest['archive_bytes']
assert hashlib.sha256(archive).hexdigest() == manifest['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    members = tar.getmembers()
    assert len({m.name for m in members}) == len(members)
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    files = {m.name: tar.extractfile(m).read() for m in members}

def parsed(name):
    return json.loads(files[name])

completion = 'context-split-model-completion-v1/'
v2 = 'context-split-model-lifecycle-v2/'
build = 'context-split-model-build-v2/'
file_manifest = parsed(completion + 'archive-files.json')
assert set(file_manifest) == set(files) - {completion + 'archive-files.json'}
for name, receipt in file_manifest.items():
    assert len(files[name]) == receipt['bytes']
    assert hashlib.sha256(files[name]).hexdigest() == receipt['sha256']
for name, digest in parsed(build + 'sources.json').items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
assert parsed(build + 'exit.json')['exit_code'] == 0
assert all(x['exit_code'] == 0 for x in parsed(build + 'mixed-build.json'))
assert {x['name']: x['exit_code'] for x in parsed(v2 + 'execution.json')} == {
    'natural': 0, 'free': 101, 'memcheck': 0, 'free-memcheck': 101, 'metrics': 2}
assert all(x['exit_code'] == 0 for x in parsed(completion + 'execution.json'))
assert len(parsed(completion + 'execution.json')) == 4
assert parsed(completion + 'blender-restored.json')['restored'] is True
assert all(code == 200 for code in parsed(completion + 'provenance.json')['viewers'].values())
for prefix, name in [(v2, 'memcheck'), (v2, 'free-memcheck'), (completion, 'mixed-memcheck')]:
    assert 'ERROR SUMMARY: 0 errors' in files[prefix + name + '.log'].decode()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in files[completion + 'mixed-racecheck.log'].decode()
metrics = parsed(v2 + 'natural/metrics.json')
metric_check = adjudicate(metrics)
assert metric_check['relative_screen_passed'] is True
for lane in ['baseline', 'candidate']:
    data = files[v2 + f'natural/{lane}.bf16']
    assert len(data) == 8 * 32 * 49152 * 2
    assert hashlib.sha256(data).hexdigest() == metrics['logit_hashes'][lane]
assert parsed(v2 + 'natural/cases.json') == json.loads((root / 'benchmarks/results/20260913-flashinfer-natural-screen/cases.json').read_text())

def generation(name):
    text = files[name].decode()
    lanes = {}
    for lane in ['baseline', 'candidate']:
        line = re.findall(r'FREE_TOKENS ' + lane + r'=(.*)', text)
        assert len(line) == 1
        lanes[lane] = ast.literal_eval(line[0])
        assert len(lanes[lane]) == 32 and all(len(row) == 32 for row in lanes[lane])
        assert all(type(token) is int and 0 <= token < 49152 for row in lanes[lane] for token in row)
    return lanes
lanes = generation(v2 + 'free.log')
assert lanes == generation(v2 + 'free-memcheck.log')
differences = sum(a != b for xs, ys in zip(lanes['baseline'], lanes['candidate']) for a, b in zip(xs, ys))
sequence_differences = sum(xs != ys for xs, ys in zip(lanes['baseline'], lanes['candidate']))
assert differences == 168 and sequence_differences == 12
assert all(rows[i] == rows[i % 8] for rows in lanes.values() for i in range(8, 32))
assert 'differing_sequences=12 differing_tokens=168 baseline_invariance=true candidate_invariance=true' in files[v2 + 'free.log'].decode()
initial = generation('context-split-model-lifecycle-v1/free.log')
assert any(initial['candidate'][i] != initial['candidate'][i % 8] for i in range(8, 32))
result = {'archive_sha256': manifest['archive_sha256'], 'regular_members': len(files),
          'file_hashes_verified': True, 'current_source_hashes_verified': True,
          'relative_metric_gate': metric_check, 'aggregate': metrics['aggregate'],
          'generation': {'requests': 32, 'tokens': 1024, 'differing_sequences': sequence_differences,
                         'differing_tokens': differences, 'batch_invariance': True,
                         'strict_gate_passed': False, 'memcheck_outputs_identical': True},
          'initial_batch_invariance_failed': True, 'blender_restored': True,
          'general_quality_accepted': False, 'serving_measured': False}
(directory / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
