#!/usr/bin/env python3
"""Recheck the fixed multi-position target-head milestone; no serving claim."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[2]
directory = root / 'benchmarks/results/20260914-verification-head'
archive = directory / 'evidence.tar.gz'
with tarfile.open(archive) as tar:
    members = tar.getmembers()
    assert len({m.name for m in members}) == len(members)
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    files = {m.name: tar.extractfile(m).read() for m in members}
life = 'verification-head-lifecycle-v2/'
build = 'verification-head-build-v2/'
def parsed(name):
    return json.loads(files[name])
manifest = parsed(life + 'archive-files.json')
assert set(manifest) == set(files) - {life + 'archive-files.json'}
for name, row in manifest.items():
    assert len(files[name]) == row['bytes'] and hashlib.sha256(files[name]).hexdigest() == row['sha256']
for name, digest in parsed(build + 'sources.json').items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
assert parsed(build + 'exit.json')['exit_code'] == 0
assert parsed(build + 'binary.json')['sha256'] == parsed(life + 'provenance.json')['binary_sha256']
assert {row['name']: row['exit_code'] for row in parsed(life + 'execution.json')} == {'select': 0, 'select-memcheck': 0, 'select-racecheck': 0, 'probe': 0, 'memcheck': 0}
assert 'ERROR SUMMARY: 0 errors' in files[life + 'memcheck.log'].decode()
assert 'ERROR SUMMARY: 0 errors' in files[life + 'select-memcheck.log'].decode()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in files[life + 'select-racecheck.log'].decode()
assert all(row['exit_code']==0 for row in parsed('verification-head-build-v1/select-builds.json'))
for name in ['probe', 'memcheck']:
    assert 'differing_bf16=0 allocation_zero=true speculative_scheduler_implemented=false serving_measured=false' in files[life + name + '.log'].decode()
a = files[life + 'logits/serial.bf16']
b = files[life + 'logits/multi.bf16']
assert len(a) == len(b) == 7077888 and a == b
assert parsed(life + 'logits/cases.json') == json.loads((root / 'benchmarks/results/20260913-flashinfer-natural-screen/cases.json').read_text())
assert parsed(life + 'blender-restored.json')['restored'] is True
assert all(code == 200 for code in parsed(life + 'provenance.json')['viewers'].values())
assert 'lane=multi model_iterations=10 multi_endpoint_calls=2' in files[life+'probe.log'].decode()
report = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'regular_members': len(files),
          'current_source_hashes_verified': True, 'raw_logits_identical': True, 'logit_rows': 72,
          'logit_bytes_per_lane': len(a), 'memcheck_errors': 0, 'blender_restored': True,
          'multi_position_head_verified': True, 'speculative_scheduler_implemented': False, 'serving_measured': False}
(directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
