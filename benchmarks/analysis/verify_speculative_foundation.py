#!/usr/bin/env python3
"""Recheck the fixed native prefix-equivalence milestone; no serving claim."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[2]
directory = root / 'benchmarks/results/20260914-speculative-foundation'
archive = directory / 'evidence.tar.gz'
with tarfile.open(archive) as tar:
    members = tar.getmembers()
    assert len({m.name for m in members}) == len(members)
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    files = {m.name: tar.extractfile(m).read() for m in members}
life = 'speculative-prefix-lifecycle-v1/'
build = 'speculative-prefix-build-v1/'
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
assert {row['name']: row['exit_code'] for row in parsed(life + 'execution.json')} == {'probe': 0, 'memcheck': 0}
assert 'ERROR SUMMARY: 0 errors' in files[life + 'memcheck.log'].decode()
for name in ['probe', 'memcheck']:
    assert 'differing_bf16=0 allocation_zero=true fused_verification_implemented=false serving_measured=false' in files[life + name + '.log'].decode()
a = files[life + 'logits/serial.bf16']
b = files[life + 'logits/batched-prefix.bf16']
assert len(a) == len(b) == 7077888 and a == b
assert parsed(life + 'logits/cases.json') == json.loads((root / 'benchmarks/results/20260913-flashinfer-natural-screen/cases.json').read_text())
assert parsed(life + 'blender-restored.json')['restored'] is True
assert all(code == 200 for code in parsed(life + 'provenance.json')['viewers'].values())
report = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'regular_members': len(files),
          'current_source_hashes_verified': True, 'raw_logits_identical': True, 'logit_rows': 72,
          'logit_bytes_per_lane': len(a), 'memcheck_errors': 0, 'blender_restored': True,
          'fused_verification_implemented': False, 'serving_measured': False}
(directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
