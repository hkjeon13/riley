"""Verify retained wide-generation evidence; this is not a serving speed test."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
d = root / 'benchmarks/results/20260914-speculative-wide'
m = json.loads((d / 'manifest.json').read_text())
assert hashlib.sha256((d / 'evidence.tar.gz').read_bytes()).hexdigest() == m['archive_sha256']
with tarfile.open(d / 'evidence.tar.gz') as archive:
    files = {p.name: archive.extractfile(p).read() for p in archive.getmembers() if p.isfile()}
assert set(files) == set(m['files'])
for name, digest in m['files'].items():
    assert hashlib.sha256(files[name]).hexdigest() == digest, name
for name, digest in m['sources'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
for name, digest in m['local_evidence'].items():
    assert hashlib.sha256((d / name).read_bytes()).hexdigest() == digest, name
for version, requests, serial_calls, accepted in [(1, 12, 36, 144), (2, 32, 44, 581)]:
    base = f'speculative-wide-lifecycle-v{version}/'
    execution = json.loads(files[base + 'execution.json'])
    assert all(x['exit_code'] == 0 for x in execution)
    assert {x['name'] for x in execution} >= {'probe', 'memcheck'}
    assert b'ERROR SUMMARY: 0 errors' in files[base + 'memcheck.log']
    for run in ['probe', 'memcheck']:
        j = json.loads(files[base + run + '-generation.json'])
        assert len(j['serial']) == requests and all(len(x) == 32 for x in j['serial'])
        assert j['serial'] == j['speculative'] and j['differing_tokens'] == 0
        assert (j['serial_calls'], j['speculative_calls'], j['accepted_draft_tokens']) == (serial_calls, 33, accepted)
    assert json.loads(files[base + 'blender-restored.json'])['restored']
assert b'ERROR SUMMARY: 0 errors' in files['speculative-wide-lifecycle-v2/select-memcheck.log']
assert b'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in files['speculative-wide-lifecycle-v2/select-racecheck.log']
assert all(x['exit_code'] == 0 for x in json.loads(files['speculative-wide-build-v2/native-builds.json']))
assert '70 passed; 0 failed' in (d / 'scheduler-tests.log').read_text()
assert '1 passed; 0 failed' in (d / 'records-tests.log').read_text()
assert '0 failed' in (d / 'wire-tests.log').read_text()
assert 'corruptions_rejected=10' in (d / 'packet-probe.log').read_text()
print('PASS: exact tokens, accepted drafts, wide contracts, sanitizers and restoration; serving unmeasured')
