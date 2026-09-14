"""Check shared-prefix GPU evidence and provenance, not serving performance."""
import hashlib
import json
import tarfile
from pathlib import Path
root = Path(__file__).resolve().parents[2]
d = root / 'benchmarks/results/20260914-speculative-shared-wide'
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
b = 'speculative-shared-wide-lifecycle-v1/'
assert json.loads(files['speculative-shared-wide-build-v1/exit.json'])['exit_code'] == 101
assert json.loads(files['speculative-shared-wide-build-v2/exit.json'])['exit_code'] == 0
assert {x['name'] for x in json.loads(files[b+'execution.json'])} == {'probe', 'memcheck'}
assert all(x['exit_code'] == 0 for x in json.loads(files[b+'execution.json']))
for run in ['probe', 'memcheck']:
    j = json.loads(files[b+run+'-generation.json'])
    assert len(j['serial']) == 32 and all(len(x) == 32 for x in j['serial'])
    assert j['serial'] == j['speculative'] and j['differing_tokens'] == 0
    assert (j['serial_calls'],j['speculative_calls'],j['accepted_draft_tokens']) == (39,33,629)
    assert files[b+run+'.log'].count(b'SPECULATIVE_PREFIX hits=52 reused_tokens=6640') == 2
assert b'ERROR SUMMARY: 0 errors' in files[b+'memcheck.log']
assert json.loads(files[b+'blender-restored.json'])['restored']
assert '71 passed; 0 failed' in (d/'scheduler-tests.log').read_text()
assert '21 passed; 0 failed' in (d/'wire-tests.log').read_text()
assert 'corruptions_rejected=5' in (d/'packet-probe.log').read_text()
print('PASS: shared-prefix reuse, exact generation, memcheck, contracts, restoration; serving unmeasured')
