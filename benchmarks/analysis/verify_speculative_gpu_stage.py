"""Verify exact generation, real draft acceptance, failure history and provenance."""
import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];d=root/'benchmarks/results/20260914-speculative-gpu-stage'
m=json.loads((d/'manifest.json').read_text());archive=d/'evidence.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['archive_sha256']
with tarfile.open(archive) as t:files={p.name:t.extractfile(p).read() for p in t.getmembers() if p.isfile()}
assert set(files)==set(m['files'])
for name,digest in m['files'].items():assert hashlib.sha256(files[name]).hexdigest()==digest,name
for name,digest in m['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,digest in m['local_evidence'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest,name
for version,expected_exit,requests in [(1,0,8),(2,101,12),(3,0,12)]:
    base=f'speculative-stage-lifecycle-v{version}/'
    execution=json.loads(files[base+'execution.json']);assert [x['name'] for x in execution]==['probe','memcheck']
    assert all(x['exit_code']==expected_exit for x in execution)
    assert b'ERROR SUMMARY: 0 errors' in files[base+'memcheck.log']
    for run in ['probe','memcheck']:
        j=json.loads(files[base+run+'-generation.json']);assert len(j['serial'])==requests
        assert j['serial']==j['speculative'] and all(len(x)==32 for x in j['serial'])
        assert j['differing_tokens']==0 and j['verification_calls']>0
        if version==2:assert j['accepted_draft_tokens']==0
        if version==3:assert j['accepted_draft_tokens']>0 and j['speculative_calls']>j['serial_calls']
    assert json.loads(files[base+'blender-restored.json'])['restored']
assert all(x==200 for x in json.loads(files['speculative-stage-lifecycle-v3/viewers.json']).values())
assert all(x['exit_code']==0 for x in json.loads(files['speculative-stage-build-v3/native-builds.json']))
assert '54 passed; 0 failed; 1 ignored' in (d/'runtime-tests.log').read_text()
assert '69 passed; 0 failed' in (d/'scheduler-tests.log').read_text()
assert 'corruptions_rejected=10' in (d/'packet-probe.log').read_text()
print('PASS: exact generation, accepted GPU drafts, memcheck, failed selection gate retained, restoration; serving performance unproven')
