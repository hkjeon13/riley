"""Verify archived exact target tokens and bounded CUDA safety evidence."""
import hashlib,json,struct,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
directory=root/'benchmarks/results/20260914-verification-greedy'
manifest=json.loads((directory/'manifest.json').read_text())
archive=directory/'evidence.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
with tarfile.open(archive) as tar:
    files={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
assert set(files)==set(manifest['files'])
for name,digest in manifest['files'].items():
    assert hashlib.sha256(files[name]).hexdigest()==digest,name
for name,digest in manifest['sources'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
base='verification-greedy-lifecycle-v1/'
a=files[base+'logits/serial.bf16'];b=files[base+'logits/multi.bf16'];tokens=files[base+'logits/greedy.bf16']
assert len(a)==7077888 and a==b and len(tokens)==288
for i,(token,) in enumerate(struct.iter_unpack('<I',tokens)):
    values=[struct.unpack('<f',struct.pack('<I',x<<16))[0] for (x,) in struct.iter_unpack('<H',a[i*98304:(i+1)*98304])]
    assert token==max(range(49152),key=values.__getitem__)
assert all(x['exit_code']==0 for x in json.loads(files[base+'execution.json']))
assert len(json.loads(files[base+'execution.json']))==5
assert json.loads(files[base+'blender-restored.json'])['restored']
for name in ['probe','memcheck']:
    assert b'VERIFICATION_GREEDY_GATE rows=72 exact_tokens=true' in files[base+name+'.log']
for name in ['memcheck','select-memcheck']:
    assert b'ERROR SUMMARY: 0 errors' in files[base+name+'.log']
assert b'0 hazards' in files[base+'select-racecheck.log']
assert all(x['exit_code']==0 for x in json.loads(files['verification-greedy-build-v1/native-builds.json']))
print('PASS: 72 exact token rows, full logits unchanged, safety gates, source hashes, restoration; no serving claim')
