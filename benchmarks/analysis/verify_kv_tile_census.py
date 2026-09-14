"""Verify archived diagnostic responses and reconstruct physical-sharing summaries."""
import ast,hashlib,json,re,sys,tarfile
from pathlib import Path
from serving_evidence_validation import validate_row

d=Path(sys.argv[1]);m=json.loads((d/'manifest.json').read_text())
sha=lambda data:hashlib.sha256(data).hexdigest()
assert sha((d/'evidence.tar.gz').read_bytes())==m['archive_sha256']
assert sha((d/'source-snapshot.tar.gz').read_bytes())==m['source_snapshot_sha256']
def read_archive(path):
    with tarfile.open(path) as t:return {p.name:t.extractfile(p).read() for p in t if p.isfile()}
files=read_archive(d/'evidence.tar.gz');sources=read_archive(d/'source-snapshot.tar.gz')
assert {n:sha(v) for n,v in files.items()}==m['files']
assert {n:sha(v) for n,v in sources.items()}==m['sources']
load=lambda n:json.loads(files[n])
b='riley-kv-census-v2/';l='kv-census-lifecycle-v2/'
assert load(b+'complete.json')=={'lanes':2,'all_complete':True}
assert load(l+'execution.json')==[{'name':'serving','exit_code':0}]
assert load(l+'blender-restored.json')['restored'] is True
build=load('kv-census-build-v2.json');preparation=load(b+'preparation.json')
assert build['build_exit_code']==0 and build['diagnostic_marker'] is True
assert preparation['hashes']['candidate']==build['sha256']
assert preparation['concurrency']==preparation['active_capacity']==32
assert preparation['diagnostic_only'] is True
assert files[b+'controller-snapshot.py']==sources['benchmarks/analysis/kv_tile_census_screen.py']
assert files[b+'client-snapshot.py']==sources['benchmarks/analysis/paired_decode_serving_screen.py']
fixtures=load(b+'fixtures.json');result={};total=0
for kind in ('shared','unique'):
    name=kind+'-p0-candidate';byid={f['id']:f for f in fixtures[kind]}
    assert load(b+name+'-launch.json')['environment']['RILEY_SERVING_PHASE_TIMING']=='1'
    assert load(b+name+'-exit.json')['exit_code']==0
    for phase,count in [('warmup',64),('retained',128)]:
        rows=load(b+name+'-'+phase+'.json');assert len(rows)==count
        for row in rows:assert len(row['token_ids'])==32 and all(validate_row(row,byid[row['id']]).values())
        total+=count
    reports=re.findall(r'RILEY_KV_TILE_GROUPS decode_batches=(\d+) groups_by_owner_count=(\[[^\n]+?\])',files[b+name+'.log'].decode())
    assert len(reports)==1
    batches,raw=reports[0];h=ast.literal_eval(raw);c=load(b+name+'-census.json')
    assert int(batches)==c['decode_batches']>0 and h==c['groups_by_owner_count']
    assert len(h)==33 and h[0]==0 and all(isinstance(n,int) and n>=0 for n in h)
    refs=sum(i*n for i,n in enumerate(h))
    result[kind]={'decode_observations':int(batches),'row_tile_references':refs,'shared_row_tile_references':refs-h[1],'max_owners':max(i for i,n in enumerate(h) if n)}
receipt={'verified_responses':total,'scope':'diagnostic only; no serving performance qualification','results':result}
(d/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
