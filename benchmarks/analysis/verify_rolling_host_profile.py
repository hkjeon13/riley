import gzip,hashlib,json,re,tarfile
from pathlib import Path
from serving_evidence_validation import validate_row
r=Path(__file__).resolve().parents[2];d=r/'benchmarks/results/20260914-rolling-host-profile'
m=json.loads((d/'manifest.json').read_text())
assert hashlib.sha256((d/'evidence.tar.gz').read_bytes()).hexdigest()==m['archive_sha256']
assert hashlib.sha256((d/'source-snapshot.tar.gz').read_bytes()).hexdigest()==m['source_snapshot_sha256']
assert hashlib.sha256(gzip.decompress((d/'fixtures.json.gz').read_bytes())).hexdigest()==m['fixtures_sha256']
with tarfile.open(d/'evidence.tar.gz') as t:f={p.name:t.extractfile(p).read() for p in t.getmembers() if p.isfile()}
assert set(f)==set(m['files'])
for n,h in m['files'].items():assert hashlib.sha256(f[n]).hexdigest()==h,n
with tarfile.open(d/'source-snapshot.tar.gz') as t:
 for p in t.getmembers():
  if p.isfile():assert hashlib.sha256(t.extractfile(p).read()).hexdigest()==m['sources'][p.name]
load=lambda n:json.loads(f[n]);fixtures=json.loads(gzip.decompress((d/'fixtures.json.gz').read_bytes()));total=0
for lane in ['control-shared','instrumented-shared','instrumented-unique','control-unique']:
 kind=lane.split('-')[1];byid={x['id']:x for x in fixtures[kind]}
 receipt=load('profile/'+lane+'-receipt.json');assert receipt['exit_code']==0 and not receipt['remaining_owned_pids'] and 'limit_exceeded' not in receipt
 launch=load('profile/'+lane+'-launch.json');assert launch['fixture_sha256']==m['fixtures_sha256'] and launch['rolling_decode'] and not launch['speculative_decode']
 for phase in ['warmup','rows']:
  rows=load('profile/'+lane+'-'+phase+'.json');assert len(rows)==64
  for row in rows:assert all(validate_row(row,byid[row['id']]).values())
  total+=len(rows)
 if lane.startswith('instrumented'):
  assert launch['binary_sha256']==load('build/binary.json')['sha256']
  lines=load('profile/'+lane+'-host.json')
  values=lambda tag:dict(re.findall(r'(\w+)=(\d+)',next(s for s in lines if tag in s)))
  a=values('RILEY_ROLLING_HOST kind=prefix');b=values('RILEY_ROLLING_HOST kind=drain');c=values('RILEY_ROLLING_DECODE completed')
  assert int(a['steps'])==int(c['completed_steps'])+int(c['drains']) and b['steps']==c['drains']
assert total==512 and load('build/exit.json')['exit_code']==0
assert load('lifecycle/execution.json')==[{'name':'profile','exit_code':0}]
assert load('lifecycle/blender-restored.json')['restored']
assert load('profile/collection-audit.json')['wrapper_exit_code']==1
print('PASS 512 exact responses, four profiler exits, rolling counter identities, restored Blender; wrapper post-collection assertion failure retained')
