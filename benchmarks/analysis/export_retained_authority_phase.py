"""Reconcile baseline/new host diagnostic rows and nested timer means."""
import hashlib,json,pathlib,re,sys,tarfile
root=pathlib.Path(sys.argv[1]);results={}
for name,path,prefix in [('prior',root.parent/'20260914-composed-host-phase/evidence.tar.gz','composed-host-phase-v1/'),('new',root/'host-phase.tar.gz','composed-host-phase-retained-v1/')]:
 with tarfile.open(path) as tar:
  def read(n):return tar.extractfile(prefix+n).read()
  def load(n):return json.loads(read(n))
  assert load('complete.json')=={'lanes':2,'all_complete':True}
  preparation=load('preparation.json');assert preparation['host_phase_diagnostic']
  assert preparation['controller_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('composed_host_phase_diagnostic.py').read_bytes()).hexdigest()
  result={}
  for lane in ('shared','unique'):
   stem=lane+'-p0-composed';assert load(stem+'-exit.json')['exit_code']==0
   for phase,count in [('warmup',64),('retained',256)]:
    rows=load(stem+'-'+phase+'.json');assert len(rows)==count and all(r['valid'] and all(r['checks'].values()) for r in rows)
   stages={}
   for line in read(stem+'.log').decode().splitlines():
    if line.startswith('RILEY_RUNTIME_PHASE '):
     d=dict(re.findall(r'(\w+)=(\w+)',line));calls=int(d['calls']);ns=int(d['wall_ns']);stages[d['kind']]={'calls':calls,'wall_ns':ns,'mean_us':ns/calls/1000 if calls else None}
   assert len(stages)==6;result[lane]=stages
  results[name]={'preparation':preparation,'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'stages':result}
(root/'phase-comparison.json').write_text(json.dumps({'scope':'nested host diagnostics, warmup+retained; call mixes differ; not a speedup estimate','results':results},indent=2)+'\n')
print('verified prior/new host traces and 1280 reference-exact requests')
