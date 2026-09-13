"""Verify non-profiler serving diagnostics and summarize nested host timers."""
import hashlib,json,pathlib,re,sys,tarfile
root=pathlib.Path(sys.argv[1]);archive=root/'evidence.tar.gz';results={};prefix='composed-host-phase-v1/'
with tarfile.open(archive) as tar:
 def read(name):return tar.extractfile(prefix+name).read()
 def load(name):return json.loads(read(name))
 preparation=load('preparation.json')
 assert preparation['host_phase_diagnostic'] and preparation['concurrency']==32
 assert preparation['controller_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('composed_host_phase_diagnostic.py').read_bytes()).hexdigest()
 assert load('complete.json')=={'lanes':2,'all_complete':True}
 assert not read('compute-after.csv').strip()
 for lane in ('shared','unique'):
  name=lane+'-p0-composed';launch=load(name+'-launch.json')
  assert launch['environment']['RILEY_SERVING_PHASE_TIMING']=='1' and launch['environment']['RILEY_MIXED_QUERY_REUSE']=='0'
  assert launch['environment']['RILEY_PREFIX_CACHE_PAGES']=='512'
  assert load(name+'-exit.json')['exit_code']==0
  for phase,count in [('warmup',64),('retained',256)]:
   rows=load(name+'-'+phase+'.json');assert len(rows)==count
   assert all(r['valid'] and all(r['checks'].values()) and len(r['token_ids'])==32 for r in rows)
  host={};runtime={}
  for line in read(name+'.log').decode().splitlines():
   if line.startswith('RILEY_HOST_PHASE '):
    d=dict(re.findall(r'(\w+)=(\w+)',line));kind=d.pop('kind');host[kind]={k:int(v) for k,v in d.items()}
   if line.startswith('RILEY_RUNTIME_PHASE '):
    d=dict(re.findall(r'(\w+)=(\w+)',line));kind=d.pop('kind');calls=int(d['calls']);ns=int(d['wall_ns']);runtime[kind]={'calls':calls,'wall_ns':ns,'mean_us':ns/calls/1000 if calls else None}
  assert set(host)=={'decode','prefill_or_mixed','paired_decode'} and len(runtime)==6
  results[lane]={'host':host,'runtime':runtime}
(root/'receipt.json').write_text(json.dumps({'scope':'host timers enabled, no Nsight; warmup+retained combined; nested timers overlap and are not a speedup estimate','verified_requests':640,'preparation':preparation,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'results':results},indent=2)+'\n')
print('verified 640 reference-exact requests and both host phase reports')
