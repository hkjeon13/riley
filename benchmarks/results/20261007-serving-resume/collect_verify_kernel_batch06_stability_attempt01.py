"""Preserve every terminal lifecycle raw file; replay every actual all8 GPU case."""
import pathlib,json,hashlib,subprocess,shlex,time,tarfile,sys
from verify_serving_stability_v2 import verify
R=pathlib.Path(__file__).resolve().parent;O=R/'kernel-batch06-stability-independent-collection-attempt01';O.mkdir();WAIT=R/'kernel-batch06-stability-local-dispatch-attempt01/completion.json';REMOTE='/data/riley-serving-261007/kernel-batch06-stability-attempt01';SCREEN=R/'kernel-batch06-independent-replay-attempt02/summary.json';failure=None;records=[];launched=False

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,d):(O/name).write_text(json.dumps(d,indent=2)+'\n')
def remote(code):return subprocess.check_output(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant','python3 -c '+shlex.quote(code)],text=True)
write('preparation.json',{'wait_for':str(WAIT),'verifier_sha256':sha(R/'verify_serving_stability_v2.py'),'scope':'all actual lifecycle raw SSE/metrics/host/c02 final marker; no partial qualification or exclusions','GPU_stability':'미실행 until dispatcher actual cases exist'})
try:
 deadline=time.monotonic()+43200
 while not WAIT.exists():
  assert time.monotonic()<deadline,'lifecycle dispatcher still pending; no process restart or inferred result'
  time.sleep(30)
 dispatch=json.loads(WAIT.read_bytes());launched=dispatch['launched'];write('dispatcher-terminal-snapshot.json',dispatch)
 if not launched:
  write('skipped.json',{'reason':dispatch['decision'] if dispatch['decision'] else dispatch['failure'],'actual_GPU_stability':'미실행','serving_performance':'미실행','goal_achieved':False})
 else:
  probe="import json,subprocess;from pathlib import Path;live=[]\nfor p in Path('/proc').iterdir():\n if not p.name.isdigit():continue\n try:a=p.joinpath('cmdline').read_bytes().split(bytes([0]))\n except (PermissionError,FileNotFoundError,ProcessLookupError):continue\n if b'/data/riley-serving-261007/kernel-batch06-stability-controller-attempt01/run_serving_stability.py' in a:live.append(int(p.name))\nprint(json.dumps({'live':live,'GPU':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()}))"
  while True:
   state=json.loads(remote(probe));write('wait-state.json',state)
   if not state['live'] and not state['GPU']:break
   assert time.monotonic()<deadline,'original lifecycle still live or GPU occupied; no raw archive IO/restart'
   time.sleep(30)
  inventory="import json,hashlib;from pathlib import Path;r=Path("+repr(REMOTE)+");files={}\nfor p in sorted(r.rglob('*')):\n assert not p.is_symlink(),'unexpected evidence symlink'\n if not p.is_file():continue\n h=hashlib.sha256()\n with p.open('rb') as f:\n  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)\n files[str(p.relative_to(r))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}\nprint(json.dumps(files))"
  before=json.loads(remote(inventory));write('source-manifest-before.json',before);require_cases=[f'c{c}-{w}' for c in [1,8,16,32] for w in ['fixed','natural']]
  archive=O/'kernel-batch06-stability-attempt01.tar.gz'
  with archive.open('xb') as f:subprocess.run(['ssh','-o','ServerAliveInterval=20','ai-assistant',shlex.join(['tar','--use-compress-program=gzip -1','-C',REMOTE,'-cf','-','.'])],stdout=f,check=True)
  source=O/'kernel-batch06-stability-attempt01';source.mkdir();observed={}
  with tarfile.open(archive,'r|gz') as t:
   for m in t:
    if m.isdir():continue
    name=m.name.removeprefix('./');assert m.isfile() and name in before and name not in observed and not pathlib.Path(name).is_absolute() and '..' not in pathlib.Path(name).parts
    path=source/name;path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
    stream=t.extractfile(m)
    with path.open('xb') as f:
     for chunk in iter(lambda:stream.read(1024*1024),b''):
      h.update(chunk);size+=len(chunk);f.write(chunk)
    observed[name]={'sha256':h.hexdigest(),'bytes':size}
  assert observed==before,'raw extracted bytes differ'
  after=json.loads(remote(inventory));write('source-manifest-after.json',after);assert before==after,'remote evidence changed during collection'
  receipt={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files_verified':len(observed),'source_before':before,'source_after':after,'source_preserved':True};write('archive-receipt.json',receipt)
  assert dispatch['failure'] is None and len(dispatch['cases'])==8 and [c['case'] for c in dispatch['cases']]==require_cases,'actual matrix failed/partial; raw preserved, no qualification'
  for label in require_cases:
   result=verify(source/label,R/'kernel-batch06-controller/fixtures.json',SCREEN);write(label+'-independent-verification.json',result);records.append({'case':label,'passed':result['passed'],'all_exact_HTTP_requests_replayed':result['all_exact_HTTP_requests_replayed'],'cancellations':result['cancelled_owned_connections_verified'],'native_final_reclaimed':result['native_final_reclaimed']});write('progress.json',records);print(label+' actual GPU lifecycle raw independent replay PASS',flush=True)
  write('all8-independent-verification.json',{'all8_passed':len(records)==8 and all(x['passed'] for x in records),'cases':records,'archive_sha256':receipt['archive_sha256'],'adopted':False,'goal_achieved':False,'scope':'actual lifecycle correctness/600s sustained/cancellation/re-request/native reclamation; final full-goal audit remains required'})
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'dispatcher_launched':launched,'records':records,'adopted':False,'goal_achieved':False})
