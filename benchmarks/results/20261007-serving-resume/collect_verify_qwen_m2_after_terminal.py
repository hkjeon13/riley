"""Collect actual terminal M2 evidence and independently compare native BF16 bytes."""
import gzip,hashlib,json,shlex,subprocess,sys,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'qwen-m2-independent-collection-attempt01';O.mkdir()
D=R/'qwen-m2-local-dispatch-attempt01/completion.json'
REMOTE='/data/riley-serving-261007';SCOPE='qwen-m2-bounded-validation-attempt01'
TEACHER='/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z/cache-on-logits.safetensors'
EXPECTED_TEACHER='d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def remote(code):return subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(code)],text=True)
probe="""import json,subprocess
from pathlib import Path
root=Path('/data/riley-serving-261007');d=root/'qwen-m2-bounded-validation-attempt01'
actors=subprocess.check_output(['ps','-eo','pid,args'],text=True)
live=[line for line in actors.splitlines() if 'run_qwen_m2_bounded_attempt01.py' in line and 'python' in line and 'python3 -c' not in line]
print(json.dumps({'live_native_controllers':live,'exists':d.exists(),'terminal':(d/'completion.json').exists(),
 'serving02terminal':(root/'kernel-batch02-matched-queue-attempt01/completion.json').exists(),'serving03terminal':(root/'kernel-batch03-quiet-attempt01/completion.json').exists()}))
"""
inventory="""import hashlib,json
from pathlib import Path
p=Path('/data/riley-serving-261007/qwen-m2-bounded-validation-attempt01');assert (p/'completion.json').exists()
files={}
for f in sorted(p.rglob('*')):
 assert not f.is_symlink()
 if not f.is_file():continue
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 files[str(f.relative_to(p))]={'sha256':h.hexdigest(),'bytes':f.stat().st_size}
print(json.dumps(files))
"""
pin_names=['verify_qwen_m2_native_raw.py','qwen-m2-bounded-source-receipt-attempt01.json','verified-stages-attempt02/hf-original/qwen3b-p2048-cache-on-layer-stage.json','verified-stages-attempt02/hf-original/qwen3b-p2048-cache-on-layer-stage.safetensors']
pins={name:sha(R/name) for name in pin_names}
write('preparation.json',{'script_sha256':sha(Path(__file__)),'pins':pins,'wait_for':str(D),'teacher_expected_sha256':EXPECTED_TEACHER,'scope':'actual terminal only; preserve failed build and numerical evidence; native104 byte replay; serving 미실행/full128 unverified'})
failure=None;steps=[]
try:
 deadline=time.monotonic()+21600
 while not D.exists():
  write('wait-state.json',{'time_ns':time.time_ns(),'dispatch_terminal':False,'collection_executed':False})
  if time.monotonic()>deadline:raise RuntimeError('actual M2 dispatcher terminal absent at deadline; no collection inferred')
  time.sleep(30)
 dispatch=json.loads(D.read_text());write('dispatch-receipt-snapshot.json',dispatch)
 state=json.loads(remote(probe));write('remote-terminal-state.json',state)
 assert not state['live_native_controllers'],'native controller still live; no raw IO started'
 assert state['serving02terminal'] and state['serving03terminal'],'serving terminal receipts missing'
 if not state['exists']:
  steps.append({'native_execution':'미실행; actual source/output directory absent','dispatch_failure':dispatch['failure']})
 else:
  assert state['terminal'],'native directory exists without terminal; no restart or execution inferred'
  before=json.loads(remote(inventory));write('source-manifest-before.json',before)
  archive=O/(SCOPE+'.tar.gz')
  with (O/'ssh-tar.log').open('x') as log:
   proc=subprocess.Popen(['ssh','ai-assistant','tar','-cf','-','-C',REMOTE,SCOPE],stdout=subprocess.PIPE,stderr=log)
   try:
    with archive.open('xb') as f:
     with gzip.GzipFile(filename='',mode='wb',fileobj=f,compresslevel=1,mtime=0) as z:
      for chunk in iter(lambda:proc.stdout.read(1024*1024),b''):z.write(chunk)
    proc.stdout.close();assert proc.wait()==0,'remote evidence tar failed'
   finally:
    proc.stdout.close()
    if proc.poll() is None:
     proc.terminate()
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
  observed={};destination=O/SCOPE;destination.mkdir()
  with tarfile.open(archive,'r|gz') as tar:
   for member in tar:
    assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    if member.isdir():continue
    assert member.isfile() and member.name.startswith(SCOPE+'/')
    name=member.name[len(SCOPE)+1:];assert name not in observed
    data=tar.extractfile(member).read();observed[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    assert name in before and observed[name]==before[name],'raw archive file hash differs'
    target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(data)
  assert observed==before,'missing archive member'
  after=json.loads(remote(inventory));write('source-manifest-after.json',after);assert before==after,'remote terminal evidence changed during collection'
  steps.append({'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files_verified':len(before),'native_terminal':json.loads((destination/'completion.json').read_text())});write('progress.json',steps)
  result=destination/'qwen3b-p2048-cache-on-m1-m2-native-result.json'
  if result.exists():
   assert all(sha(R/name)==digest for name,digest in pins.items()),'local verifier/oracle/source changed after declaration'
   teacher=O/'teacher-cache-on-logits.safetensors'
   with teacher.open('xb') as stream:
    subprocess.run(['ssh','ai-assistant','cat',TEACHER],stdout=stream,check=True)
   assert sha(teacher)==EXPECTED_TEACHER,'immutable full teacher sidecar hash mismatch'
   write('teacher-download-receipt.json',{'remote_path':TEACHER,'bytes':teacher.stat().st_size,'sha256':sha(teacher)})
   cmd=[sys.executable,str(R/'verify_qwen_m2_native_raw.py'),str(destination),'--hf-directory',str(R/'verified-stages-attempt02/hf-original'),'--teacher-sidecar',str(teacher),'--source-receipt',str(R/'qwen-m2-bounded-source-receipt-attempt01.json'),'--output',str(O/'independent-native104-verification.json')]
   with (O/'independent-verifier.log').open('x') as log:exitcode=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
   steps.append({'independent_raw_verifier_exit':exitcode});write('progress.json',steps);assert exitcode==0,'independent native104 replay failed; raw evidence preserved'
  else:steps.append({'independent_native_raw_replay':'미실행; actual result missing','native_failure':json.loads((destination/'completion.json').read_text())['failure']})
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'steps':steps,'serving_performance':'미실행','full128_generation':'unverified','source_deleted':False,'goal_achieved':False})
