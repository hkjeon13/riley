"""Collect all terminal serving raw files without IO during serving comparisons."""
import gzip,hashlib,json,shlex,subprocess,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'batch02-batch03-terminal-collection-attempt01';O.mkdir()
REMOTE='/data/riley-serving-261007'
SCOPES=['kernel-batch03-quiet-attempt01','kernel-batch02-quiet-attempt01','kernel-batch02-matched-queue-attempt01']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def remote(code):return subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(code)],text=True)
probe="""import json,subprocess,time
from pathlib import Path
root=Path('/data/riley-serving-261007');live=[]
waits=[(183749,'python run.py --attempt 1 --mode quiet'),(270594,'run_batch02_matched_after_batch03.py')]
child=root/'kernel-batch02-matched-queue-attempt01/process.json'
if child.exists():waits.append((json.loads(child.read_text())['pid'],'python run.py --attempt 1 --mode quiet'))
for pid,name in waits:
 p=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
 if p.returncode==0 and name in p.stdout:live.append(pid)
print(json.dumps({'time_ns':time.time_ns(),'live_pids':live,
 'terminal03':(root/'kernel-batch03-quiet-attempt01/completion.json').exists(),
 'terminal02queue':(root/'kernel-batch02-matched-queue-attempt01/completion.json').exists()}))
"""
inventory="""import json,hashlib
from pathlib import Path
root=Path('/data/riley-serving-261007');scopes=SCOPES_PLACEHOLDER;records={}
for scope in scopes:
 source=root/scope
 if not source.exists():records[scope]={'state':'absent; no execution was inferred'};continue
 assert (source/'completion.json').exists(),'nonterminal source '+scope
 files={}
 for file in sorted(source.rglob('*')):
  assert not file.is_symlink(),'unexpected symlink'
  if not file.is_file():continue
  h=hashlib.sha256()
  with file.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  files[str(file.relative_to(source))]={'sha256':h.hexdigest(),'bytes':file.stat().st_size}
 records[scope]={'state':'terminal','files':files,'completion':json.loads((source/'completion.json').read_text())}
print(json.dumps(records))
""".replace('SCOPES_PLACEHOLDER',repr(SCOPES))
failure=None;records=[]
write('preparation.json',{'waits':{'183749':'Batch03 controller','270594':'serialized Batch02 controller'},
    'scopes':SCOPES,'script_sha256':sha(__file__),'compression':'local gzip1; remote tar only after both serving terminals',
    'scope':'all raw samples, including failed whole attempts; source preserved; may overlap untimed Qwen correctness only'})
try:
    deadline=time.monotonic()+14400
    while True:
        state=json.loads(remote(probe));write('wait-state.json',state)
        with (O/'wait-observations.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        if not state['live_pids']:
            assert state['terminal03'] and state['terminal02queue'],'missing process without both terminal receipts; no restart'
            break
        if time.monotonic()>deadline:raise RuntimeError('serving still live at collection deadline; no raw IO started')
        time.sleep(30)
    before=json.loads(remote(inventory));write('source-manifest-before.json',before)
    for scope,item in before.items():
        if item['state']!='terminal':records.append({'scope':scope,**item});continue
        destination=O/(scope+'.tar.gz')
        with (O/(scope+'-ssh-tar.log')).open('x') as log:
            proc=subprocess.Popen(['ssh','ai-assistant','tar','-cf','-','-C',REMOTE,scope],stdout=subprocess.PIPE,stderr=log)
            try:
                with destination.open('xb') as f:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=f,compresslevel=1,mtime=0) as z:
                        for chunk in iter(lambda:proc.stdout.read(1024*1024),b''):z.write(chunk)
                proc.stdout.close();assert proc.wait()==0,'remote tar failed '+scope
            finally:
                proc.stdout.close()
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
        observed={}
        with tarfile.open(destination,'r|gz') as tar:
            for member in tar:
                assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                if member.isdir():continue
                assert member.isfile() and member.name.startswith(scope+'/'),'unexpected archive member'
                name=member.name[len(scope)+1:];assert name not in observed,'duplicate archive member'
                h=hashlib.sha256()
                stream=tar.extractfile(member)
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                observed[name]={'sha256':h.hexdigest(),'bytes':member.size}
        assert observed==item['files'],'archive/file SHA inventory differs '+scope
        records.append({'scope':scope,'archive':str(destination),'archive_sha256':sha(destination),'bytes':destination.stat().st_size,
            'files_verified':len(observed),'source_bytes':sum(x['bytes'] for x in observed.values()),'completion':item['completion']})
        write('progress.json',records);print(scope+' raw archive SHA verified',flush=True)
    after=json.loads(remote(inventory));write('source-manifest-after.json',after);assert before==after,'source changed during raw collection'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'records':records,'source_deleted':False,'goal_achieved':False})
