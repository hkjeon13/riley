"""Collect all terminal serving raw files without IO during serving comparisons."""
import gzip,hashlib,json,shlex,subprocess,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch03-profile-terminal-collection-attempt02';O.mkdir()
REMOTE='/data/riley-serving-261007'
SCOPES=['kernel-batch03-profiling-attempt01']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def remote(code):return subprocess.check_output(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant','python3 -c '+shlex.quote(code)],text=True)
probe="""import json,subprocess,time
from pathlib import Path
root=Path('/data/riley-serving-261007');live=[]
p=subprocess.run(['ps','-p','1051472','-o','args='],capture_output=True,text=True)
if p.returncode==0 and 'run_kernel_batch03_profile.py' in p.stdout:live.append(1051472)
terminal=(root/'kernel-batch03-profiling-attempt01/completion.json').exists()
print(json.dumps({'time_ns':time.time_ns(),'live_pids':live,'terminal03':terminal,'terminal02queue':terminal}))
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
write('preparation.json',{'waits':{'1051472':'actual Batch03 profiling controller'},
    'scopes':SCOPES,'script_sha256':sha(__file__),'compression':'remote tar plus gzip1 only after actual profiling terminal; source and partial attempt01 preserved',
    'scope':'all diagnostic raw profiles; source preserved; no serving performance claim'})
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
            proc=subprocess.Popen(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant',shlex.join(['tar','--use-compress-program=gzip -1','-cf','-','-C',REMOTE,scope])],stdout=subprocess.PIPE,stderr=log)
            try:
                with destination.open('xb') as f:
                    for chunk in iter(lambda:proc.stdout.read(1024*1024),b''):f.write(chunk)
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
