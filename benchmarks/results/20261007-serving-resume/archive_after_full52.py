"""Archive terminal evidence after all serialized GPU work; no benchmark overlap."""
import hashlib,json,subprocess,time,tarfile
from pathlib import Path
R=Path('/data/riley-serving-261007');O=R/'terminal-evidence-archive-attempt01';O.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
waits={3605799:'kernel-batch01-controller/run.py',3630166:'run_qwen_after_kernel_batch01.py',3697803:'run_kernel_batch02_after_qwen.py',3733960:'run_qwen_full52_after_batch02.py'}
scopes=['kernel-batch01-quiet-attempt01','kernel-batch01-diagnostic-attempt01','qwen-m1-regular-continuation01','qwen-m1-regular-full-model-attempt01','kernel-batch02-validation-attempt01','kernel-batch02-http-screen-attempt01','qwen-m1-regular-full52-attempt01']
records=[];failure=None
write('preparation.json',{'waits':waits,'scopes':scopes,'source_sha256':sha(__file__),'benchmark_overlap':'prohibited; wait for exact process handles and terminal receipts','old_archives':'preserved'})
try:
 while True:
  live=[]
  for pid,expected in waits.items():
   p=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
   if p.returncode==0 and expected in p.stdout:live.append(pid)
  if not live:break
  time.sleep(5)
 for scope in scopes:
  source=R/scope
  if not source.exists():records.append({'scope':scope,'state':'absent; no source was fabricated'});continue
  assert (source/'completion.json').exists(),'nonterminal source '+scope
  files={str(p.relative_to(source)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(source.rglob('*')) if p.is_file()}
  dst=O/(scope+'.tar.gz')
  with tarfile.open(dst,'x:gz',compresslevel=1) as t:t.add(source,arcname=scope,recursive=True)
  for name,receipt in files.items():assert (source/name).stat().st_size==receipt['bytes'] and sha(source/name)==receipt['sha256'],'source changed during archive '+scope+'/'+name
  rec={'scope':scope,'state':'archived','archive':str(dst),'bytes':dst.stat().st_size,'sha256':sha(dst),'files':files,'terminal':json.loads((source/'completion.json').read_text())}
  write(scope+'-manifest.json',rec);records.append({k:v for k,v in rec.items() if k!='files'});write('progress.json',records);print(scope+' archived',flush=True)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'records':records,'goal_achieved':False,'source_deleted':False})
