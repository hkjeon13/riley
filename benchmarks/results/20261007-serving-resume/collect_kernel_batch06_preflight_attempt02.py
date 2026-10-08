import subprocess,json,hashlib,tarfile,time
from pathlib import Path
R=Path('/Users/psyche/PycharmProjects/riley/benchmarks/results/20261007-serving-resume');O=R/'kernel-batch06-preflight-independent-collection-attempt02';O.mkdir()
remote='/data/riley-serving-261007/'
names=['kernel-batch06-primitive-validation-attempt01','kernel-batch06-primitive-validation-attempt02','kernel-batch06-server-validation-attempt02','kernel-batch06-http-screen-attempt02']
for name in names:
 terminal='controller-completion.json' if 'hf-validation' in name else 'completion.json'
 script="import time,json;from pathlib import Path;p=Path("+repr(remote+name+'/'+terminal)+");\nwhile not p.exists():time.sleep(5)\nprint(p.read_text())"
 result=subprocess.check_output(['ssh','ai-assistant','python3 -c '+__import__('shlex').quote(script)],text=True);completion=json.loads(result)
 def inventory():
  script="import json,hashlib;from pathlib import Path;r=Path("+repr(remote+name)+");print(json.dumps({str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(r.rglob('*')) if p.is_file()}))"
  return json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+__import__('shlex').quote(script)],text=True))
 before=inventory();archive=O/(name+'.tar.gz')
 with archive.open('xb') as f:subprocess.run(['ssh','ai-assistant','tar','-C',remote+name,'-czf','-','.'],stdout=f,check=True)
 after=inventory();assert before==after
 dest=O/name;dest.mkdir();local={}
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   if m.isdir():continue
   n=m.name.removeprefix('./');assert m.isfile() and n in before and not Path(n).is_absolute() and '..' not in Path(n).parts
   b=t.extractfile(m).read();local[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};q=dest/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 assert local==before
 receipt={'completion':completion,'source_before':before,'source_after':after,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'files_verified':len(local),'source_preserved':True,'serving_performance':'미실행'}
 (O/(name+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(name,len(local),completion,flush=True)
