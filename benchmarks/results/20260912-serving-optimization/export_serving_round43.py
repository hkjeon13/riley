from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round43-analysis.json','serving-screen-round43.log','round43-restoration-public.json','variable-candidate-v33/build.json','v33-nsys-natural-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v33-*.log') if p.is_file())
for directory in ['variable-serving-screen-round43','v3-http-v33-shared-final','v3-http-v33-c8-shared-final','mixed-natural-nsys-v33-c4-client']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
commit='214c8ed7309d00ea908c7e29e3c2ed962f8f945d';src=r/'prefill-shapes-source-v11'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==commit and not subprocess.check_output(['git','status','--porcelain'],cwd=src)
assert hashlib.sha256((r/'prefill-shapes-target-v11/release/riley').read_bytes()).hexdigest()=='5ffceea0d2494f951cf3b92115ad04063bbadb8e5c45c3c5ffdf67a3cd8ec7a3'
(r/'projection-v33.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller',commit],cwd=src));names.append('projection-v33.patch')
x={'commit':commit,'accepted_baseline':True,'files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round43-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round43-export.tar.gz','w:gz') as t:
 for n in names+['serving-round43-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
