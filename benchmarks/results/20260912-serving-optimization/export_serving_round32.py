from pathlib import Path
import json,hashlib,tarfile
r=Path('/tmp/riley-opt-260912');names=['serving-round32-analysis.json','serving-screen-round32.log','round32-restoration-public.json','server-shared-v22-release.log','variable-candidate-v22/build.json','v22-nsys-run.log','v22-nsys-c4-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('server-shared-v22*') if p.is_file())
names.extend(['serving-screen-round31.log','round31-restoration-public.json'])
names=list(dict.fromkeys(names))
for directory in ['variable-serving-screen-round32','mixed-p128-nsys-v22','mixed-p128-nsys-v22-c4-client','v3-http-v22-shared-final','variable-serving-screen-round31']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
x={'commit':'52be6c4091afdd1bfaa44350a1ee5e3753d9185a','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round32-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round32-export.tar.gz','w:gz') as t:
 for n in names+['serving-round32-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
