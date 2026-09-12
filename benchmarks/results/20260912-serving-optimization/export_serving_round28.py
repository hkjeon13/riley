from pathlib import Path
import json,hashlib,tarfile
r=Path('/tmp/riley-opt-260912');names=['serving-round28-analysis.json','serving-screen-round28.log','round28-restoration-public.json','v3-serving-release-v11.log','v3-nsys-run-v11.log','variable-candidate-v11/build.json']
for directory in ['variable-serving-screen-round28','mixed-p128-nsys-v11']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
x={'commit':'13bd8fcb263f9980447fa3a4d8a2f908f6017299','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round28-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round28-export.tar.gz','w:gz') as t:
 for n in names+['serving-round28-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
