from pathlib import Path
import json,hashlib,tarfile
r=Path('/tmp/riley-opt-260912');names=['serving-round30-analysis.json','serving-screen-round30.log','round30-restoration-public.json','tiled-v14-release.log','variable-candidate-v14/build.json','v14-nsys-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('tiled-v14*') if p.is_file())
names=list(dict.fromkeys(names))
for directory in ['variable-serving-screen-round30','mixed-p128-nsys-v14']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
x={'commit':'31ecb63aa9a605ee341592f85bc1a1cc9d62d2da','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round30-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round30-export.tar.gz','w:gz') as t:
 for n in names+['serving-round30-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
