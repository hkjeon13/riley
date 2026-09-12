from pathlib import Path
import json,hashlib,tarfile
r=Path('/tmp/riley-opt-260912');names=['serving-round29-analysis.json','serving-screen-round29.log','round29-restoration-public.json','decode-batch-v12-release.log','v12-nsys-run.log','variable-candidate-v12/build.json']
names.extend(str(p.relative_to(r)) for p in r.glob('decode-batch-v12*') if p.is_file())
names=list(dict.fromkeys(names))
for directory in ['variable-serving-screen-round29','mixed-p128-nsys-v12']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
x={'commit':'00820aed7e13e49e0815882774105c8dcb89110f','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round29-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round29-export.tar.gz','w:gz') as t:
 for n in names+['serving-round29-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
