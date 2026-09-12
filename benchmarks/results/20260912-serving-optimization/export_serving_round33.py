from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round33-analysis.json','serving-screen-round33.log','round33-restoration-public.json','variable-candidate-v23/build.json','v23-nsys-c4-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v23-*.log') if p.is_file())
for directory in ['variable-serving-screen-round33','mixed-p128-nsys-v23-c4-client','v3-http-v23-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'shared-v23.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'a399f04cbe853f836567a66e0689de0c3a51e09f','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round33-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round33-export.tar.gz','w:gz') as t:
 for n in names+['serving-round33-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
