from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round36-analysis.json','serving-screen-round36.log','round36-restoration-public.json','variable-candidate-v26/build.json','v26-nsys-c4-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v26-*.log') if p.is_file())
for directory in ['variable-serving-screen-round36','mixed-p128-nsys-v26-c4-client','v3-http-v26-shared-final','v3-http-v26-c8-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'shared-v26.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'6818d7264a4aa845c15813ba5bdcfd419daeb14d','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round36-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round36-export.tar.gz','w:gz') as t:
 for n in names+['serving-round36-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
