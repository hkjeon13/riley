from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['prefill_projection_v26_reference.cuh','serving-round37-analysis.json','serving-screen-round37.log','round37-restoration-public.json','variable-candidate-v27/build.json','v27-nsys-c4-run.log','v27-nsys-natural-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v27-*.log') if p.is_file())
for directory in ['variable-serving-screen-round37','mixed-p128-nsys-v27-c4-client','mixed-natural-nsys-v27-c4-client','v3-http-v27-shared-final','v3-http-v27-c8-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'shared-v27.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'7f0865f07433f9af53931df1e308f9e8b94692f7','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round37-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round37-export.tar.gz','w:gz') as t:
 for n in names+['serving-round37-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
