from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['fill-v29-analysis.json','fill-v29-corrected-analysis.json','fill-v29-instrumentation.patch','fill-v29-release.log','fill-diagnostic-v29/build.json','serving-round39-analysis.json','serving-screen-round39.log','round39-restoration-public.json','variable-candidate-v29/build.json','v29-nsys-c4-run.log','v29-nsys-natural-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v29-*.log') if p.is_file())
names.extend(str(p.relative_to(r)) for p in r.glob('fusion-v29-*.log') if p.is_file())
for directory in ['natural-fill-v29','natural-fill-v29-corrected','variable-serving-screen-round39','mixed-p128-nsys-v29-c4-client','mixed-natural-nsys-v29-c4-client','v3-http-v29-shared-final','v3-http-v29-c8-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'shared-v29.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'0526d4f3d341a3aa815f806c6f2e6cbab8d86d07','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round39-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round39-export.tar.gz','w:gz') as t:
 for n in names+['serving-round39-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
