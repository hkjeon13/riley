from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round35-analysis.json','serving-screen-round35.log','round35-restoration-public.json','variable-candidate-v25/build.json','host-v25-comparison.json','host-v25-analysis.json','host-diagnostic-v25-after/build.json']
names.extend(str(p.relative_to(r)) for p in r.glob('host-v25-*') if p.is_file() and p.suffix in ('.log','.patch'))
for directory in ['variable-serving-screen-round35','v3-http-v25-shared-final','v3-http-v25-host-shared-final','v3-http-v25-host-after-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'host-v25.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'a12e94cd5df75963d8af3bd93420eb553465dfda','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round35-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round35-export.tar.gz','w:gz') as t:
 for n in names+['serving-round35-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
