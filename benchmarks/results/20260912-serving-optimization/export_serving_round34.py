from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['shared-v24-host-gap.json','serving-round34-analysis.json','serving-screen-round34.log','round34-restoration-public.json','variable-candidate-v24/build.json','v24-nsys-c4-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v24-*.log') if p.is_file())
for directory in ['variable-serving-screen-round34','mixed-p128-nsys-v24-c4-client','v3-http-v24-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'shared-v24.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'190dec94b99af52ba7c17fb34c628506c4302841','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round34-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round34-export.tar.gz','w:gz') as t:
 for n in names+['serving-round34-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
