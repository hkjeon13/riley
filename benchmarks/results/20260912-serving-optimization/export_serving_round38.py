from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['prefill_attention_v27_reference.cuh','serving-round38-analysis.json','serving-screen-round38.log','round38-restoration-public.json','variable-candidate-v28/build.json','v28-nsys-c4-run.log','v28-nsys-natural-run.log']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v28-*.log') if p.is_file())
for directory in ['variable-serving-screen-round38','mixed-p128-nsys-v28-c4-client','mixed-natural-nsys-v28-c4-client','v3-http-v28-shared-final','v3-http-v28-c8-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'shared-v28.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'e181ae97e1202050654e887753893b609dd5ff44','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round38-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round38-export.tar.gz','w:gz') as t:
 for n in names+['serving-round38-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
