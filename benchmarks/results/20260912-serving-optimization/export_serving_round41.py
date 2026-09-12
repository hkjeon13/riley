from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round41-analysis.json','serving-screen-round41.log','round41-restoration-public.json','variable-candidate-v31/build.json','v31-rejection.json','v31-nsys-natural-run.log','decode_shared_attention_v30_reference.cuh']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v31-*.log') if p.is_file())
for directory in ['variable-serving-screen-round41','v3-http-v31-shared-final','v3-http-v31-c8-shared-final','mixed-natural-nsys-v31-c4-client']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
for name,commit in [('attention-v31.patch','3bc31075ba70c95ed58ad5372ed7ad005730c4e6'),('attention-v31-revert.patch','b5072806119663baa425fb8e6e06844db30e7789')]:
 (r/name).write_bytes(subprocess.check_output(['git','show','--format=fuller',commit],cwd=r/'prefill-shapes-source-v11'));names.append(name)
x={'commit':'3bc31075ba70c95ed58ad5372ed7ad005730c4e6','rejected':True,'files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round41-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round41-export.tar.gz','w:gz') as t:
 for n in names+['serving-round41-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
