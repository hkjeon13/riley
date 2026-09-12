from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round42-analysis.json','serving-screen-round42.log','round42-restoration-public.json','variable-candidate-v32/build.json','v32-rejection.json','v32-nsys-natural-run.log','decode_shared_attention_v30_reference.cuh']
names.extend(str(p.relative_to(r)) for p in r.glob('shared-v32-*.log') if p.is_file())
for directory in ['variable-serving-screen-round42','v3-http-v32-shared-final','v3-http-v32-c8-shared-final','v3-http-v32-shared-r2-final','v3-http-v32-c8-shared-r2-final','mixed-natural-nsys-v32-c4-client']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
for name,commit in [('attention-v32.patch','fa06f6c31ceee4591313ad36c56ebfe70fda1f43'),('attention-v32-revert.patch','8b6ed46dc8fe4cbe48844ad0162285b5952bfe0a')]:
 (r/name).write_bytes(subprocess.check_output(['git','show','--format=fuller',commit],cwd=r/'prefill-shapes-source-v11'));names.append(name)
x={'commit':'fa06f6c31ceee4591313ad36c56ebfe70fda1f43','rejected':True,'files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round42-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round42-export.tar.gz','w:gz') as t:
 for n in names+['serving-round42-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
