from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round40-analysis.json','serving-screen-round40.log','round40-restoration-public.json','variable-candidate-v30/build.json','host-v30-comparison.json','host-v30-diagnostic-builds.json','host-v30-before-diagnostic.patch','host-v30-after-diagnostic.patch','host-v30-before-release.log','host-v30-after-release.log','host-v30-pair.log']
names.extend(str(p.relative_to(r)) for p in r.glob('cpu-v30-*.log') if p.is_file())
for directory in ['variable-serving-screen-round40','v3-http-v30-shared-final','v3-http-v30-c8-shared-final','fallback-http-v30','v3-http-v30-host-before-shared-final','v3-http-v30-host-after-shared-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
patch=r/'cpu-v30.patch';patch.write_bytes(subprocess.check_output(['git','show','--format=fuller','00c9da9c9c50d214c1f2f111c2ce8ab0d2e04d1e'],cwd=r/'prefill-shapes-source-v11'));names.append(patch.name)
x={'commit':'00c9da9c9c50d214c1f2f111c2ce8ab0d2e04d1e','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round40-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round40-export.tar.gz','w:gz') as t:
 for n in names+['serving-round40-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
