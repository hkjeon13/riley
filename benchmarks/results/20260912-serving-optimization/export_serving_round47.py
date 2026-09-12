from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');names=['serving-round46-analysis.json','serving-round47-analysis.json','serving-screen-round46.log','serving-screen-round47.log','variable-candidate-v36/build.json','variable-candidate-v37/build.json','blender-keep-stopped-public.json']
for v in (36,37):names.extend(str(p.relative_to(r)) for p in r.glob(f'cpu-v{v}-*.log') if p.is_file())
for directory in ['variable-serving-screen-round46','variable-serving-screen-round47','v3-http-v36-c32-shared-final','v3-http-v36-c32-shared-r2-final','v3-http-v37-c32-shared-r2-final','v3-http-v37-c32-shared-r3-final']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file())
for name,commit in [('active-v36.patch','562fa77305f09f2ccfbf1c0fc0c459e3d7b963e7'),('http-v37.patch','424b6edec524fac0c67d7126f272f5945e6bf52a')]:
 (r/name).write_bytes(subprocess.check_output(['git','show','--format=fuller',commit],cwd=r/'prefill-shapes-source-v11'));names.append(name)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r/'prefill-shapes-source-v11')
x={'source_commit':'424b6edec524fac0c67d7126f272f5945e6bf52a','serving_goal_achieved':False,'files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'blender_left_stopped':True}
(r/'serving-round47-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round47-export.tar.gz','w:gz') as t:
 for n in names+['serving-round47-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
