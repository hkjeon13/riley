from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');files=set()
for pattern in ['compact-v46*.log','compact-v46*.json','compact-v46.patch','*compact*v46.py','compact_result_probe_v46.cu']:
 files.update(r.glob(pattern))
for dirname in ['v4-http-v46-c32-final','v4-http-v46-c32-retry1','v4-http-v46-c32-retry2','v4-http-v46-c32-final-verified','v4-fallback-v46-cpu','v4-fallback-v46-gpu-greedy','v4-fallback-ordered-v46-cpu','v4-fallback-ordered-v46-gpu-greedy']:
 files.update(p for p in (r/dirname).rglob('*') if p.is_file())
files.update([r/'variable-candidate-v46/build.json',r/'serving_screen_round53.py',r/'analyze_serving_round53.py',r/'blender-keep-stopped-public.json'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
(r/'compact-v46-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'compact-v46-export.tar.gz','w:gz') as tar:
 for p in sorted(files):tar.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(files),'archive_sha256':sha(r/'compact-v46-export.tar.gz')}))
