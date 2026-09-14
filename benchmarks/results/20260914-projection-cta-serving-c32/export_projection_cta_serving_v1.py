from pathlib import Path
import hashlib,json,tarfile,subprocess
root=Path('/data/riley-serving-260913-recovery');spool=Path('/dev/shm/riley-projection-cta-serving-v2');out=root/'projection-cta-serving-export-v1'
assert json.loads((spool/'complete.json').read_text())=={'lanes':12,'all_complete':True}
assert json.loads((root/'projection-cta-serving-lifecycle-v2/blender-restored.json').read_text())['restored']
out.mkdir()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
files={}
with tarfile.open(out/'evidence.tar.gz','w:gz') as tar:
 for prefix,directory in [('serving',spool),('lifecycle',root/'projection-cta-serving-lifecycle-v2'),('build',root/'projection-cta-serving-build-v1')]:
  for f in sorted(directory.iterdir()):
   if not f.is_file():continue
   name=prefix+'/'+f.name;files[name]=digest(f);tar.add(f,arcname=name)
subprocess.run([str(root/'vllm029-venv/bin/python'),' -m'.strip(),'pip','freeze'],stdout=(out/'vllm029-packages.txt').open('w'),check=True)
manifest={'files':files,'archive_sha256':digest(out/'evidence.tar.gz'),'vllm029_packages_sha256':digest(out/'vllm029-packages.txt')}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(files),'archive_bytes':(out/'evidence.tar.gz').stat().st_size,'sha256':manifest['archive_sha256']}),flush=True)
