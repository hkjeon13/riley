import pathlib,json,hashlib,tarfile,subprocess
r=pathlib.Path('/tmp/riley-opt-260912')
raw=r/'shared-natural-v10/dumps'
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.rglob('*')) if p.is_file()}
(r/'shared-natural-v10/raw-logits-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
patch=subprocess.check_output(['git','diff','d589dba24ddb23ca37d66b6338fa9999a10abf29','d073b1e8665ec1f5eb4fc56b6ac0a75e8cd3ce65'],cwd=r/'multisequence-shared-source-v10')
(r/'shared-v10-source.patch').write_bytes(patch)
files=[]
for d in ('mixed-p128-screen-round27','shared-natural-v10','mixed-p128-nsys-v10'):
 for p in (r/d).rglob('*'):
  if p.is_file() and 'dumps' not in p.parts and p.suffix not in ('.sqlite','.nsys-rep'):files.append(p)
for pattern in ('shared-v10-*.log','shared-v10-source.patch','natural-eval-wheel-manifest.json','mixed-p128-screen-round27-analysis.json'):
 files.extend(r.glob(pattern))
files.append(r/'multisequence-candidate-v10/build.json')
manifest={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
m=r/'v10-round27-export-sha256.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'v10-round27-export.tar.gz','w:gz') as tar:
 for p in sorted(set(files))+[m]:tar.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(manifest),'logit_files_remote':len(hashes),'archive_bytes':(r/'v10-round27-export.tar.gz').stat().st_size}))
