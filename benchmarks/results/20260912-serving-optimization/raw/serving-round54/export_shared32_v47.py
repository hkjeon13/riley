from pathlib import Path
import json,statistics,hashlib,tarfile
r=Path('/tmp/riley-opt-260912');files=set()
for name in ['shared32-v47','shared32-qkv-v47']:
 d=r/name;g={}
 for line in (d/'timing.log').read_text().splitlines():
  x=json.loads(line);g.setdefault((x['kind'],x['rows'],x['context']),{}).setdefault(x['candidate'],[]).append(x['us'])
 rows=[]
 for (kind,n,c),q in g.items():
  b=statistics.median(q[False]);v=statistics.median(q[True]);rows.append({'kind':kind,'rows':n,'context':c,'baseline_us':b,'candidate_us':v,'change_pct':100*(v/b-1)})
 (d/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
 files.update(p for p in d.iterdir() if p.suffix in ['.cu','.cuh','.log','.json'])
for n in ['prepare_shared32_v47.py','check_shared32_v47.py','add_shared32_qkv_v47.py','check_shared32_qkv_v47.py']:files.add(r/n)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m={'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]};(r/'shared32-v47-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
with tarfile.open(r/'shared32-v47-export.tar.gz','w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(files),'archive_sha256':sha(r/'shared32-v47-export.tar.gz')}))
for x in rows:
 if x['rows']==32:print(x)
