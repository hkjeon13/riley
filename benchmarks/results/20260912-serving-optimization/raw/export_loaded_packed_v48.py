from pathlib import Path
import json,hashlib,tarfile,subprocess,re,statistics
r=Path('/tmp/riley-opt-260912');d=r/'packed-prefill-loaded-v48';src=r/'prefill-shapes-source-v11'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['memcheck','racecheck']:
 s=(d/(n+'.log')).read_text();assert ('ERROR SUMMARY: 0 errors' if n=='memcheck' else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)') in s,n
s=(d/'timing.log').read_text();s,n=re.subn(r'packed full model cases=32 compared_bytes=3021078528\n','',s);assert n==1
rows=[json.loads(l) for l in s.splitlines() if l.startswith('{')];assert len(rows)==96
(d/'timing-clean.json').write_text(json.dumps({'note':'Removed exactly one stderr completion summary interleaved into buffered stdout; original timing.log retained.','rows':rows},indent=2)+'\n')
a=[]
for pattern,owners in sorted({(x['pattern'],x['owners']) for x in rows}):
 v=[x for x in rows if x['pattern']==pattern and x['owners']==owners]
 b=statistics.median(x['us'] for x in v if not x['candidate']);c=statistics.median(x['us'] for x in v if x['candidate'])
 a.append(dict(pattern=pattern,owners=owners,tokens=v[0]['tokens'],sequential_us=b,packed_us=c,change_pct=100*(c/b-1)))
(d/'analysis.json').write_text(json.dumps(a,indent=2)+'\n')
headers=['prefill_shape_model.cuh','prefill_query_tile_attention.cuh','prefill_shape_rope_kv.cuh','prefill_shape_pointwise.cuh','prefill_shape_projection.cuh','decode_tiled.cuh']
(d/'source-binding.json').write_text(json.dumps({'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'headers':{n:sha(src/'kernels/src'/n) for n in headers},'weights_sha256':sha(r/'loaded-rope-fixture-v11/weights.bin'),'rope_sha256':sha(r/'loaded-rope-fixture-v11/rope.bin'),'scope':'Current serving loader fixture; standalone prefix/selected-hidden/entire-KV equality. Timing excludes LM head, D2H results, scheduler and HTTP. Not serving qualification.'},indent=2)+'\n')
files={p for p in d.iterdir() if p.is_file() and p.suffix in ['.cu','.cuh','.log','.json']}
for n in ['prepare_loaded_packed_v48.py','check_loaded_packed_v48.py','export_loaded_packed_v48.py']:files.add(r/n)
m={'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]};(r/'packed-prefill-loaded-v48-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
with tarfile.open(r/'packed-prefill-loaded-v48-export.tar.gz','w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(files),'archive_sha256':sha(r/'packed-prefill-loaded-v48-export.tar.gz'),'analysis':a}))
