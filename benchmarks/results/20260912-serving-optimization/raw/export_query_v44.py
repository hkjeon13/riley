from pathlib import Path
import hashlib,json,subprocess,tarfile,datetime
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
completion=json.loads((r/'variable-serving-screen-round52/completion.json').read_text());assert len(completion['records'])==48
x=json.loads((r/'blender-keep-stopped-public.json').read_text())
for item in x['processes']:
 assert not Path(f"/proc/{item['pid']}").exists()
 assert not subprocess.check_output(['ss','-H','-ltn','sport','=',str(item['port'])],text=True).strip()
x['verified_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(r/'blender-keep-stopped-public.json').write_text(json.dumps(x,indent=2)+'\n')
files=set()
for dirname in ['variable-serving-screen-round52','v4-http-v44-c32-final','query-dispatch-v44','native-trace-v44']:
 files.update(p for p in (r/dirname).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log','.stdout','.py','.cuh','.cu','.nsys-rep','.sqlite'])
for pattern in ['query-v44*.log','query-v44.patch','serving-round52-analysis*','serving-screen-round52.log','*query_v44.py','run_v4_http_v44_c32.py','*trace_v44.py','serving_screen_round52.py','analyze_serving_round52.py']:
 files.update(r.glob(pattern))
files.update([r/'blender-keep-stopped-public.json',r/'variable-candidate-v42/build.json',r/'variable-candidate-v44/build.json'])
manifest={'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
(r/'serving-round52-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'serving-round52-export.tar.gz','w:gz') as tar:
 for p in sorted(files):tar.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(files),'archive_bytes':(r/'serving-round52-export.tar.gz').stat().st_size,'archive_sha256':sha(r/'serving-round52-export.tar.gz')}))
