from pathlib import Path
import hashlib,json,subprocess,tarfile,datetime
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
build=json.loads((r/'variable-candidate-v42/build.json').read_text());assert all(sha(Path(p))==h for p,h in build['binaries'].items())
x=json.loads((r/'blender-keep-stopped-public.json').read_text())
for item in x['processes']:
 assert not Path(f"/proc/{item['pid']}").exists()
 assert not subprocess.check_output(['ss','-H','-ltn','sport','=',str(item['port'])],text=True).strip()
x['verified_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(r/'blender-keep-stopped-public.json').write_text(json.dumps(x,indent=2)+'\n')
(r/'attention-v42.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
a=json.loads((r/'serving-round50-analysis.json').read_text());a['decision']='Retain V42 as next high-concurrency baseline: throughput gains 2.47-3.33 percent at C16/C32, TPOT and E2E tails improve, some TTFT regresses. Low-concurrency natural throughput is flat/slightly worse. vLLM high-concurrency gap remains. Shape diagnostic motivates evaluating bucketed prefill graphs; not a serving win claim.';(r/'serving-round50-analysis.json').write_text(json.dumps(a,indent=2)+'\n')
files=set()
for dirname in ['variable-serving-screen-round50','native-trace-v41','native-trace-v42','native-shape-v42','v4-http-v42-c32-final','attention-v42-baseline','attention-v42-decode-reference']:
 files.update(p for p in (r/dirname).rglob('*') if p.is_file())
for pattern in ['attention-v42*.log','attention-v42.patch','serving-round50-analysis*','serving-screen-round50.log','*trace_v4*.py','*attention_v42.py','serving_screen_round50.py','analyze_serving_round50.py','run_v4_http_v42_c32.py','prefill-attention-v42.cu']:
 files.update(r.glob(pattern))
files.update([r/'blender-keep-stopped-public.json',r/'variable-candidate-v41/build.json',r/'variable-candidate-v42/build.json'])
manifest={'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
(r/'serving-round50-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'serving-round50-export.tar.gz','w:gz') as tar:
 for p in sorted(files):tar.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(files),'archive_bytes':(r/'serving-round50-export.tar.gz').stat().st_size,'archive_sha256':sha(r/'serving-round50-export.tar.gz')}))
