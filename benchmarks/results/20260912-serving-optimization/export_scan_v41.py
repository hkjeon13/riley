from pathlib import Path
import hashlib,json,subprocess,tarfile,datetime
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
build=json.loads((r/'variable-candidate-v41/build.json').read_text());assert all(sha(Path(p))==h for p,h in build['binaries'].items());assert sha(r/'prefill-shapes-target-v11/release/riley')==next(iter(build['binaries'].values()))
x=json.loads((r/'blender-keep-stopped-public.json').read_text())
for item in x['processes']:
 assert not Path(f"/proc/{item['pid']}").exists()
 assert not subprocess.check_output(['ss','-H','-ltn','sport','=',str(item['port'])],text=True).strip()
x['verified_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(r/'blender-keep-stopped-public.json').write_text(json.dumps(x,indent=2)+'\n')
(r/'scan-v41.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
receipt={'build':build,'cpu_tests':40,'guard_page_test':True,'exhaustive_bf16_and_unaligned_tails':True,'gpu_tests':3,'full_logit_rows':4896,'http_completed':37,'http_disconnect_recovery':True,'cleanup_zero':True,'cuda_kernels_changed':False,'retained_requests':18432,'riley_reference_matches':12288,'diagnostic_requests':1536,'goal_achieved':False,'decision':'Retain SIMD validation for both widths. Serving throughput improves4.05-9.67 percent over matched V40 profiles, TTFT/TPOT and sampled E2E tails improve. vLLM gap remains at higher concurrency. Next isolate native execution/synchronization/readback with GPU trace. No long-term stability claim.'}
(r/'scan-v41-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
a=json.loads((r/'serving-round49-analysis.json').read_text());a['decision']=receipt['decision'];(r/'serving-round49-analysis.json').write_text(json.dumps(a,indent=2)+'\n')
files=set()
for dirname in ['variable-serving-screen-round49','natural-fill-v41','v4-http-v41-c32-final']:
 files.update(p for p in (r/dirname).rglob('*') if p.is_file())
for pattern in ['scan-v41*.log','fill-v41*.log','scan-v41.patch','scan-v41-receipt.json','fill-v41-instrumentation.patch','serving-round49-analysis*','fill-v41-analysis*','serving-screen-round49.log']:
 files.update(r.glob(pattern))
files.update([r/'blender-keep-stopped-public.json',r/'variable-candidate-v40/build.json',r/'variable-candidate-v41/build.json',r/'fill-diagnostic-v41/build.json'])
manifest={'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
(r/'serving-round49-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'serving-round49-export.tar.gz','w:gz') as tar:
 for p in sorted(files):tar.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(files),'archive_bytes':(r/'serving-round49-export.tar.gz').stat().st_size,'archive_sha256':sha(r/'serving-round49-export.tar.gz')}))
