from pathlib import Path
import hashlib,json,subprocess,tarfile,datetime
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
x=json.loads((r/'blender-keep-stopped-public.json').read_text())
for item in x['processes']:
 assert not Path(f"/proc/{item['pid']}").exists()
 assert not subprocess.check_output(['ss','-H','-ltn','sport','=',str(item['port'])],text=True).strip()
x['verified_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(r/'blender-keep-stopped-public.json').write_text(json.dumps(x,indent=2)+'\n')
(r/'graph16-v40.patch').write_bytes(subprocess.check_output(['git','diff','e530ec7c01c96de2dc1db02c321da72e07b24a1c','HEAD'],cwd=src))
build=json.loads((r/'variable-candidate-v40/build.json').read_text());assert all(sha(Path(p))==h for p,h in build['binaries'].items())
assert sha(r/'prefill-shapes-target-v11/release/riley')==next(iter(build['binaries'].values()))
receipt={'source_commit':build['source_commit'],'documentation_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'build':build,'tests':{'owned_gpu':3,'owned_logits_rows':4896,'owned16_memcheck_errors':0,'recorder_memcheck_errors':0,'scheduler_cpu':35,'cli_cpu':3,'http_completed':37,'http_disconnects':1,'cleanup_allocation_zero':True},'retained_serving_requests':18432,'riley_reference_matches':12288,'diagnostic_requests':1536,'goal_achieved':False,'decision':'Keep V3 for low concurrency; V4 explicit high-concurrency candidate improves throughput/TPOT but remains below vLLM. No default replacement or long-term stability qualification.','test_binaries':{str(p):sha(p) for p in [r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1',r/'prefill-shapes-target-v11/release/deps/v4_shared_recorder_gpu-63c869c763789df9']}}
(r/'graph16-v40-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=set()
for dirname in ['variable-serving-screen-round48','natural-fill-v40','v4-http-v40-c32-final']:
 files.update(p for p in (r/dirname).rglob('*') if p.is_file())
for pattern in ['graph16-v40*.log','fill-v40*.log','graph16-v40.patch','graph16-v40-receipt.json','fill-v40-instrumentation.patch','serving-round48-analysis*','fill-v40-analysis*','serving-screen-round48.log']:
 files.update(r.glob(pattern))
files.update([r/'blender-keep-stopped-public.json',r/'variable-candidate-v40/build.json',r/'fill-diagnostic-v40/build.json'])
manifest={'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
(r/'serving-round48-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'serving-round48-export.tar.gz','w:gz') as tar:
 for p in sorted(files):tar.add(p,arcname=str(p.relative_to(r)))
print(json.dumps({'files':len(files),'archive_bytes':(r/'serving-round48-export.tar.gz').stat().st_size,'archive_sha256':sha(r/'serving-round48-export.tar.gz')}))
