from pathlib import Path
import json,hashlib,tarfile,subprocess,sys
r=Path('/tmp/riley-opt-260912');sys.path.insert(0,str(r));from serving_token_client_v2 import summarize_phase
reports={}
for c in (4,8,16,32):
 d=r/('natural-fill-v34' if c<=8 else 'natural-fill-v34-high');a=json.loads((d/f'c{c}-accounting.json').read_text());rows=json.loads((d/f'c{c}-rows.json').read_text());assert a['completed'] and a['strict_reference_pass'] and a['requested']==384 and a['observed_max_request_in_flight']==c
 reports[str(c)]=summarize_phase(rows,a)
(r/'fill-v34-client-analysis.json').write_text(json.dumps({'scope':'Nonexclusive instrumented V33; active=min(client,8), no matched vLLM comparison','clients':reports},indent=2)+'\n')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r/'prefill-shapes-source-v11')
assert hashlib.sha256((r/'prefill-shapes-target-v11/release/riley').read_bytes()).hexdigest()=='5ffceea0d2494f951cf3b92115ad04063bbadb8e5c45c3c5ffdf67a3cd8ec7a3'
names=['fill-v34-analysis.json','fill-v34-client-analysis.json','fill-v34-instrumentation.patch','fill-v34-release.log','fill-v34-run.log','fill-v34-high-run.log','fill-diagnostic-v34/build.json','mixed_phase_v34.py']
for directory in ['natural-fill-v34','natural-fill-v34-high']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file())
m={'base_commit':'214c8ed7309d00ea908c7e29e3c2ed962f8f945d','production_restored':True,'requests':1536,'reference_tokens':sum(z['successful_output_tokens'] for z in reports.values()) if all('successful_output_tokens' in z for z in reports.values()) else None,'files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}}
(r/'fill-v34-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
with tarfile.open(r/'fill-v34-export.tar.gz','w:gz') as t:
 for n in names+['fill-v34-manifest.json']:t.add(r/n,arcname=n)
for c,z in reports.items():print(c,{k:z[k] for k in ['successful_output_tokens_per_phase_wall_second','token_ttft_ms','token_tpot_ms','e2e_ms']})
