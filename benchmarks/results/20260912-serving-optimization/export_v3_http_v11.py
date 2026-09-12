from pathlib import Path
import hashlib,json,tarfile
r=Path('/tmp/riley-opt-260912');names=['v3-http-server-v11.patch','v3-http-binary-build-v11.log','v3-http-run-v11.log','v3-http-final-run-v11.log','v3-http-cpu-tests-v11.log','v3-http-cpu-private-tmp-v11.log','v3-http-cpu-private-umask-v11.log']
for d in ['v3-http-v11','v3-http-v11-final']:names.extend(str(p.relative_to(r)) for p in (r/d).iterdir() if p.is_file())
s=json.loads((r/'v3-http-v11-final/shutdown.json').read_text());assert all(s[x]==0 for x in ['active_requests','waiting_requests','kv_allocated_blocks']);assert all(x==0 for x in s['allocation'].values());res=json.loads((r/'v3-http-v11-final/responses.json').read_text());assert len(res)==8
summary={'completed_responses':len(res),'generated_tokens':sum(x['generated'] for x in res),'streaming_completions':sum(x['stream'] for x in res),'single_active_request':True,'queued_concurrent_submissions':3,'invalid_output_bound_http_status':400,'client_stream_disconnect_followed_by_correct_completion':True,'shutdown_all_allocations_zero':True,'shape_metrics_degraded':s['batch_shapes']['metrics_degraded'],'serving_performance_qualified':False}
(r/'v3-http-summary-v11.json').write_text(json.dumps(summary,indent=2));names.append('v3-http-summary-v11.json')
m={'commit':'13bd8fcb263f9980447fa3a4d8a2f908f6017299','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'binary_sha256':hashlib.sha256((r/'prefill-shapes-target-v11/debug/riley').read_bytes()).hexdigest(),'binary_scope':'debug HTTP correctness binary; final source adds CLI unit test and comments only'}
(r/'v3-http-server-v11-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
with tarfile.open(r/'v3-http-server-v11.tar.gz','w:gz') as t:
 for n in names+['v3-http-server-v11-manifest.json']:t.add(r/n,arcname=n)
print(json.dumps(summary));print('exported',len(names))
