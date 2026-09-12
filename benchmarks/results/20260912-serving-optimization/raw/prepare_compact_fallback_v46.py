from pathlib import Path
r=Path('/tmp/riley-opt-260912')
s=(r/'run_v4_http_v44_c32.py').read_text();prefix=s[:s.index('def send(c,stream=False):')];prefix=prefix.replace("out=r/'v4-http-v44-c32-final'","out=r/('v4-fallback-v46-'+os.environ['V46_BACKEND'])").replace("'--sampling-backend','cpu'","'--sampling-backend',os.environ['V46_BACKEND']")
body='''
def send(i):
 c=cases[i%3];body={'model':'v3','prompt':c['prompt'],'temperature':0.7 if i%2 else 0,'top_p':0.9,'seed':1234+i,'max_tokens':32,'return_token_ids':True}
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as resp:data=json.load(resp)
 choice=data['choices'][0];assert choice['prompt_token_ids']==c['token_ids'];assert len(choice['token_ids'])==data['usage']['completion_tokens']
 return {'index':i,'request':body,'tokens':choice['token_ids'],'text':choice['text'],'finish_reason':choice['finish_reason']}
try:
 deadline=time.monotonic()+120
 while True:
  if p.poll() is not None:raise RuntimeError('server stopped during startup')
  try:
   with urllib.request.urlopen(base+'/v1/models',timeout=1) as res:assert res.status==200
   break
  except (OSError,urllib.error.URLError):
   if time.monotonic()>deadline:raise
   time.sleep(.2)
 for i in range(6):responses.append(send(i))
 import threading
 barrier=threading.Barrier(16)
 def concurrent_send(i):barrier.wait();return send(i)
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:responses.extend(pool.map(concurrent_send,range(16)))
 (out/'responses.json').write_text(json.dumps(responses,indent=2));print('FALLBACK_HTTP',len(responses),flush=True)
finally:
 if p.poll() is None:
  p.stdin.write('\\n');p.stdin.flush()
  try:p.wait(timeout=60)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
 log.close();(out/'process.json').write_text(json.dumps({'command':cmd,'returncode':p.returncode,'responses':len(responses)},indent=2));print('SERVER_EXIT',p.returncode,flush=True)
assert p.returncode==0
'''
(r/'run_compact_fallback_lane_v46.py').write_text(prefix+body)
