import os,subprocess,socket,time,json,urllib.request,struct,concurrent.futures,pathlib
r=pathlib.Path('/tmp/riley-opt-260912');out=r/('v6-fallback-ordered-v48-'+os.environ['V46_BACKEND']);out.mkdir()
corpus=json.loads((r/'variable-corpus-v11/requests.json').read_text());cases=[]
with (r/'prefill-full-model-v11/requests.bin').open('rb') as f:
 for _ in range(struct.unpack('<I',f.read(4))[0]):
  n=struct.unpack('<I',f.read(4))[0];ids=list(struct.unpack('<'+'I'*n,f.read(4*n)));c=next(x for x in corpus if x['token_ids']==ids);cases.append(c)
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
cmd=[str(r/'prefill-shapes-target-v11/release/riley'),'serve','--model','/data/riley-benchmark/20260827T051948Z-d7ad713a/model','--model-id','v3','--bind',f'127.0.0.1:{port}','--graph-numerics','variable-smol-v6','--execution-graph-policy','require','--max-active-sequences','32','--kv-blocks','2048','--max-sequence-tokens','1024','--max-output-tokens','128','--batch-token-budget','1024','--prefill-chunk-tokens','512','--sampling-backend',os.environ['V46_BACKEND'],'--shutdown-on-stdin']
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
env['RILEY_SHUTDOWN_METRICS_PATH']=str(out/'shutdown.json')
log=(out/'server.log').open('w');p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,text=True);responses=[]
base=f'http://127.0.0.1:{port}'

def send(i,admitted=None):
 c=cases[i%3];body={'model':'v3','prompt':c['prompt'],'temperature':0.7 if i%2 else 0,'top_p':0.9,'seed':1234+i,'max_tokens':32,'return_token_ids':True,'stream':True}
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as resp:
  if admitted is not None:admitted.set()
  raw=resp.read().decode()
 chunks=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
 assert 'data: [DONE]' in raw
 ids={c['id'] for c in chunks};assert len(ids)==1
 choices=[choice for chunk in chunks for choice in chunk['choices']]
 return {'index':i,'request':body,'response_id':ids.pop(),'tokens':[t for choice in choices for t in choice.get('token_ids',[])],'text':''.join(choice.get('text','') for choice in choices),'finish_reason':next(choice['finish_reason'] for choice in choices if choice.get('finish_reason'))}

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
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  pending=[]
  for i in range(16):
   admitted=threading.Event();pending.append(pool.submit(send,i,admitted));assert admitted.wait(30),'headers not admitted'
  responses.extend(f.result() for f in pending)
 (out/'responses.json').write_text(json.dumps(responses,indent=2));print('FALLBACK_HTTP',len(responses),flush=True)
finally:
 if p.poll() is None:
  p.stdin.write('\n');p.stdin.flush()
  try:p.wait(timeout=60)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
 log.close();(out/'process.json').write_text(json.dumps({'command':cmd,'returncode':p.returncode,'responses':len(responses)},indent=2));print('SERVER_EXIT',p.returncode,flush=True)
assert p.returncode==0
