import os,subprocess,socket,time,json,urllib.request,struct,concurrent.futures,pathlib
r=pathlib.Path('/tmp/riley-opt-260912');out=r/'v3-http-v11';out.mkdir(exist_ok=True)
corpus=json.loads((r/'variable-corpus-v11/requests.json').read_text());cases=[]
with (r/'prefill-full-model-v11/requests.bin').open('rb') as f:
 for _ in range(struct.unpack('<I',f.read(4))[0]):
  n=struct.unpack('<I',f.read(4))[0];ids=list(struct.unpack('<'+'I'*n,f.read(4*n)));c=next(x for x in corpus if x['token_ids']==ids);cases.append(c)
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
cmd=[str(r/'prefill-shapes-target-v11/debug/riley'),'serve','--model','/data/riley-benchmark/20260827T051948Z-d7ad713a/model','--model-id','v3','--bind',f'127.0.0.1:{port}','--graph-numerics','variable-smol-v3','--execution-graph-policy','require','--max-active-sequences','1','--max-sequence-tokens','1024','--max-output-tokens','128','--batch-token-budget','73','--prefill-chunk-tokens','73','--sampling-backend','cpu','--shutdown-on-stdin']
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'
log=(out/'server.log').open('w');p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,text=True);responses=[]
base=f'http://127.0.0.1:{port}'
def send(c,stream=False):
 n=c['prompt_tokens'];limit={16:32,128:64,398:128}[n];body={'model':'v3','prompt':c['prompt'],'temperature':0,'max_tokens':limit,'return_token_ids':True,'stream':stream}
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as resp: raw=resp.read().decode()
 result={'prompt':n,'stream':stream,'raw':raw};
 if not stream:
  data=json.loads(raw);choice=data['choices'][0];tokens=choice['token_ids'];reference=json.loads((r/f'loaded-rope-fixture-v11/decode-tokens-{n}.json').read_text());assert tokens==reference[:len(tokens)],(n,tokens,reference)
  assert choice['prompt_token_ids']==c['token_ids'];assert data['usage']['completion_tokens']==len(tokens);result['generated']=len(tokens)
 else:
  chunks=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line!='data: [DONE]'];tokens=[t for chunk in chunks for choice in chunk['choices'] for t in choice.get('token_ids',[])];reference=json.loads((r/f'loaded-rope-fixture-v11/decode-tokens-{n}.json').read_text());assert tokens==reference[:len(tokens)] and tokens;assert 'data: [DONE]' in raw;result['generated']=len(tokens)
 return result
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
 for c in cases:responses.append(send(c))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:responses.extend(pool.map(send,cases))
 responses.append(send(cases[-1],True))
 (out/'responses.json').write_text(json.dumps(responses,indent=2));print('HTTP',[(x['prompt'],x['generated'],x['stream']) for x in responses],flush=True)
finally:
 if p.poll() is None:
  p.stdin.write('\n');p.stdin.flush()
  try:p.wait(timeout=60)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
 log.close();(out/'process.json').write_text(json.dumps({'command':cmd,'returncode':p.returncode,'responses':len(responses)},indent=2));print('SERVER_EXIT',p.returncode,flush=True)
assert p.returncode==0
