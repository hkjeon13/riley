"""Correctness-only release HTTP check. No latency/throughput measurements."""
import hashlib,http.client,json,os,socket,subprocess,time
from pathlib import Path
from tokenizers import Tokenizer
r=Path('/tmp/riley-g04-vllm-profile-260911');r.mkdir(exist_ok=True)
binary=Path('/tmp/riley-g04-vllm-profile-target/release/riley');model='/data/riley-benchmark/20260827T051948Z-d7ad713a/model'
ref=json.loads(Path('/tmp/riley-g04-serving-trace-260911/invariance.json').read_text())['before'];tokenizer=Tokenizer.from_file(model+'/tokenizer.json');expected=tokenizer.decode(ref,skip_special_tokens=True)
prompt='Hello'*128
assert len(tokenizer.encode(prompt,add_special_tokens=True).ids)==128
results=[]
def send(port,prompt=prompt,outputs=32,stream=False):
 c=http.client.HTTPConnection('127.0.0.1',port,timeout=120);c.request('POST','/v1/completions',json.dumps({'model':'g04-smol','prompt':prompt,'temperature':0,'max_tokens':outputs,'stream':stream}),{'Content-Type':'application/json'});return c,c.getresponse()
def complete(port,stream=False):
 c,res=send(port,stream=stream);assert res.status==200,(res.status,res.read())
 if not stream:
  obj=json.loads(res.read());assert obj['usage']['prompt_tokens']==128 and obj['usage']['completion_tokens']==32;assert obj['choices'][0]['finish_reason']=='length';text=obj['choices'][0]['text']
 else:
  text='';done=False;finish=None
  for line in res:
   if not line.startswith(b'data: '):continue
   data=line[6:].strip()
   if data==b'[DONE]':done=True;break
   for choice in json.loads(data).get('choices',[]):
    text+=choice.get('text','');finish=choice.get('finish_reason') or finish
  assert done and finish=='length'
 c.close();assert text==expected,(text,expected);return text
for sampling in ['cpu','gpu-greedy']:
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 argv=[str(binary),'serve','--model',model,'--model-id','g04-smol','--bind',f'127.0.0.1:{port}','--max-active-sequences','1','--max-waiting-requests','4','--batch-token-budget','1','--prefill-chunk-tokens','1','--max-sequence-tokens','160','--max-output-tokens','32','--kv-blocks','10','--residual-rmsnorm','separate','--execution-completion','iteration-batch','--metadata-transport','packed-async','--execution-graph-policy','require','--graph-numerics','vllm-smol-p128-v1','--sampling-backend',sampling]
 env=os.environ.copy();env['LD_LIBRARY_PATH']='/data/riley-g04-cuda13/lib'
 with (r/f'http-{sampling}.log').open('w') as log:
  process=subprocess.Popen(argv,stdout=log,stderr=log,env=env,cwd='/tmp/riley-g04-vllm-profile-source-260911')
  try:
   ready=False;deadline=time.monotonic()+120
   while process.poll() is None and time.monotonic()<deadline:
    try:
     c=http.client.HTTPConnection('127.0.0.1',port,timeout=1);c.request('GET','/v1/models');res=c.getresponse();res.read();c.close()
     if res.status==200:ready=True;break
    except OSError:pass
    time.sleep(.1)
   assert ready,'server not ready'
   complete(port);complete(port,True)
   rejected=[]
   for bad,outputs in [('Hello'*127,32),('Hello'*129,31),(prompt,33)]:
    c,res=send(port,bad,outputs);body=res.read();assert 400<=res.status<500,(res.status,body);rejected.append(res.status);c.close()
   # Disconnect before the first token (prefill), then after a decoded token.
   for after_token in [False,True]:
    c,res=send(port,stream=True);assert res.status==200
    if after_token:
     for line in res:
      if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
       obj=json.loads(line[6:]);
       if any(x.get('text') for x in obj.get('choices',[])):break
    c.close();res.close();complete(port)
   process.terminate();process.wait(timeout=30);assert process.returncode==0
   results.append({'sampling':sampling,'full_text_exact':True,'streaming_exact':True,'invalid_request_statuses':rejected,'post_disconnect_reuse_exact':True,'argv':argv})
  finally:
   if process.poll() is None:process.terminate();process.wait(timeout=30)
result={'profile':'vllm-smol-p128-v1','binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'reference_tokens':ref,'results':results,'performance_trials':0}
(r/'http-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
