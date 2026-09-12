import concurrent.futures, hashlib, json, os, pathlib, socket, subprocess, sys
sys.path.insert(0,'/tmp/riley-opt-260912')
from batch7_http_check import wait_ready, complete, stop_owned
from tokenizers import Tokenizer
from serving_token_client_v2 import TokenHttpClient, TokenReference, run_phase
root=pathlib.Path('/tmp/riley-opt-260912')
out=root/'native-trace-v42'; out.mkdir()
binary=root/'variable-candidate-v42/riley'
model=pathlib.Path('/data/riley-benchmark/20260827T051948Z-d7ad713a/model')
base=pathlib.Path('/tmp/riley-g04-vllm-profile-260911')
request=json.loads((base/'request.json').read_text()); binding=json.loads((base/'native-binding.json').read_text())
tok=Tokenizer.from_file(str(model/'tokenizer.json'))
assert tok.encode(request['prompt']).ids == binding['input_token_ids']
expected=tok.decode(binding['generated_token_ids'],skip_special_tokens=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(root/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
from mixed_phase_v34 import mixed_phase
all_corpus=json.loads((root/'variable-corpus-v11/requests.json').read_text());corpus=[];refs=[]
for response in json.loads((root/'v3-http-v11-final/responses.json').read_text())[:3]:
 body=json.loads(response['raw']);choice=body['choices'][0];ids=choice['prompt_token_ids'];c=next(c for c in all_corpus if c['token_ids']==ids)
 corpus.append({'id':'natural-'+str(len(ids)),'prompt':c['prompt'],'prompt_token_ids':ids,'max_tokens':len(choice['token_ids'])});refs.append(body)

records=[]
for cap in (16,32):
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],env=env,text=True).strip()
 with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
 command=[str(binary),'serve','--model',str(model),'--model-id','g04-smol','--bind',f'127.0.0.1:{port}','--max-active-sequences',str(cap),'--max-sequence-tokens','1024','--max-output-tokens','128','--batch-token-budget','512','--prefill-chunk-tokens','512','--kv-blocks',str(cap*64),'--metadata-transport','synchronous','--execution-graph-policy','require','--graph-numerics','variable-smol-v4','--sampling-backend','cpu','--shutdown-on-stdin']
 command=['/data/cuda-12.8.1/bin/nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--output',str(out/f'c{cap}'),'--force-overwrite=false']+command
 (out/f'c{cap}-launch.json').write_text(json.dumps({'argv':command,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2))
 with (out/f'c{cap}.log').open('w') as log:
  p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.PIPE)
  try:
   wait_ready(p,port)
   with TokenHttpClient() as client:
    def one(i):
     item=corpus[i%len(corpus)];choice=refs[i%len(refs)]['choices'][0]
     ref=TokenReference('g04-smol',tuple(item['prompt_token_ids']),tuple(choice['token_ids']),choice['text'],choice['finish_reason'])
     row=client.request(port,{'model':'g04-smol','prompt':item['prompt'],'temperature':0,'max_tokens':item['max_tokens']},ref,streaming=True,mode='observe')
     row['corpus_id']=item['id'];return row
    rows,accounting,_=mixed_phase(client,one,concurrency=cap,count=96,phase='diagnostic',corpus_ids=[x['id'] for x in corpus])
    (out/f'c{cap}-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/f'c{cap}-accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
    assert accounting['completed'] and accounting['strict_reference_pass'], accounting
   p.stdin.write(b'\n');p.stdin.flush(); assert p.wait(timeout=60)==0
   records.append({'capacity':cap,'requests':96,'streaming_requests':96,'exact_reference_text_and_token_ids':accounting['strict_reference_pass'],'exit_code':p.returncode})
  finally:stop_owned(p)
receipt={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'numerical_observation_only':True,'records':records}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
