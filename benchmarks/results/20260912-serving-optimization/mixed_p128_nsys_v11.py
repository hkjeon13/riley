import concurrent.futures, hashlib, json, os, pathlib, socket, subprocess, sys
sys.path.insert(0,'/tmp/riley-opt-260912')
from batch7_http_check import wait_ready, complete, stop_owned
from tokenizers import Tokenizer
from serving_token_client_v2 import TokenHttpClient, TokenReference, run_phase
root=pathlib.Path('/tmp/riley-opt-260912')
out=root/'mixed-p128-nsys-v11'; out.mkdir()
binary=root/'variable-candidate-v11/riley'
model=pathlib.Path('/data/riley-benchmark/20260827T051948Z-d7ad713a/model')
base=pathlib.Path('/tmp/riley-g04-vllm-profile-260911')
request=json.loads((base/'request.json').read_text()); binding=json.loads((base/'native-binding.json').read_text())
tok=Tokenizer.from_file(str(model/'tokenizer.json'))
assert tok.encode(request['prompt']).ids == binding['input_token_ids']
expected=tok.decode(binding['generated_token_ids'],skip_special_tokens=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(root/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
from mixed_phase_v1 import mixed_phase
corpus=json.loads((root/'diverse-p128-correctness-v1/corpus.json').read_text())
refs=json.loads((root/'diverse-p128-correctness-v1/references.json').read_text())
records=[]
for cap in (1,):
 with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
 command=[str(binary),'serve','--model',str(model),'--model-id','g04-smol','--bind',f'127.0.0.1:{port}','--max-active-sequences',str(cap),'--max-sequence-tokens','160','--max-output-tokens','32','--batch-token-budget','128','--prefill-chunk-tokens','128','--kv-blocks',str(cap*10),'--metadata-transport','packed-async','--execution-graph-policy','require','--graph-numerics','variable-smol-v3','--sampling-backend','cpu','--shutdown-on-stdin']
 command=['/data/cuda-12.8.1/bin/nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--output',str(out/f'c{cap}'),'--force-overwrite=false']+command
 with (out/f'c{cap}.log').open('w') as log:
  p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.PIPE)
  try:
   wait_ready(p,port)
   with TokenHttpClient() as client:
    def one(i):
     item=corpus[i%12];choice=refs[i%12]['choices'][0]
     ref=TokenReference('g04-smol',tuple(item['prompt_token_ids']),tuple(choice['token_ids']),choice['text'],choice['finish_reason'])
     row=client.request(port,{'model':'g04-smol','prompt':item['prompt'],'temperature':0,'max_tokens':item['max_tokens']},ref,streaming=True,mode='observe')
     row['corpus_id']=item['id'];return row
    rows,accounting,_=mixed_phase(client,one,concurrency=1,count=12,phase='diagnostic',corpus_ids=[x['id'] for x in corpus])
    (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
    assert accounting['completed'], accounting
   p.stdin.write(b'\n');p.stdin.flush(); assert p.wait(timeout=30)==0
   records.append({'capacity':cap,'requests':12,'streaming_requests':12,'exact_reference_text_and_token_ids':accounting['strict_reference_pass'],'exit_code':p.returncode})
  finally:stop_owned(p)
receipt={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'numerical_observation_only':True,'records':records}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
