import concurrent.futures,hashlib,json,os,pathlib,signal,socket,subprocess,sys
r=pathlib.Path('/tmp/riley-opt-260912');sys.path.insert(0,str(r))
from batch7_http_check import wait_ready
from serving_token_client_v2 import TokenHttpClient,TokenReference,overlap_peak
out=r/'diverse-vllm-observation-v1';out.mkdir();prior=r/'diverse-p128-correctness-v1'
corpus=json.loads((prior/'corpus.json').read_text());refs=json.loads((prior/'references.json').read_text());plan=json.loads((r/'token-serving-round16-plan-v2.json').read_text());lane=plan['lanes']['vllm']
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
records=[]
for capacity in [1,4]:
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 args=[v.format(port=port,concurrency=capacity,vllm_budget=128) for v in lane['argv']];env=os.environ.copy();env.update(plan['base_environment']);env.update(plan['lanes']['baseline']['env']);env.update(lane['env']);env['VLLM_BATCH_INVARIANT']='0'
 write(f'c{capacity}-launch.json',{'argv':args,'binary_sha256':hashlib.sha256(pathlib.Path(args[0]).read_bytes()).hexdigest(),'invariant':False})
 with (out/f'c{capacity}.log').open('w') as log:
  p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   wait_ready(p,port)
   with TokenHttpClient() as client:
    def one(i):
     item=corpus[i%12];choice=refs[i%12]['choices'][0];ref=TokenReference('g04-smol',tuple(item['prompt_token_ids']),tuple(choice['token_ids']),choice['text'],choice['finish_reason'])
     row=client.request(port,{'model':'g04-smol','prompt':item['prompt'],'max_tokens':item['max_tokens'],'temperature':0},ref,streaming=True,mode='observe');row['corpus_id']=item['id'];return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=capacity) as pool:rows=list(pool.map(one,range(12 if capacity==1 else 48)))
   write(f'c{capacity}-rows.json',rows)
   assert all(x['protocol_valid'] and x['transport_complete'] for x in rows)
   records.append({'capacity':capacity,'requests':len(rows),'strict_matches':sum(x['reference_match'] for x in rows),'observed_overlap':overlap_peak(rows),'mismatches':[{'case':x['corpus_id'],'comparison':x['reference_comparison']} for x in rows if not x['reference_match']]})
  finally:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=40)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
 print(json.dumps(records[-1]),flush=True)
write('completion.json',{'records':records,'performance_claim':False,'observation_not_correctness_acceptance':True})
