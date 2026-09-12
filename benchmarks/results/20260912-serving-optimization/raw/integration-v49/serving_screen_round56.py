import hashlib,json,os,pathlib,signal,socket,subprocess,sys,time
root=pathlib.Path('/tmp/riley-opt-260912');sys.path.insert(0,str(root))
from serving_token_client_v2 import TokenHttpClient,TokenReference
from batch7_http_check import wait_ready
from mixed_phase_v34 import mixed_phase
out=root/'variable-serving-screen-round56';out.mkdir()
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def option(a,k,v):a[a.index(k)+1]=str(v)
plan=json.loads((root/'token-serving-round16-plan-v2.json').read_text())
for name in ['variable-candidate-v48','variable-candidate-v49']:
 build=json.loads((root/name/'build.json').read_text());assert all(sha(p)==h for p,h in build['binaries'].items())
for lane in plan['lanes'].values():
 assert sha(lane['argv'][0])==plan['immutable_files'][lane['argv'][0]]
 assert all(sha(p)==h for p,h in lane['model_files'].items())
fixed=json.loads((root/'diverse-p128-correctness-v1/corpus.json').read_text());bodies=json.loads((root/'diverse-p128-correctness-v1/references.json').read_text())
fixed_refs=[TokenReference('g04-smol',tuple(c['prompt_token_ids']),tuple(b['choices'][0]['token_ids']),b['choices'][0]['text'],b['choices'][0]['finish_reason']) for c,b in zip(fixed,bodies)]
all_corpus=json.loads((root/'variable-corpus-v11/requests.json').read_text());natural=[];natural_refs=[]
for response in json.loads((root/'v3-http-v11-final/responses.json').read_text())[:3]:
 b=json.loads(response['raw']);choice=b['choices'][0];ids=choice['prompt_token_ids'];c=next(c for c in all_corpus if c['token_ids']==ids)
 natural.append({'id':'natural-'+str(len(ids)),'prompt':c['prompt'],'max_tokens':len(choice['token_ids'])})
 natural_refs.append(TokenReference('g04-smol',tuple(ids),tuple(choice['token_ids']),choice['text'],choice['finish_reason']))
env=os.environ.copy();env.update(plan['base_environment']);env.update(plan['lanes']['baseline']['env']);smi=plan['nvidia_smi']['path']
def gpu(query,kind='gpu'):return subprocess.check_output([smi,'--query-'+kind+'='+query,'--format=csv,noheader,nounits'],env=env,text=True).strip()
def idle():
 assert not gpu('pid','compute-apps'),'foreign compute process present'
 deadline=time.monotonic()+120
 while int(gpu('temperature.gpu'))>48:
  assert time.monotonic()<deadline,'cooldown timeout';time.sleep(1)
def stop(p):
 if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=40)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
write('preparation.json',{'source_commit':json.loads((root/'variable-candidate-v49/build.json').read_text())['source_commit'],'blender_policy':'leave stopped; no stop or restore helper','controller_sha256':sha(__file__),'client_sha256':sha(root/'serving_token_client_v2.py'),'mixed_client_sha256':sha(root/'mixed_phase_v34.py'),'build_sha256':{n:sha(root/n/'build.json') for n in ['variable-candidate-v48','variable-candidate-v49']},'workloads':{'fixed':fixed,'natural':natural},'reference_sha256':{'fixed':[x.sha256 for x in fixed_refs],'natural':[x.sha256 for x in natural_refs]},'concurrency':[16,32],'active_capacity':'all admission equals client; V48 packed GPU32 atC16/C32, V49 mixed GPU32 atC16/C32','pairs':2,'retained_per_lane':384,'warmups':96,'qualification':False,'configuration_note':'both Riley budget512; chunk512 natural,128 fixed; both Riley active capacity equals client; V48 packed GPU32, V49 mixed GPU32 atC16/C32; both compact GPU greedy; vLLM budget512', 'scope':'C16/C32; V48 versus V49 mixed execution versus vLLM, Riley matched profile by concurrency; all admission capacities match clients, budget512, same external workload and KV budget'})
records=[]
try:
 for workload,concurrency in [(w,c) for c in (16,32) for w in ('fixed','natural')]:
  corpus,refs=(fixed,fixed_refs) if workload=='fixed' else (natural,natural_refs)
  for pair in [0,1]:
   order=['previous','new','vllm'];order=order if pair==0 else list(reversed(order))
   for name in order:
    idle();lane=plan['lanes']['vllm' if name=='vllm' else 'baseline']
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    argv=[v.format(port=port,concurrency=concurrency,vllm_budget=512) for v in lane['argv']]
    if name in ('previous','new'):
     argv[0]=str(root/('variable-candidate-v49/riley' if name=='new' else 'variable-candidate-v48/riley'));option(argv,'--graph-numerics','variable-smol-v7' if name=='new' else 'variable-smol-v6');option(argv,'--sampling-backend','gpu-greedy');option(argv,'--metadata-transport','synchronous')
    if name in ('previous','new'):
     option(argv,'--max-active-sequences',concurrency);option(argv,'--kv-blocks',10*concurrency)
    if name in ('previous','new'):
     option(argv,'--batch-token-budget',512);option(argv,'--prefill-chunk-tokens',512 if workload=='natural' else 128)
    if workload=='natural':
     if name=='vllm':option(argv,'--max-model-len',1024)
     else:option(argv,'--max-sequence-tokens',1024);option(argv,'--max-output-tokens',128);option(argv,'--kv-blocks',64*concurrency)
    active_option='--max-num-seqs' if name=='vllm' else '--max-active-sequences'
    assert int(argv[argv.index(active_option)+1])==(concurrency)
    child=env.copy();child.update(lane['env']);child['VLLM_BATCH_INVARIANT']='0';prefix=f'c{concurrency}-{workload}-p{pair}-{name}'
    write(prefix+'-launch.json',{'argv':argv,'binary_sha256':sha(argv[0]),'gpu_before':gpu('uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem')})
    with (out/(prefix+'.log')).open('w') as log:
     p=subprocess.Popen(argv,env=child,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:
      wait_ready(p,port)
      with TokenHttpClient() as client:
       def request_one(index):
        i=index%len(corpus);c=corpus[i];row=client.request(port,{'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0},refs[i],streaming=True,mode='observe');row['corpus_id']=c['id'];return row
       for phase,count in [('warmup',96),('retained',384)]:
        rows,account,summary=mixed_phase(client,request_one,concurrency=concurrency,count=count,phase=phase,corpus_ids=[c['id'] for c in corpus])
        write(prefix+'-'+phase+'-rows.json',rows);write(prefix+'-'+phase+'-accounting.json',account)
        if not account['completed']:raise RuntimeError('incomplete token transport '+prefix)
        if name!='vllm' and not account['strict_reference_pass']:raise RuntimeError('Riley reference mismatch '+prefix)
        if phase=='retained':write(prefix+'-summary.json',summary);records.append({'case':prefix,'complete':True,'strict_reference_pass':account['strict_reference_pass'],'summary':summary})
     finally:stop(p);write(prefix+'-exit.json',{'exit_code':p.returncode,'gpu_after':gpu('uuid,temperature.gpu,power.draw')})
    write('progress.json',records);print(prefix+' complete',flush=True)
finally:
 write('completion.json',{'records':records,'restoration_requested':False,'performance_qualified':False})
