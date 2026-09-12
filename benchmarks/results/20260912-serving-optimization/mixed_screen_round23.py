import hashlib,json,os,pathlib,signal,socket,subprocess,sys,time
root=pathlib.Path('/tmp/riley-opt-260912');sys.path.insert(0,str(root))
import remote_session_round23 as session
from serving_token_client_v2 import TokenHttpClient,TokenReference,run_phase,summarize_phase
from batch7_http_check import wait_ready
from mixed_phase_v1 import mixed_phase
out=root/'mixed-p128-screen-round23';out.mkdir()
def write(name,x):(out/name).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
plan=json.loads((root/'token-serving-round16-plan-v2.json').read_text())
build=json.loads((root/'multisequence-candidate-v5/build.json').read_text())
assert all(sha(p)==v for p,v in build['binaries'].items())
build2=json.loads((root/'multisequence-candidate-v6/build.json').read_text())
assert all(sha(p)==v for p,v in build2['binaries'].items())
assert sha(root/'remote_session_round23.py')=='4a3f127a901e420b1fa2c4e7c1430916860fa56cab47cfda9eb3b7cbe0af56b1'
for lane in plan['lanes'].values():
 assert sha(lane['argv'][0])==plan['immutable_files'][lane['argv'][0]]
 assert all(sha(p)==v for p,v in lane['model_files'].items())
corpus=json.loads((root/'diverse-p128-correctness-v1/corpus.json').read_text())
references=json.loads((root/'diverse-p128-correctness-v1/references.json').read_text())
refs=[TokenReference('g04-smol',tuple(item['prompt_token_ids']),tuple(body['choices'][0]['token_ids']),body['choices'][0]['text'],body['choices'][0]['finish_reason']) for item,body in zip(corpus,references)]
assert len(corpus)==len(refs)==12
env=os.environ.copy();env.update(plan['base_environment']);env.update(plan['lanes']['baseline']['env'])
smi=plan['nvidia_smi']['path']
def gpu(query,kind='gpu'):
 return subprocess.check_output([smi,'--query-'+kind+'='+query,'--format=csv,noheader,nounits'],env=env,text=True).strip()
def idle():
 assert not gpu('pid','compute-apps'), 'foreign GPU compute process present'
 deadline=time.monotonic()+120
 while int(gpu('temperature.gpu'))>48:
  assert time.monotonic()<deadline,'GPU cooldown timeout';time.sleep(1)
def option(args,key,v):args[args.index(key)+1]=str(v)
def stop(p):
 if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=40)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
records=[];paused=False
write('preparation.json',{'candidate_build_sha256':sha(root/'multisequence-candidate-v5/build.json'),'new_build_sha256':sha(root/'multisequence-candidate-v6/build.json'),'helper_sha256':sha(root/'remote_session_round23.py'),'controller_sha256':sha(__file__),'purpose':'screening, not final qualification','concurrency':[1,4,8],'pairs':2,'retained':240,'warmups':24,'reference_sha256s':[ref.sha256 for ref in refs],'corpus_sha256':sha(root/'diverse-p128-correctness-v1/corpus.json'),'mixed_client_sha256':sha(root/'mixed_phase_v1.py')})
try:
 paused=True
 with (out/'session-stop.log').open('w') as log:subprocess.run([sys.executable,str(root/'remote_session_round23.py'),'stop'],stdout=log,stderr=subprocess.STDOUT,check=True)
 for concurrency in (1,4,8):
  for pair in (0,1):
   order=['baseline','candidate','new','vllm'] if pair==0 else ['vllm','new','candidate','baseline']
   for name in order:
    idle(); lane=plan['lanes']['baseline' if name in ('candidate','new') else name]
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    argv=[s.format(port=port,concurrency=concurrency,vllm_budget=128) for s in lane['argv']]
    if name in ('candidate','new'):
     argv[0]=str(root/('multisequence-candidate-v6/riley' if name=='new' else 'multisequence-candidate-v5/riley'));option(argv,'--max-active-sequences',1 if concurrency==1 else 4);option(argv,'--kv-blocks',10 if concurrency==1 else 40)
    child_env=env.copy();child_env.update(lane['env']);child_env['VLLM_BATCH_INVARIANT']='0';prefix=f'c{concurrency}-p{pair}-{name}'
    write(prefix+'-launch.json',{'argv':argv,'binary_sha256':sha(argv[0]),'gpu_before':gpu('uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem')})
    with (out/(prefix+'.log')).open('w') as log:
     p=subprocess.Popen(argv,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:
      wait_ready(p,port)
      with TokenHttpClient() as client:
       def request_one(index):
        item=corpus[index%12]
        row=client.request(port,{'model':'g04-smol','prompt':item['prompt'],'max_tokens':item['max_tokens'],'temperature':0},refs[index%12],streaming=True,mode='observe' if name=='vllm' else 'strict')
        row['corpus_id']=item['id'];return row
       for phase,count in [('warmup',24),('retained',240)]:
        rows,account,summary=mixed_phase(client,request_one,concurrency=concurrency,count=count,phase=phase,corpus_ids=[x['id'] for x in corpus])
        write(prefix+'-'+phase+'-rows.json',rows);write(prefix+'-'+phase+'-accounting.json',account)
        if not account['completed']:
         records.append({'case':prefix,'phase':phase,'complete':False,'account':account});break
        if phase=='retained':
         write(prefix+'-summary.json',summary);records.append({'case':prefix,'complete':True,'strict_reference_pass':account['strict_reference_pass'],'summary':summary})
     finally:stop(p);write(prefix+'-exit.json',{'exit_code':p.returncode,'gpu_after':gpu('uuid,temperature.gpu,power.draw')})
    write('progress.json',records);print(prefix+' completed',flush=True)
finally:
 if paused and session.SNAPSHOT.exists():
  with (out/'session-restore.log').open('w') as log:subprocess.run([sys.executable,str(root/'remote_session_round23.py'),'restore'],stdout=log,stderr=subprocess.STDOUT,check=True)
 write('completion.json',{'records':records,'restored':(session.ROOT/'verified.json').exists(),'performance_qualified':False})
