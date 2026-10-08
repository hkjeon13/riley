"""Matched production-binary stability evidence; never promotes performance."""
import argparse,json,time,threading,signal,os,socket,http.client,hashlib,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from serving_token_client_v2 import TokenHttpClient,TokenReference,SSEFramer
from mixed_phase_v34 import mixed_phase
from frozen_helpers import sha,host,process_group_memory,ready,stop,quiet_host_gate
import frozen_helpers as helpers
ROOT=Path('/data/riley-serving-261007')
OUTPUT_FOR_FAILURE=None
CODE=Path(__file__).parent
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def metrics(port):
 c=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
 try:
  c.request('GET','/metrics');r=c.getresponse();b=r.read()
  assert r.status==200
  return {'time_ns':time.time_ns(),'raw_utf8':b.decode(),'json':json.loads(b)}
 finally:c.close()
def drained(port,out,label):
 observations=[];deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  m=metrics(port);observations.append(m);x=m['json']
  if all(x[k]==0 for k in ['active_requests','waiting_requests','kv_allocated_blocks']):
   write(out/(label+'-drain.json'),observations);return m
  time.sleep(.1)
 write(out/(label+'-drain.json'),observations);raise RuntimeError('ownership did not drain '+label)
def cancel(port,body,after_token,evidence):
 c=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
 encoded=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
 response=None;raw=b'';seen=False;frames=[];status=None;headers=[];error=None
 deadline=time.monotonic()+120;framer=SSEFramer()
 try:
  c.request('POST','/v1/completions',body=encoded,headers={'Content-Type':'application/json','Connection':'close'})
  transport=c.sock
  if after_token:
   response=c.getresponse();status=response.status;headers=response.getheaders()
   assert status==200,'cancel response status'
   assert 'text/event-stream' in response.getheader('Content-Type',''),'cancel response content type'
   while not seen:
    remaining=deadline-time.monotonic();assert remaining>0,'cancel token deadline'
    transport.settimeout(remaining)
    chunk=response.read1(4096)
    if not chunk:raise RuntimeError('EOF before cancellation token')
    raw+=chunk;assert len(raw)<=8*1024*1024,'cancel prefix bound'
    for payload,arrived in framer.feed(chunk,time.perf_counter_ns()):
     frames.append({'arrived_ns':arrived,'data':payload.decode()})
     assert payload!=b'[DONE]','DONE before cancellation token'
     value=json.loads(payload);assert isinstance(value,dict) and 'error' not in value
     assert value.get('model')=='g04-smol' and value.get('object')=='text_completion','cancel identity'
     choices=value.get('choices');assert isinstance(choices,list),'cancel choices'
     for choice in choices:
      tokens=choice.get('token_ids',[])
      assert isinstance(tokens,list) and all(type(x) is int and x>=0 for x in tokens),'cancel token shape'
      if tokens:seen=True
     if seen:break
 except BaseException as e:
  error={'type':type(e).__name__,'message':str(e)};raise
 finally:
  if response:response.close()
  c.close()
  write(evidence,{'request':body,'encoded_request_sha256':hashlib.sha256(encoded).hexdigest(),'closed_after_observed_token':seen,'HTTP_status':status,'headers':headers,'raw_decoded_HTTP_body_prefix_hex':raw.hex(),'frames':frames,'error':error,'cancel_time_ns':time.time_ns(),'phase_classification':'after-first-token' if seen else 'immediate-after-request-write; prefill phase unproven'})
 return {'closed_after_observed_token':seen,'raw_evidence':str(evidence),'phase_classification':'after-first-token' if seen else 'immediate-after-request-write; prefill phase unproven'}
def main():
 global OUTPUT_FOR_FAILURE
 a=argparse.ArgumentParser();a.add_argument('--case',required=True);a.add_argument('--engine',choices=['v52','candidate','vllm'],required=True);a.add_argument('--repeat',type=int,choices=[0,1],required=True);a.add_argument('--attempt',type=int,required=True);args=a.parse_args()
 plan=json.loads((CODE/'stability-plan.json').read_text())
 assert plan['status']=='FROZEN_FOR_EXECUTION','Draft cannot dispatch GPU work'
 assert args.case in plan['cases'] and args.attempt>0
 # Exact benchmark and independent raw replay must be terminal before any GPU use.
 for path in plan['prerequisite_receipts']:
  x=json.loads(Path(path).read_text());assert x['failure'] is None,path
 replay=json.loads((ROOT/'kernel-batch11-independent-analysis-attempt02/raw-replay.json').read_text())
 assert replay['matrix_complete'] and replay['raw_replay_passed']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 out=ROOT/('kernel-batch11-stability-'+args.case+'-'+args.engine+'-r%d-attempt%02d'%(args.repeat,args.attempt));out.mkdir();OUTPUT_FOR_FAILURE=out
 helpers.out=out
 for path,digest in plan['pins'].items():assert sha(path)==digest,path
 launch=json.loads((ROOT/'kernel-batch11-quiet-attempt02'/(args.case+'-r0-'+args.engine+'-launch.json')).read_text())
 argv=launch['argv'].copy();env=launch['env'].copy()
 assert sha(argv[0])==launch['binary_sha256']
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 if args.engine=='vllm':argv[argv.index('--port')+1]=str(port)
 else:
  argv[argv.index('--bind')+1]='127.0.0.1:'+str(port)
  env['RILEY_SHUTDOWN_METRICS_PATH']=str(out/'native-close.json')
 write(out/'preparation.json',{'plan':plan,'argv':argv,'env':env,'original_matched_launch':launch,'performance_qualification':False,'repeat':args.repeat,'internal_reserved_KV_and_outstanding_iterations':'requires separate source-owned harness evidence'})
 quiet_host_gate('startup');done=threading.Event();samples=[];observer_errors=[];records=[];p=None;observer=None
 try:
  with (out/'server.log').open('x') as log:
   p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   def observe():
    while not done.wait(1):
     try:
      h=host();samples.append({'host':h,'process_group_memory':process_group_memory(p.pid)})
      assert float(h['gpu'].split(',')[4].strip())<=20480,'observed GPU memory above20GiB'
     except BaseException as e:
      observer_errors.append({'time_ns':time.time_ns(),'type':type(e).__name__,'message':str(e)});return
   observer=threading.Thread(target=observe);observer.start();ready(p,port)
   concurrency=int(args.case.split('-')[0][1:]);workload=args.case.split('-')[1]
   data=json.loads((CODE/'fixtures.json').read_text())[workload];corpus=data['corpus']
   refs=[TokenReference('g04-smol',tuple(b['choices'][0]['prompt_token_ids']),tuple(b['choices'][0]['token_ids']),b['choices'][0]['text'],b['choices'][0]['finish_reason']) for b in data['responses']]
   with TokenHttpClient() as client:
    def request(index):
     i=index%len(corpus);c=corpus[i]
     row=client.request(port,{'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0},refs[i],streaming=True,mode='observe');row['corpus_id']=c['id'];return row
    quiet_host_gate('warmup')
    warm_rows,warm_account,warm_summary=mixed_phase(client,request,concurrency=concurrency,count=96,corpus_ids=[x['id'] for x in corpus],phase='warmup')
    write(out/'warmup-rows.json',warm_rows);write(out/'warmup-account.json',warm_account)
    assert warm_account['completed'] and warm_account['strict_reference_pass']
    quiet_host_gate('sustained')
    start=time.monotonic();round_index=0
    while time.monotonic()-start<plan['sustained_seconds']:
     rows,account,summary=mixed_phase(client,request,concurrency=concurrency,count=384,corpus_ids=[x['id'] for x in corpus],phase='retained')
     write(out/('sustained-%04d-rows.json'%round_index),rows);write(out/('sustained-%04d-account.json'%round_index),account)
     records.append({'round':round_index,'accounting':account,'summary':summary})
     assert account['completed'] and account['failed']==0 and account['strict_reference_pass']
     assert not observer_errors,'host or GPU monitoring failure'
     round_index+=1
    write(out/'sustained.json',{'elapsed_seconds':time.monotonic()-start,'records':records,'no_exclusions':True})
    if args.engine!='vllm':drained(port,out,'after-sustained')
    for mode in [False,True]:
     for trial in range(plan['cancellation_trials_each_mode']):
      c=corpus[trial%len(corpus)]
      label=('decode' if mode else 'early')+'-%02d'%trial
      result=cancel(port,{'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0,'stream':True,'return_token_ids':True,'stream_options':{'include_usage':True}},mode,out/(label+'-raw-cancel.json'))
      write(out/(label+'-cancel.json'),result)
      if args.engine!='vllm':drained(port,out,label)
      def preserved_request(index):
       try:return request(index)
       except BaseException as e:return {'index':index,'reference_match':False,'exception':{'type':type(e).__name__,'message':str(e)}}
      with ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(preserved_request,range(len(corpus))))
      write(out/(label+'-re-request.json'),rows)
      assert all(x['reference_match'] for x in rows),'re-request correctness'
      if args.engine!='vllm':drained(port,out,label+'-reuse')
 finally:
  if p:
   try:stop(p)
   finally:
    done.set()
    if observer:observer.join(timeout=15)
    write(out/'host-samples.json',samples);write(out/'observer-errors.json',observer_errors)
   assert not observer or not observer.is_alive(),'monitor failed to join'
   write(out/'process-exit.json',{'returncode':p.returncode,'GPU_after':host(),'forced_stop_is_not_success':p.returncode!=0})
 if args.engine!='vllm':
  x=json.loads((out/'native-close.json').read_text())
  assert x['active_requests']==x['waiting_requests']==x['kv_allocated_blocks']==0
  assert all(x['allocation'][k]==0 for k in ['device_live_count','device_live_bytes','pinned_live_count','pinned_live_bytes'])
 assert p.returncode==0
 assert not observer_errors,'host or GPU monitoring failure'
 write(out/'completion.json',{'failure':None,'sustained_and_reuse_pass':True,'prefill_phase_cancellation_proven':False,'reserved_KV_outstanding_iteration_proven':False,'all_stability_gates_complete':False,'goal_achieved':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUTPUT_FOR_FAILURE is not None and not (OUTPUT_FOR_FAILURE/'completion.json').exists():
   write(OUTPUT_FOR_FAILURE/'completion.json',{'failure':{'type':type(e).__name__,'message':str(e)},'all_stability_gates_complete':False,'goal_achieved':False,'no_exclusions':True})
  raise
