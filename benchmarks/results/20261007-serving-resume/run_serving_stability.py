"""Source-bound real HTTP sustained-load/cancel/re-request gate; no performance promotion."""
import argparse,concurrent.futures,gzip,hashlib,http.client,json,math,os,queue,signal,socket,struct,subprocess,threading,time
from pathlib import Path
from serving_token_client_v2 import TokenHttpClient,TokenReference
from mixed_phase_v34 import mixed_phase
P=argparse.ArgumentParser();P.add_argument('--launch',type=Path,required=True);P.add_argument('--fixtures',type=Path,required=True);P.add_argument('--workload',choices=['fixed','natural'],required=True);P.add_argument('--screen',type=Path,required=True);P.add_argument('--out',type=Path,required=True);P.add_argument('--steady-seconds',type=int,default=600);A=P.parse_args()
O=A.out;O.mkdir();L=json.loads(A.launch.read_text());screen=json.loads(A.screen.read_text())
assert screen['matrix_complete'] and screen['all_core_minimum_mean_screen'] is True,'full-core screen must pass before this candidate stability gate'
assert screen['pressure_policy']=='quiet','diagnostic-only screen cannot promote to stability qualification'
assert A.steady_seconds==600,'duration frozen before measurement'
fixture_bytes=A.fixtures.read_bytes();F=json.loads(fixture_bytes)[A.workload];corpus=F['corpus'];responses=F['responses'];refs=[]
for c,b in zip(corpus,responses):
 choice=b['choices'][0];refs.append(TokenReference('g04-smol',tuple(choice.get('prompt_token_ids',c.get('prompt_token_ids',[]))),tuple(choice['token_ids']),choice['text'],choice['finish_reason']))
argv=list(L['argv']);binary=Path(argv[0]);assert hashlib.sha256(binary.read_bytes()).hexdigest()==L['binary_sha256']==screen['input_pins'][str(binary)],'binary changed or unrelated matched screen'
assert hashlib.sha256(fixture_bytes).hexdigest() in screen['input_pins'].values(),'fixture changed since matched screen'
for flag,expected in [('--prefill-chunk-tokens',512 if A.workload=='natural' else 128),('--max-sequence-tokens',1024 if A.workload=='natural' else 160),('--max-output-tokens',128 if A.workload=='natural' else 32)]:assert int(argv[argv.index(flag)+1])==expected,'stability launch workload differs from matched screen'
C=int(argv[argv.index('--max-active-sequences')+1]);assert C in [1,8,16,32]
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
argv[argv.index('--bind')+1]=f'127.0.0.1:{port}';env=dict(L['env']);env.pop('RILEY_SHUTDOWN_METRICS_PATH',None)
audit=O/'c02';audit.mkdir()
argv.extend(['--c02-candidate-id','stability-candidate','--c02-configuration-profile','stable-default','--c02-startup-artifact',str(O/'startup.json'),'--c02-audit-dir',str(audit),'--c02-shutdown-artifact',str(audit/'shutdown.json')])
def write(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
def host():return {'time_ns':time.time_ns(),'GPU_compute':gpu(),'GPU_state':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']}}
def metrics(path="/metrics"):
 c=http.client.HTTPConnection('127.0.0.1',port,timeout=30)
 try:c.request('GET',path);r=c.getresponse();b=r.read();assert r.status==200;return json.loads(b)
 finally:c.close()
def idle(label,allocation=None):
 records=[];deadline=time.monotonic()+40
 while time.monotonic()<deadline:
  m=metrics();records.append({'time_ns':time.time_ns(),'metrics':m})
  if all(m[k]==0 for k in ['active_requests','waiting_requests','kv_allocated_blocks']):
   write(label+'-idle.json',records)
   if allocation is not None:assert m['allocation']==allocation,'idle native allocation growth/change'
   native=metrics('/v1/c02/metrics');write(label+'-c02-native.json',native)
   assert native['schema_version']=='riley.c02-capture-metrics.v2'
   assert native['request_states']['active']==native['request_states']['pending_requests']==0
   assert native['kv_blocks']['active']==native['kv_blocks']['reserved']==0
   assert native['quiescence']['completion_outbox']==native['quiescence']['outstanding_iterations']==0
   m['native']=native
   return m
  time.sleep(.1)
 write(label+'-idle.json',records);raise RuntimeError('live request/KV ownership not cleared '+label)
def request(client,index):
 k=index%len(corpus);c=corpus[k];r=client.request(port,{'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0},refs[k],streaming=True,mode='observe');r['corpus_id']=c['id'];return r

def continuous(client):
 q=queue.Queue(maxsize=2048);halt=threading.Event();lock=threading.Lock();state={'next':0,'stop':None,'written':0,'writer_error':None,'blocked_puts':0};errors=[]
 def writer():
  try:
   with gzip.open(O/'steady-rows.jsonl.gz','wt',compresslevel=1) as f:
    while True:
     row=q.get()
     if row is None:break
     f.write(json.dumps(row,separators=(',',':'))+'\n');state['written']+=1
  except BaseException as e:state['writer_error']=str(e);halt.set();client.abort_pending('evidence writer failed')
 t=threading.Thread(target=writer);t.start();barrier=threading.Barrier(C+1);deadline=None
 def worker(worker):
  barrier.wait(timeout=30)
  while not halt.is_set():
   with lock:
    if time.monotonic()>=deadline and state['stop'] is None:state['stop']=math.ceil(state['next']/len(corpus))*len(corpus)
    if state['stop'] is not None and state['next']>=state['stop']:return
    index=state['next'];state['next']+=1
   try:
    row=request(client,index);row.update(index=index,worker_id=worker,phase='steady')
    while True:
     if state['writer_error']:
      write(f'emergency-row-{index}.json',row);raise RuntimeError('raw writer failed')
     try:q.put(row,timeout=1);break
     except queue.Full:state['blocked_puts']+=1
    assert row['status']=='success' and row['reference_match'] and row['transport_complete'],'steady HTTP/reference failure'
   except BaseException as e:
    with lock:errors.append({'index':index,'type':type(e).__name__,'message':str(e)})
    halt.set();client.abort_pending('steady failure');return
 start=time.perf_counter_ns();deadline=time.monotonic()+A.steady_seconds
 with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool:
  futures=[pool.submit(worker,i) for i in range(C)];barrier.wait(timeout=30)
  for f in futures:f.result()
 if not state['writer_error']:q.put(None);t.join(timeout=60)
 else:t.join(timeout=60)
 end=time.perf_counter_ns();receipt={'started_ns':start,'finished_ns':end,'duration_ns':end-start,'state':state,'errors':errors,'row_format':'every completed response with complete raw SSE frames; gzip JSONL; emergency rows preserved on writer failure','claim':'stability workload only; raw delivery metrics do not replace matched serving benchmark'};write('steady-accounting.json',receipt)
 assert not errors and not t.is_alive() and not state['writer_error'],'steady/raw evidence failure'
 assert state['written']==state['next'] and state['next']%len(corpus)==0,'steady requests omitted or unbalanced'
 assert end-start>=A.steady_seconds*10**9,'sustained interval too short'

def cancel_request(index,stage):
 k=index%len(corpus);c=corpus[k];conn=http.client.HTTPConnection('127.0.0.1',port,timeout=30);response=None;frames=[];started=time.perf_counter_ns()
 payload={'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0,'stream':True,'stream_options':{'include_usage':True},'return_token_ids':True}
 try:
  conn.request('POST','/v1/completions',body=json.dumps(payload),headers={'Content-Type':'application/json'});owned_sock=conn.sock;response=conn.getresponse();assert response.status==200
  if stage=='after_first_token':
   buffer=b'';seen=[]
   while not seen:
    chunk=response.read1(65536);assert chunk,'stream ended before cancellation';buffer+=chunk
    while b'\n\n' in buffer:
     frame,buffer=buffer.split(b'\n\n',1);data=b'\n'.join(line[6:] for line in frame.splitlines() if line.startswith(b'data: '))
     if not data:continue
     assert data!=b'[DONE]','request finished before cancellation'
     value=json.loads(data);frames.append({'arrived_ns':time.perf_counter_ns(),'data':data.decode()})
     assert 'error' not in value
     for choice in value.get('choices',[]):seen.extend(choice.get('token_ids') or [])
   assert seen==list(refs[k].output_token_ids[:len(seen)]) and len(seen)<len(refs[k].output_token_ids),'cancelled prefix differs or generation already complete'
  sock=owned_sock;assert sock is not None and sock.fileno()>=0,'no owned cancellation socket'
  sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));response.close();conn.close()
  return {'index':index,'stage':stage,'started_ns':started,'cancelled_ns':time.perf_counter_ns(),'corpus_id':c['id'],'http_status':response.status,'frames':frames,'closed_owned_connection':True}
 finally:
  if response is not None:response.close()
  conn.close()

failure=None;process=None;done=threading.Event();host_samples=[]
write('preparation.json',{'launch':L,'argv':argv,'binary_sha256':L['binary_sha256'],'fixtures_sha256':hashlib.sha256(fixture_bytes).hexdigest(),'screen_sha256':hashlib.sha256(A.screen.read_bytes()).hexdigest(),'concurrency':C,'workload':A.workload,'steady_seconds':600,'cancel_cycles':5,'cancel_stages':['after_headers_before_consuming_tokens','after_first_token'],'no_retry_or_sample_exclusion':True,'screen_serving_metrics':'from independently verified matched benchmark; this controller does not promote performance'})
try:
 verified_pins={}
 for name,expected in screen['input_pins'].items():
  h=hashlib.sha256()
  with Path(name).open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  verified_pins[name]=h.hexdigest()
  assert h.hexdigest()==expected,'matched screen input changed: '+name
 write('matched-input-pin-recheck.json',verified_pins)
 assert not gpu(),'GPU overlap'
 with (O/'server.log').open('x') as log:
  process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  def monitor():
   while not done.wait(1):host_samples.append(host())
  monitor_thread=threading.Thread(target=monitor);monitor_thread.start()
  deadline=time.monotonic()+300
  while True:
   try:m=metrics();break
   except (OSError,AssertionError,http.client.HTTPException):
    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('stability server startup failed')
    time.sleep(.1)
  with TokenHttpClient() as client:
   rows,account,_=mixed_phase(client,lambda i:request(client,i),concurrency=C,count=96,phase='warmup',corpus_ids=[c['id'] for c in corpus]);write('warmup-rows.json',rows);write('warmup-accounting.json',account);assert account['strict_reference_pass']
   baseline=idle('initial');allocation=baseline['allocation'];assert allocation['device_live_count']>0 and allocation['device_live_bytes']>0,'native gauges absent'
   continuous(client);idle('after-steady',allocation)
   for cycle in range(5):
    for stage in ['after_headers_before_consuming_tokens','after_first_token']:
     before=idle(f'cycle{cycle}-{stage}-before',allocation)
     with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool:
      rr=list(pool.map(lambda i:cancel_request(i,stage),range(C)))
     write(f'cycle{cycle}-{stage}-cancelled.json',rr);after=idle(f'cycle{cycle}-{stage}-after',allocation)
     assert after['counters']['disconnects']-before['counters']['disconnects']>=C,'disconnects not witnessed by server'
     assert after['native']['request_states']['cancelled']-before['native']['request_states']['cancelled']>=C,'source-owned scheduler cancellations not witnessed'
     rows,account,_=mixed_phase(client,lambda i:request(client,i),concurrency=C,count=96,phase='re-request',corpus_ids=[c['id'] for c in corpus]);write(f'cycle{cycle}-{stage}-rerequest-rows.json',rows);write(f'cycle{cycle}-{stage}-rerequest-accounting.json',account);assert account['strict_reference_pass'];idle(f'cycle{cycle}-{stage}-rerequest-idle',allocation)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:
 if process is not None:
  if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=40)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10);failure=failure or {'type':'shutdown','message':'graceful timeout; forced own process'}
  done.set();monitor_thread.join(timeout=5);write('host.json',host_samples)
  deadline=time.monotonic()+40
  while gpu() and time.monotonic()<deadline:time.sleep(1)
  artifact=audit/'shutdown.json';marker=audit/'shutdown.json.complete'
  doc=json.loads(artifact.read_text()) if artifact.exists() else None
  final=doc['final_metrics'] if doc else None
  completion_marker=json.loads(marker.read_text()) if marker.exists() else None
  bound=completion_marker is not None and completion_marker['artifact_filename']=='shutdown.json' and completion_marker['artifact_sha256']==hashlib.sha256(artifact.read_bytes()).hexdigest()
  reclaimed=not gpu() and bound and final is not None and final['request_states']['active']==final['request_states']['pending_requests']==0 and final['kv_blocks']['active']==final['kv_blocks']['reserved']==0 and all(v==0 for v in final['allocation'].values()) and final['quiescence']['completion_outbox']==final['quiescence']['outstanding_iterations']==final['quiescence']['riley_owned_live_allocations']==0 and final['quiescence']['worker_accepting'] is False and final['quiescence']['scheduler_accepting'] is False
  if not reclaimed:failure=failure or {'type':'resource_reclamation','message':'native final gauges/process GPU ownership not zero'}
 write('completion.json',{'failure':failure,'native_shutdown_reclaimed':locals().get('reclaimed',False),'screen_performance_adoption':'pending independent all-core screen plus this gate for every case','goal_achieved':False})
