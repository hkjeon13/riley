"""Independent raw SSE and typed lifecycle audit. Mock checks never prove GPU stability."""
import argparse,collections,gzip,hashlib,json,pathlib,math,re
from audit_kernel_batch06_serving_contract import EXPECTED_ENV
from verify_kernel_batch06 import replay,decode,require,distribution

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def quiescent(native,final=False):
 require(native['schema_version']=='riley.c02-capture-metrics.v2','wrong source metrics schema')
 for group,keys in [('request_states',['active','pending_requests']),('kv_blocks',['active','reserved']),('quiescence',['completion_outbox','outstanding_iterations'])]:
  require(all(type(native[group][k]) is int and native[group][k]==0 for k in keys),'nonquiescent '+group)
 require(set(native['allocation'])=={'device_live_count','device_live_bytes','pinned_live_count','pinned_live_bytes'},'incomplete native allocation schema')
 require(all(type(v) is int and v>=0 for v in native['allocation'].values()),'invalid native allocation')
 if final:
  require(all(v==0 for v in native['allocation'].values()),'live final allocation')
  require(type(native['quiescence']['riley_owned_live_allocations']) is int and native['quiescence']['riley_owned_live_allocations']==0,'owned final allocations')
  require(native['quiescence']['worker_accepting'] is False and native['quiescence']['scheduler_accepting'] is False,'final admission still enabled')
def verify(directory,fixtures_path,screen_path):
 o=pathlib.Path(directory)
 def read(name):return decode((o/name).read_bytes())
 prep=read('preparation.json');completed=read('completion.json');screen=decode(pathlib.Path(screen_path).read_bytes());allfixtures=decode(pathlib.Path(fixtures_path).read_bytes())
 require(completed['failure'] is None and completed['native_shutdown_reclaimed'] is True,'actual controller failed or ownership uncleared')
 require(screen['matrix_complete'] and screen['all_core_minimum_mean_screen'] is True and screen['pressure_policy']=='quiet','all-core admitted screen missing')
 require(prep['screen_sha256']==sha(screen_path) and prep['fixtures_sha256']==sha(fixtures_path),'screen/fixture byte binding differs')
 require(prep['steady_seconds']==600 and prep['cancel_cycles']==5 and prep['cancel_stages']==['after_headers_before_consuming_tokens','after_first_token'] and prep['no_retry_or_sample_exclusion'] is True,'lifecycle scope differs')
 fixture=allfixtures[prep['workload']];corpus=fixture['corpus'];responses=fixture['responses'];C=prep['concurrency'];require(C in [1,8,16,32],'capacity differs')
 L=prep['launch'];a=L['argv'];require(L['env']==EXPECTED_ENV,'stability launch environment differs from admitted screen');actual=prep['argv'];base=list(a);bind=actual[actual.index('--bind')+1];require(re.fullmatch(r'127\.0\.0\.1:\d+',bind),'invalid owned bind');base[base.index('--bind')+1]=bind;require(actual[:len(base)]==base,'actual stability numerical/workload argv differs');tail=actual[len(base):];require(len(tail)==10 and tail[0:4]==['--c02-candidate-id','stability-candidate','--c02-configuration-profile','stable-default'] and tail[4]=='--c02-startup-artifact' and pathlib.Path(tail[5]).name=='startup.json' and tail[6]=='--c02-audit-dir' and pathlib.Path(tail[7]).name=='c02' and tail[8]=='--c02-shutdown-artifact' and pathlib.Path(tail[9]).name=='shutdown.json','actual audit flags differ');require(L['binary_sha256']==prep['binary_sha256']==screen['input_pins'][a[0]],'binary unbound to full screen')
 require(int(a[a.index('--max-active-sequences')+1])==C,'capacity/launch differs')
 for flag,value in [('--prefill-chunk-tokens',512 if prep['workload']=='natural' else 128),('--max-output-tokens',128 if prep['workload']=='natural' else 32),('--batch-token-budget',512)]:require(int(a[a.index(flag)+1])==value,'workload contract differs '+flag)
 require(read('matched-input-pin-recheck.json')==screen['input_pins'],'input recheck differs')
 identities=set();metrics=collections.defaultdict(list);total_requests=0
 def row_check(row,index,phase):
  nonlocal total_requests
  k=index%len(corpus);require(row['index']==index and row['phase']==phase and row['corpus_id']==corpus[k]['id'],'row identity/phase/corpus differs')
  measured,output,input_,exact=replay(row,corpus[k],responses[k],'candidate');ident=row['response_identity']['id'];require(ident not in identities,'reused response identity');identities.add(ident);total_requests+=1
  return measured,output,input_
 def phase(label,phase_name):
  rows=read(label+'-rows.json');account=read(label+'-accounting.json');require(account['completed'] and len(rows)==96 and account['succeeded']==96 and account['failed']==0 and account['unresolved_attempts']==account['not_started']==0,'incomplete phase '+label)
  require([r['index'] for r in rows]==list(range(96)),'phase dropped/duplicated index')
  counts=collections.Counter();events=[]
  for i,row in enumerate(rows):
   row_check(row,i,phase_name);counts[row['corpus_id']]+=1;require(account['phase_started_ns']<=row['call_started_ns']<=row['started_ns']<=row['finished_ns']<=row['call_finished_ns']<=account['phase_finished_ns'],'phase clock envelope')
   events.extend([(row['started_ns'],1),(row['finished_ns'],-1)])
  require(dict(counts)==account['actual_corpus_counts']==account['expected_corpus_counts'],'phase corpus imbalance')
  active=peak=0
  for _,delta in sorted(events):active+=delta;peak=max(peak,active)
  require(active==0 and peak==C==account['observed_max_request_in_flight'],'phase concurrency differs')
  return {'phase':label,'requests':96,'exact':96,'errors':0}
 phases=[phase('warmup','warmup')]
 initial=read('initial-idle.json')[-1]['metrics'];allocation=initial['allocation'];require(allocation['device_live_count']>0 and allocation['device_live_bytes']>0,'initial live gauges absent')
 def idle(label):
  rows=read(label+'-idle.json');require(rows,'missing idle observations');m=rows[-1]['metrics'];require(all(m[k]==0 for k in ['active_requests','waiting_requests','kv_allocated_blocks']),'generic request/KV not idle');require(m['allocation']==allocation,'idle allocation changed')
  n=read(label+'-c02-native.json');quiescent(n);require(n['allocation']==allocation,'typed allocation differs');return m,n
 idle('initial');idle('after-steady')
 account=read('steady-accounting.json');state=account['state'];require(not account['errors'] and state['writer_error'] is None,'steady failure/writer error')
 require(account['finished_ns']-account['started_ns']==account['duration_ns']>=600*10**9,'steady phase duration short')
 require(state['written']==state['next']==state['stop'] and state['next']%len(corpus)==0 and state['next']>0,'steady dropped/unbalanced counts')
 indexes=set();counts=collections.Counter();events=[];tokens=inputs=0;first=last=None
 with gzip.open(o/'steady-rows.jsonl.gz','rb') as f:
  for line in f:
   row=decode(line);i=row['index'];require(type(i) is int and 0<=i<state['next'] and i not in indexes,'steady omitted/duplicate/out-of-range index');indexes.add(i)
   values,out,inp=row_check(row,i,'steady');require(account['started_ns']<=row['call_started_ns']<=row['started_ns']<=row['finished_ns']<=row['call_finished_ns']<=account['finished_ns'],'steady clock envelope')
   for k,v in values.items():metrics[k].append(v/1e6)
   counts[row['corpus_id']]+=1;tokens+=out;inputs+=inp;events.extend([(row['started_ns'],1),(row['finished_ns'],-1)])
   first=row['started_ns'] if first is None else min(first,row['started_ns']);last=row['finished_ns'] if last is None else max(last,row['finished_ns'])
 require(len(indexes)==state['next'],'steady missing indexes');require(set(counts)=={c['id'] for c in corpus} and len(set(counts.values()))==1,'steady unbalanced corpus')
 active=peak=0
 for _,delta in sorted(events):active+=delta;peak=max(peak,active)
 require(active==0 and peak==C,'steady concurrent HTTP scope differs');require(last-first>=600*10**9,'actual steady HTTP interval shorter than600s')
 cancel_count=0
 for cycle in range(5):
  for stage in prep['cancel_stages']:
   label=f'cycle{cycle}-{stage}';before,bnative=idle(label+'-before');after,anative=idle(label+'-after');rr=read(label+'-cancelled.json');require(len(rr)==C and [r['index'] for r in rr]==list(range(C)),'cancel wave incomplete')
   for row in rr:
    i=row['index'];k=i%len(corpus);require(row['stage']==stage and row['corpus_id']==corpus[k]['id'] and row['http_status']==200 and row['closed_owned_connection'] is True,'cancellation scope differs');require(row['started_ns']<row['cancelled_ns'],'cancel time order')
    ids=[];previous=row['started_ns']
    for frame in row['frames']:
     require(previous<=frame['arrived_ns']<=row['cancelled_ns'],'cancel frame ordering');previous=frame['arrived_ns'];v=decode(frame['data']);require('error' not in v,'cancel error payload')
     for choice in v.get('choices',[]):ids.extend(choice.get('token_ids') or [])
    if stage=='after_first_token':require(0<len(ids)<len(responses[k]['choices'][0]['token_ids']) and ids==responses[k]['choices'][0]['token_ids'][:len(ids)],'cancel token prefix differs or already completed')
    else:require(not row['frames'],'before-consumption cancellation read frames')
    cancel_count+=1
   require(after['counters']['disconnects']-before['counters']['disconnects']>=C and anative['request_states']['cancelled']-bnative['request_states']['cancelled']>=C,'cancel not witnessed in service/source scheduler')
   phases.append(phase(label+'-rerequest','re-request'));idle(label+'-rerequest')
 shutdown=read('c02/shutdown.json');marker=read('c02/shutdown.json.complete');require(marker['artifact_filename']=='shutdown.json' and marker['artifact_sha256']==sha(o/'c02/shutdown.json'),'shutdown marker not bound');quiescent(shutdown['final_metrics'],final=True)
 return {'passed':True,'concurrency':C,'workload':prep['workload'],'screen_sha256':sha(screen_path),'binary_sha256':prep['binary_sha256'],'all_exact_HTTP_requests_replayed':total_requests,'cancelled_owned_connections_verified':cancel_count,'phases':phases,'steady':{'requests':state['next'],'input_tokens':inputs,'output_tokens':tokens,'actual_HTTP_interval_ns':last-first,'throughput_tokens_s':tokens/((last-first)/1e9),'descriptive_delivery_metrics_ms':{k:distribution(v) for k,v in metrics.items()},'errors':0,'concurrency_peak':peak},'native_final_reclaimed':True,'serving_performance':'stability diagnostic only; matched serving performance comes from independent all8 screen','memory_scope':'retained allocation equality plus zero typed final gauges; host/GPU peaks require separate sampled host analysis','adopted':False,'goal_achieved':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=pathlib.Path,required=True);p.add_argument('--fixtures',type=pathlib.Path,required=True);p.add_argument('--screen',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();result=verify(a.directory,a.fixtures,a.screen);a.output.open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','all_exact_HTTP_requests_replayed','cancelled_owned_connections_verified','native_final_reclaimed']}))
