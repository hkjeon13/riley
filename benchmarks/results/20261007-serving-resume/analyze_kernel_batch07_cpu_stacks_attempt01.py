"""Actual retained CPU samples and matched sched-in/out spans, no wait inference."""
import collections,hashlib,json,pathlib,sqlite3,time
from analyze_profile_intervals import merged,length
R=pathlib.Path(__file__).resolve().parent
C=R/'kernel-batch07-cpu-capture-all8-profile-independent-collection-attempt01'
P=R/'kernel-batch07-cpu-capture-all8-profile-phase-analysis-attempt01/summary.json'
O=R/'kernel-batch07-cpu-stack-analysis-attempt01';O.mkdir()
phase=json.loads(P.read_bytes());receipt=json.loads((C/'kernel-batch07-cpu-capture-all8-profile-attempt01-receipt.json').read_bytes())
reports=[]
for lane in phase['reports']:
 folder=C/'kernel-batch07-cpu-capture-all8-profile-attempt01'/lane['case'];path=folder/'trace.sqlite'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==lane['sqlite_sha256']
 owner=json.loads((folder/'owned-target-shutdown.json').read_bytes())['owned_pids'][0]
 with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as d:
  names=dict(d.execute('select id,value from StringIds'))
  proc=d.execute('select globalPid,pid,name from PROCESSES where pid=?',(owner,)).fetchall();assert len(proc)==1
  globalpid=proc[0][0];assert globalpid & ((1<<24)-1)==0
  lo,hi=globalpid,globalpid+(1<<24)
  counts={t:d.execute('select count(*) from '+t).fetchone()[0] for t in ['COMPOSITE_EVENTS','SAMPLING_CALLCHAINS','SCHED_EVENTS','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION','CUPTI_ACTIVITY_KIND_OVERHEAD','CUDA_GRAPH_NODE_EVENTS']}
  phases=[]
  for ph in lane['phases']:
   a,b=ph['conservative_trace_window_ns'];samples=d.execute('select id,start,globalTid from COMPOSITE_EVENTS where start>=? and start<=? and globalTid>=? and globalTid<?',(a,b,lo,hi)).fetchall()
   ids={s[0] for s in samples};leaf=collections.Counter();inclusive=collections.Counter();threads=collections.Counter(s[2] for s in samples)
   for sample,symbol,module,depth,unresolved in d.execute('select id,symbol,module,stackDepth,unresolved from SAMPLING_CALLCHAINS'):
    if sample not in ids:continue
    key=(names.get(symbol,str(symbol)),names.get(module,str(module)),bool(unresolved))
    inclusive[key]+=1
    if depth==0:leaf[key]+=1
   def ranking(counter):return [{'symbol':k[0],'module':k[1],'unresolved':k[2],'sample_frames':v,'percent_of_samples':100*v/max(len(samples),1)} for k,v in counter.most_common(40)]
   # Pair within each target thread; unmatched and boundary-crossing intervals
   # remain accounted separately, never converted into blocked-time estimates.
   sched=collections.defaultdict(list)
   for t,cpu,isin,tid,state in d.execute('select start,cpu,isSchedIn,globalTid,threadState from SCHED_EVENTS where globalTid>=? and globalTid<? order by start',(lo,hi)):sched[tid].append((t,cpu,isin,state))
   spans=[];per_thread=[];ambiguous=0;crossing=0
   for tid,events in sched.items():
    active=None;run=[];issues=0
    for t,cpu,isin,state in events:
     if isin:
      if active is not None:issues+=1
      active=(t,cpu)
     elif active is None:issues+=1
     else:
      start,core=active;active=None
      if core!=cpu or start>t:issues+=1;continue
      if start>=a and t<=b:run.append((start,t))
      elif start<b and t>a:crossing+=1
    if active is not None:issues+=1
    ambiguous+=issues;spans.extend(run)
    if run or threads[tid]:per_thread.append({'globalTid':tid,'samples':threads[tid],'wholly_contained_scheduled_ns':length(run),'paired_runs':len(run),'unmatched_or_invalid_events_full_capture':issues})
   phases.append({'phase':ph['phase'],'window_ns':[a,b],'target_samples':len(samples),'leaf_samples':sum(leaf.values()),'leaf_symbols':ranking(leaf),'inclusive_frames':ranking(inclusive),'threads':per_thread,'target_scheduled_work_sum_ns':length(spans),'target_scheduled_interval_union_ns':length(merged(spans)),'boundary_crossing_paired_runs_retained':crossing,'unmatched_or_invalid_schedule_events_full_capture':ambiguous})
  reports.append({'case':lane['case'],'sqlite_sha256':lane['sqlite_sha256'],'target_pid':owner,'globalPid':globalpid,'full_capture_counts':counts,'phases':phases})
result={'reports':reports,'source_archive_sha256':receipt['archive_sha256'],'phase_analysis_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'limits':['CPU IP samples are counts, not per-function durations; inclusive frames overlap.','Scheduled spans from matching same-thread same-CPU events are diagnostic; unmatched and boundary events are preserved.','Off-CPU gaps are not assigned to host wait, scheduler, HTTP or CUDA without causality.','UTC/perf mapping conditional on observed clock brackets; hidden steps and profiler origin precision unmeasured.','Instrumentation root UID differs from serving UID; diagnostic only; no serving speed claims.'],'serving_performance':'미실행','goal_achieved':False}
(O/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
for r in reports:
 ph=next(p for p in r['phases'] if p['phase']=='retained');print(r['case'],ph['target_samples'],[(x['symbol'],round(x['percent_of_samples'],2)) for x in ph['leaf_symbols'][:8]])
