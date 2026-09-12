from pathlib import Path
import sqlite3,json,statistics,collections,hashlib,bisect
r=Path('/tmp/riley-opt-260912/fixed-trace-v46');result={}
def union_ns(intervals):
 total=0;last=-1
 for a,b in sorted(intervals):
  if b>last:total+=b-max(a,last);last=b
 return total
def summary(rows):
 return {key:statistics.median(x[key] for x in rows) for key in ['graph_us','kernel_us','copy_us','d2h_us','h2d_us','memset_us','idle_gap_us','kernel_span_us','kernel_count']}
for cap in [32]:
 c=sqlite3.connect('file:'+str(r/f'c{cap}.sqlite')+'?mode=ro',uri=True)
 copies=list(c.execute('select start,end,bytes,copyKind,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where graphNodeId>0 order by start'))
 starts=[x[0] for x in copies if x[2]==30848 and x[3]==1];assert starts
 kernels=list(c.execute('select k.start,k.end,s.value,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.graphNodeId>0 order by k.start'))
 sets=list(c.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMSET where graphNodeId>0 order by start'))
 groups=[];ki=ci=mi=0
 for i,start in enumerate(starts):
  end=starts[i+1] if i+1<len(starts) else 2**63-1;ks=[];cs=[];ms=[]
  for events,dest,which in [(kernels,ks,'k'),(copies,cs,'c'),(sets,ms,'m')]:
   pos={'k':ki,'c':ci,'m':mi}[which]
   while pos<len(events) and events[pos][0]<start:pos+=1
   while pos<len(events) and events[pos][0]<end:dest.append(events[pos]);pos+=1
   if which=='k':ki=pos
   elif which=='c':ci=pos
   else:mi=pos
  assert ks and len([x for x in cs if x[2]==2048 and x[3]==2])==1
  end=max(x[1] for x in cs);assert all(x[1]<=end for x in ks+ms)
  stage='decode' if any('riley_shared16_model::embedding' in x[2] for x in ks) else 'prefill'
  spans=[x[:2] for x in ks+cs+ms]
  groups.append({'stage':stage,'graph_us':(end-start)/1000,'kernel_us':sum(x[1]-x[0] for x in ks)/1000,'kernel_span_us':(ks[-1][1]-ks[0][0])/1000,'copy_us':sum(x[1]-x[0] for x in cs)/1000,'d2h_us':sum(x[1]-x[0] for x in cs if x[3]==2)/1000,'h2d_us':sum(x[1]-x[0] for x in cs if x[3]==1)/1000,'memset_us':sum(x[1]-x[0] for x in ms)/1000,'idle_gap_us':((end-start)-union_ns(spans))/1000,'kernel_count':len(ks),'kernels':ks})
 assert sum(x['stage']=='prefill' for x in groups)==96
 middle=groups[len(groups)//10:len(groups)*9//10];stages={}
 for stage in ['prefill','decode']:
  chosen=[x for x in middle if x['stage']==stage];agg=collections.defaultdict(list)
  for x in chosen:
   for a,b,n,_ in x['kernels']:agg[n].append((b-a)/1000)
  ranking=[{'kernel':n,'count':len(t),'mean_us':statistics.mean(t),'total_ms':sum(t)/1000} for n,t in sorted(agg.items(),key=lambda q:-sum(q[1]))]
  stages[stage]={'all_submissions':sum(x['stage']==stage for x in groups),'middle_submissions':len(chosen),'median':summary(chosen),'kernels':ranking}
 runtime={}
 for name in ['cudaGraphLaunch','cudaStreamSynchronize']:
  times=[(b-a)/1000 for a,b in c.execute('select r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like ? and r.start>=? and r.end<=?',(name+'%',starts[len(starts)//10],starts[len(starts)*9//10]))]
  runtime[name]={'count':len(times),'median_us':statistics.median(times),'p95_us':sorted(times)[int(len(times)*.95)]}
 result[str(cap)]={'stages':stages,'runtime_api_window':runtime,'trace_sha256':hashlib.sha256((r/f'c{cap}.nsys-rep').read_bytes()).hexdigest()}
x={'scope':'V46 compact V4 fixed P128 mixed-output96 requests each atC16/C32, node-level Nsight trace. Middle80 replay window. Runtime API durations include profiler overhead and are not serving timing. GPU span separates kernel, copy, memset and unoccupied gaps; host CPU validation/sampling excluded.','cases':result}
(r/'analysis.json').write_text(json.dumps(x,indent=2)+'\n')
for cap,q in result.items():
 print(cap,q['runtime_api_window'])
 for stage,d in q['stages'].items():
  print(stage,d['all_submissions'],d['median']);print(json.dumps(d['kernels'][:6],indent=2))
