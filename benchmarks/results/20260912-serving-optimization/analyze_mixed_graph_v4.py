import sqlite3,pathlib,json,statistics
p=pathlib.Path('/tmp/riley-opt-260912/mixed-p128-nsys-v4');c=sqlite3.connect('file:'+str(p/'c4.sqlite')+'?mode=ro',uri=True)
uploads=c.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where graphNodeId>0 and copyKind=1 and bytes in (1280,600) order by start').fetchall()
kernels=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.graphNodeId>0 order by k.start').fetchall()
copies=c.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where graphNodeId>0 order by start').fetchall()
rows=[]
for i,(a,b,size) in enumerate(uploads):
 z=uploads[i+1][0] if i+1<len(uploads) else 2**63-1
 ks=[k for k in kernels if a<=k[0]<z];ms=[m for m in copies if a<=m[0]<z]
 # Only one graph submission is in flight. Post-final shutdown has no graph activities.
 stage='prefill' if any(k[2].startswith('attention(') for k in ks) else 'decode'
 end=max([k[1] for k in ks]+[m[1] for m in ms]);head=[k for k in ks if 'gemvx' in k[2]]
 rows.append({'stage':stage,'packet_bytes':size,'span_ns':end-a,'kernel_ns':sum(k[1]-k[0] for k in ks),'gemv_ns':sum(k[1]-k[0] for k in head),'copies_ns':sum(m[1]-m[0] for m in ms),'kernel_count':len(ks)})
s={}
for stage in ('prefill','decode'):
 rs=[r for r in rows if r['stage']==stage]
 s[stage]={'count':len(rs),**{k:{'sum_ms':sum(r[k] for r in rs)/1e6,'median_us':statistics.median(r[k] for r in rs)/1e3} for k in ('span_ns','kernel_ns','gemv_ns','copies_ns')}}
x={'diagnostic_not_exclusive':True,'segmentation':'graph H2D packet start to final graph activity before next packet; no concurrent owner submissions','stages':s,'rows':rows}
(p/'graph-stage-analysis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(s))
