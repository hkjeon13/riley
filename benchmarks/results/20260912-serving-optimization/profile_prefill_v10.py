import sqlite3,pathlib,json,collections,hashlib
r=pathlib.Path('/tmp/riley-opt-260912');p=r/'mixed-p128-nsys-v10';c=sqlite3.connect('file:'+str(p/'c8.sqlite')+'?mode=ro',uri=True)
u=c.execute('select start from CUPTI_ACTIVITY_KIND_MEMCPY where graphNodeId>0 and copyKind=1 and bytes in (1792,600) order by start').fetchall()
k=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.graphNodeId>0 order by k.start').fetchall()
a=collections.defaultdict(list);stages=0
for i,(start,) in enumerate(u):
 end=u[i+1][0] if i+1<len(u) else 2**63-1
 rows=[v for v in k if start<=v[0]<end]
 if not any(v[2].startswith('attention(') or 'attention_queries<' in v[2] for v in rows):continue
 stages+=1
 for s,e,n in rows:a[n].append(e-s)
rows=[{'name':n,'count':len(v),'total_ms':sum(v)/1e6,'mean_us':sum(v)/len(v)/1e3} for n,v in sorted(a.items(),key=lambda q:-sum(q[1]))]
total=sum(x['total_ms'] for x in rows)
projection=sum(x['total_ms'] for x in rows if 'gemm_prefill_m16' in x['name'])
x={'source_sha256':hashlib.sha256((p/'c8.nsys-rep').read_bytes()).hexdigest(),'diagnostic_nonexclusive':True,'prefill_submissions':stages,'kernel_total_ms':total,'custom_prefill_projection_ms':projection,'custom_projection_percent':100*projection/total,'kernels':rows,'interpretation':'gemm_prefill_m16 are custom inline MMA kernels, not cuBLAS dense GEMM; name substring gemm alone does not identify a library implementation.'}
(p/'prefill-kernel-analysis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({key:value for key,value in x.items() if key!='kernels'},indent=2))
