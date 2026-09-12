import pathlib,sqlite3,collections,statistics,json,hashlib
r=pathlib.Path('/tmp/riley-opt-260912/mixed-p128-nsys-v22');c=sqlite3.connect('file:'+str(r/'c4.sqlite')+'?mode=ro',uri=True)
starts=[x[0] for x in c.execute('select start from CUPTI_ACTIVITY_KIND_MEMCPY where graphNodeId>0 and copyKind=1 and bytes=17536 order by start')]
k=list(c.execute('select k.start,k.end,s.value,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.graphNodeId>0 order by k.start'))
reports={};cursor=0
for i,start in enumerate(starts):
 end=starts[i+1] if i+1<len(starts) else 2**63-1;batch=[]
 while cursor<len(k) and k[cursor][0]<start:cursor+=1
 while cursor<len(k) and k[cursor][0]<end:batch.append(k[cursor]);cursor+=1
 stage='decode' if any('riley_shared_model::embedding' in x[2] for x in batch) else 'prefill'
 d=reports.setdefault(stage,{'spans':[],'times':collections.defaultdict(list)});d['spans'].append((batch[-1][1]-batch[0][0])/1e3)
 for a,b,name,_ in batch:d['times'][name].append(b-a)
output={}
for stage,d in reports.items():
 rows=[{'kernel':n,'count':len(t),'total_ms':sum(t)/1e6,'mean_us':statistics.mean(t)/1e3} for n,t in sorted(d['times'].items(),key=lambda x:-sum(x[1]))];total=sum(x['total_ms'] for x in rows);proj=sum(x['total_ms'] for x in rows if 'gemm_prefill_shape_vector' in x['kernel'] or 'decode_projection_' in x['kernel'] or 'tile_projection_' in x['kernel'] or 'shared_projection_' in x['kernel']);attn=sum(x['total_ms'] for x in rows if 'attention_shape' in x['kernel'] or 'riley_decode_shape::' in x['kernel'] or 'riley_shared_attention::' in x['kernel'])
 output[stage]={'submissions':len(d['spans']),'median_gpu_kernel_span_us':statistics.median(d['spans']),'kernel_total_ms':total,'projection_percent':100*proj/total,'attention_percent':100*attn/total,'kernels':rows}
assert output['prefill']['submissions']==12 and output['decode']['submissions']>0
x={'scope':'nonexclusive diagnostic GPU trace after Blender restoration; not matched serving timing','binary_sha256':json.loads((r/'receipt.json').read_text())['binary_sha256'],'trace_sha256':hashlib.sha256((r/'c4.nsys-rep').read_bytes()).hexdigest(),'stages':output,'decision':'Dedicated M1 projection/attention after the batch; compare with V11 trace only as nonexclusive diagnostics.'}
(r/'kernel-analysis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({s:{k:v for k,v in q.items() if k!='kernels'} for s,q in output.items()},indent=2))
