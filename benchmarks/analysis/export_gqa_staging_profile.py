"""Validate curated numeric traces and request evidence; raw databases stay remote."""
import hashlib,json,pathlib,sys,tarfile,math
root=pathlib.Path(sys.argv[1]); archive=root/'evidence.tar.gz'
with tarfile.open(archive) as tar:
 def load(name): return json.loads(tar.extractfile('gqa-trace-numeric-v1/'+name).read())
 sources=load('sources.json')
 for name in ('gqa_staging_trace.py','gqa_staging_trace_analysis.py','overlap_headroom.py'):
  assert sources[name]==hashlib.sha256(pathlib.Path(__file__).with_name(name).read_bytes()).hexdigest(),name
 assert load('blender-restored.json')['restored']
 assert all(r['exit_code']==0 for r in load('execution.json'))
 result={}
 for name in ('control-shared','gqa-shared','gqa-unique','control-unique'):
  launch=load(name+'-launch.json'); receipt=load(name+'-receipt.json'); d=load(name+'-analysis.json')
  assert launch['binary_sha256']=='b51b1c4ea6a6c6d3953a639c6459081241e25b51d37ecf0f260384184d8a84b8'
  assert launch['controller_sha256']==sources['gqa_staging_trace.py'] and launch['client_sha256']==sources['paired_decode_serving_screen.py']
  assert launch['rolling_decode'] and launch['gqa_staging']==name.startswith('gqa-')
  assert launch['active_capacity']==32 and launch['client_concurrency']==32 and launch['requests']==64
  assert receipt['reference_exact'] and receipt['exit_code']==0 and not receipt['remaining_owned_pids'] and receipt['peak_owned_tree_rss_bytes']<8*1024**3
  for phase in ('warmup','rows'):
   rows=load(name+'-'+phase+'.json'); assert len(rows)==64
   assert all(r['valid'] and all(r['checks'].values()) and len(r['token_ids'])==32 for r in rows)
  stage=d['stage_graph_spans']['prefill_or_mixed']; kernels=d['stage_kernel_ms']['prefill_or_mixed']
  assert math.isclose(sum(v['sum_ms'] for v in d['stage_graph_spans'].values()),d['selected']['graph_span_ms'],abs_tol=1e-6)
  for k,v in d['kernel_ms'].items(): assert math.isclose(v,sum(s.get(k,0) for s in d['stage_kernel_ms'].values()),abs_tol=1e-6)
  categories={group:sum(v for k,v in kernels.items() if pred(k)) for group,pred in [('attention',lambda k:'mapped_attention' in k or 'gqa_staging::mapped' in k),('projection',lambda k:'gemm_prefill_shape_vector' in k),('ffn',lambda k:'prefill_ffn_pipeline' in k)]}
  assert categories['attention']>0 and any('gqa_staging::mapped' in k for k in kernels)==name.startswith('gqa-')
  result[name]={'selected':d['selected'],'stages':d['stage_graph_spans'],'prefill_kernel_ms':categories,'prefill_fraction':{k:v/stage['sum_ms'] for k,v in categories.items()},'raw_sqlite_sha256':d['source']['source_sha256']}
 out={'scope':'single diagnostic trace per condition, middle 80 percent of graph count including warmup; not serving performance','verified_requests':512,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'results':result}
 (root/'receipt.json').write_text(json.dumps(out,indent=2)+'\n')
 print('512 reference-exact requests; numeric stage accounting and source identities verified')
