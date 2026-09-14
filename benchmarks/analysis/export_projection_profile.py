"""Validate curated numeric traces and request evidence; raw databases stay remote."""
import hashlib,json,pathlib,sys,tarfile,math
from serving_evidence_validation import validate_row
root=pathlib.Path(sys.argv[1]); archive=root/'evidence.tar.gz'
with tarfile.open(archive) as tar:
 def load(name): return json.loads(tar.extractfile('projection-trace-numeric-v1/'+name).read())
 sources=load('sources.json')
 fixtures=load('fixtures.json')
 assert sources['fixtures.json']==hashlib.sha256(tar.extractfile('projection-trace-numeric-v1/fixtures.json').read()).hexdigest()
 assert sources['paired_decode_serving_screen.py']=='adf9930078b8c9fd4a83ef0fd5936749fb9088a5114c47644883458d5dd9cf7e'
 for name in ('projection_pipeline_trace.py','gqa_staging_trace_analysis.py','overlap_headroom.py'):
  assert sources[name]==hashlib.sha256(pathlib.Path(__file__).with_name(name).read_bytes()).hexdigest(),name
 assert load('blender-restored.json')['restored']
 assert all(r['exit_code']==0 for r in load('execution.json'))
 result={}
 for name in ('control-shared','projection-shared','projection-unique','control-unique'):
  launch=load(name+'-launch.json'); receipt=load(name+'-receipt.json'); d=load(name+'-analysis.json')
  assert launch['binary_sha256']=='009697527d143357c3b3d1a2daa5d8b783298024ac1f8f2aa4ea388bbd3ad929'
  assert launch['controller_sha256']==sources['projection_pipeline_trace.py'] and launch['client_sha256']==sources['paired_decode_serving_screen.py']
  assert launch['rolling_decode'] and launch['projection_pipeline']==name.startswith('projection-')
  assert launch['active_capacity']==32 and launch['client_concurrency']==32 and launch['requests']==64
  assert receipt['reference_exact'] and receipt['exit_code']==0 and not receipt['remaining_owned_pids'] and receipt['peak_owned_tree_rss_bytes']<8*1024**3
  assert launch['client_gc_policy']=='disabled'
  assert launch['fixture_sha256']==sources['fixtures.json']
  assert launch['kv_payload_bytes']==754974720
  by_id={f['id']:f for f in fixtures[name.split('-')[1]]}
  for phase in ('warmup','rows'):
   rows=load(name+'-'+phase+'.json'); assert len(rows)==64
   assert all(all(validate_row(r,by_id[r['id']]).values()) and len(r['token_ids'])==32 for r in rows)
   gc=receipt['client_gc'][phase]
   assert gc['policy']=='disabled' and gc['before']==gc['after'] and gc['restored_enabled']
  stage=d['stage_graph_spans']['prefill_or_mixed']; kernels=d['stage_kernel_ms']['prefill_or_mixed']
  assert math.isclose(sum(v['sum_ms'] for v in d['stage_graph_spans'].values()),d['selected']['graph_span_ms'],abs_tol=1e-6)
  for k,v in d['kernel_ms'].items(): assert math.isclose(v,sum(s.get(k,0) for s in d['stage_kernel_ms'].values()),abs_tol=1e-6)
  categories={group:sum(v for k,v in kernels.items() if pred(k)) for group,pred in [('attention',lambda k:'mapped_attention' in k or 'gqa_staging::mapped' in k),('projection',lambda k:'gemm_prefill_shape_vector' in k or 'riley_prefill_projection::project' in k),('ffn',lambda k:'prefill_ffn_pipeline' in k)]}
  assert categories['attention']>0 and any('riley_prefill_projection::project' in k for k in kernels)==name.startswith('projection-')
  result[name]={'selected':d['selected'],'stages':d['stage_graph_spans'],'prefill_kernel_ms':categories,'prefill_fraction':{k:v/stage['sum_ms'] for k,v in categories.items()},'raw_sqlite_sha256':d['source']['source_sha256']}
 out={'scope':'single diagnostic trace per condition, middle 80 percent of graph count including warmup; not serving performance','verified_requests':512,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'results':result}
 (root/'receipt.json').write_text(json.dumps(out,indent=2)+'\n')
 print('512 reference-exact requests; numeric stage accounting and source identities verified')
