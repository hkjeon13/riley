from pathlib import Path
import json,statistics,collections
r=Path('/tmp/riley-opt-260912');d=r/'natural-fill-v41';report={}
for p in sorted(d.glob('c*-v*.log')):
 rows=[]
 for line in p.read_text().splitlines():
  if not line.startswith('BATCH '):continue
  fields=dict(x.split('=',1) for x in line.split()[1:]);rows.append({k:v if k=='stage' else int(v) for k,v in fields.items()})
 assert rows,p
 def summarize(subset):
  total=sum(x['us'] for x in subset);out={}
  for stage in ['prefill','decode']:
   a=[x for x in subset if x['stage']==stage];out[stage]={'count':len(a),'execution_fraction_percent':100*sum(x['us'] for x in a)/total,'rows_mean':statistics.mean(x['rows'] for x in a),'rows_histogram':dict(collections.Counter(x['rows'] for x in a)),**{k+'_median':statistics.median(x[k] for x in a) for k in ['us','encode_us','native_us','validate_us']},'tokens_sum':sum(x['tokens'] for x in a)}
  return out
 report[p.stem]={'all':summarize(rows),'middle80':summarize(rows[len(rows)//10:len(rows)*9//10])}
assert set(report)=={'c16-v3','c16-v4','c32-v3','c32-v4'}
receipt=json.loads((d/'receipt.json').read_text());assert len(receipt['records'])==4 and all(x['exact_reference_text_and_token_ids'] for x in receipt['records'])
x={'scope':'instrumented V41 binary, profiles V3/V4, each384 natural mixed requests; client16/32 and equal admission; budget/chunk512; diagnostic only. Native interval includes replay, synchronization and host readback, not GPU-only time.','cases':report,'receipt':receipt}
(r/'fill-v41-analysis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
