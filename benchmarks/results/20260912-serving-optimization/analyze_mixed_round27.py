import json,pathlib,statistics
p=pathlib.Path('/tmp/riley-opt-260912');raw=p/'mixed-p128-screen-round27';completion=json.loads((raw/'completion.json').read_text());assert completion['restored'] and len(completion['records'])==24
out={'scope':'synthetic fixed P128, mixed O8/16/24/32; two repeats240 requests each; median of run summaries, descriptive only','lanes':{},'performance_qualified':False,'engine_capacity':{'baseline':1,'candidate_v9':'1/4/8','new_v10':'1/4/8','vllm':'1/4/8'},'numerical_observation_warning':'Shared-profile and vLLM exact-token mismatches are observations only, not correctness-qualified performance comparisons.'}
for c in (1,4,8):
 out['lanes'][str(c)]={}
 for lane in ('baseline','candidate','new','vllm'):
  xs=[json.loads((raw/f'c{c}-p{i}-{lane}-summary.json').read_text()) for i in (0,1)]
  assert all(x['completed'] for x in xs)
  r={'output_tokens_per_second':statistics.median(x['successful_output_tokens_per_wall_second'] for x in xs),'strict_matches':sum(x['reference_matches'] for x in xs),'responses':480}
  for k in ('token_ttft_ms','token_tpot_ms','e2e_ms'):
   r[k]={v:statistics.median(x[k][v] for x in xs) for v in ('median','p95','p99')}
  out['lanes'][str(c)][lane]=r
 for prior in ('baseline','candidate','vllm'):
  for metric in ('output_tokens_per_second','token_ttft_ms','token_tpot_ms'):
   a=out['lanes'][str(c)]['new'][metric];b=out['lanes'][str(c)][prior][metric]
   if isinstance(a,dict):a=a['median'];b=b['median']
   out['lanes'][str(c)]['v10_vs_'+prior+'_'+metric+'_percent']=100*(a/b-1)
(p/'mixed-p128-screen-round27-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
