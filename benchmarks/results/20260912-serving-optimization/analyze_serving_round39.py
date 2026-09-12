import json,pathlib,statistics
r=pathlib.Path('/tmp/riley-opt-260912');d=r/'variable-serving-screen-round39';groups={};raw=[]
for p in sorted(d.glob('*-summary.json')):
 x=json.loads(p.read_text());assert x['completed'] and x['failed']==0 and x['requested']==120;name=p.name.removesuffix('-summary.json');concurrency,workload,pair,lane=name.split('-');raw.append({'case':name,'reference_matches':x['reference_matches'],'requests':x['requested']})
 groups.setdefault(concurrency+'-'+workload,{}).setdefault(lane,[]).append({'throughput':x['successful_output_tokens_per_phase_wall_second'],'ttft_ms':x['token_ttft_ms']['median'],'tpot_ms':x['token_tpot_ms']['median'],'e2e_p95_ms':x['e2e_ms']['p95'],'e2e_p99_ms':x['e2e_ms']['p99']})
assert len(raw)==24 and set(groups)=={"c4-fixed","c4-natural","c8-fixed","c8-natural"}
assert all(set(lanes)=={"previous","new","vllm"} and all(len(rows)==2 for rows in lanes.values()) for lanes in groups.values())
med={w:{lane:{k:statistics.median(row[k] for row in rows) for k in rows[0]} for lane,rows in lanes.items()} for w,lanes in groups.items()}
ratios={w:{lane:{k:100*(lanes['new'][k]/row[k]-1) for k in row} for lane,row in lanes.items() if lane!='new'} for w,lanes in med.items()}
report={'scope':'C4/C8 client serving screen; V28 vs V29 vs vLLM, all matching active capacity, two reversed orders, not model-quality or high-concurrency/stability qualification','medians':med,'new_relative_change_percent':ratios,'requests':sum(x['requests'] for x in raw),'reference_observations':raw,'goal_achieved':False,'decision':'Assess dedicated decode batch against previous V3 and vLLM; high concurrency remains unqualified'}
(r/'serving-round39-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
