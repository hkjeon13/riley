from pathlib import Path
import json,statistics
r=Path('/tmp/riley-opt-260912/attention-state-v52');xs=[json.loads(l) for l in (r/'timing.jsonl').read_text().splitlines()];assert len(xs)==480,len(xs)
result=[]
for pattern in range(6):
 for owners in [1,4,8,16,32]:
  t={str(v):statistics.median(x['us'] for x in xs if x['pattern']==pattern and x['owners']==owners and x['variant']==v) for v in range(4)};result.append({'pattern':pattern,'owners':owners,'median_us':t,'change_percent':{str(v):100*(t[str(v)]/t['0']-1) for v in range(1,4)}})
(r/'analysis.json').write_text(json.dumps({'scope':'CUDA graph attention primitive, mapped dispatch identical, capacity512 when tokens<=512 otherwise1024; 4 reversed orders,10warmup+100replay; not serving proof','variants':{'0':'V51 baseline','1':'query tile sized state','2':'score/exponential lifetime alias','3':'both'},'records':result},indent=2)+'\n')
for q in result:print(q['pattern'],q['owners'],{k:round(v,3) for k,v in q['median_us'].items()},{k:round(v,2) for k,v in q['change_percent'].items()})
