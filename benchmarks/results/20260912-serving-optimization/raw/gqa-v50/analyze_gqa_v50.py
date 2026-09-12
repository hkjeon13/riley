from pathlib import Path
import json,statistics
r=Path('/tmp/riley-opt-260912/gqa-attention-v50')
rows=[json.loads(l) for l in (r/'timing.jsonl').read_text().splitlines()]
assert len(rows)==320,len(rows)
result=[]
for n in [1,4,8,16,32]:
 for c in [128,398,1024,4096]:
  times={str(v):statistics.median(x['us'] for x in rows if x['rows']==n and x['context']==c and x['variant']==v) for v in range(4)}
  result.append({'rows':n,'context':c,'median_us':times,'change_percent':{str(v):100*(times[str(v)]/times['0']-1) for v in range(1,4)}})
x={'scope':'Isolated attention CUDA graph, 4 reversed orders x 100 replays; 10 warmup; uniform contexts; not serving qualification','variants':{'0':'V49 QK and values','1':'GQA grouped QK only','2':'GQA shared values only','3':'both grouped'},'records':result}
(r/'analysis.json').write_text(json.dumps(x,indent=2)+'\n')
for q in result:print(q['rows'],q['context'],{k:round(v,3) for k,v in q['median_us'].items()},{k:round(v,2) for k,v in q['change_percent'].items()})
