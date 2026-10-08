from pathlib import Path
import json,statistics
r=Path('/tmp/riley-opt-260912');d=r/'decode-mask-v55'
xs=[json.loads(x) for x in (d/'timing.jsonl').read_text().splitlines()];assert len(xs)==400
rows=[]
for ctx in [128,398,1024,4096]:
 for n in [1,4,8,16,32]:
  ms=[statistics.median(x['us'] for x in xs if x['context']==ctx and x['rows']==n and x['variant']==v) for v in range(5)]
  rows.append({'context':ctx,'rows':n,'median_us':ms,'vs_v52_percent':[100*(t/ms[0]-1) for t in ms],'vs_v54_percent':[100*(t/ms[1]-1) for t in ms]})
x={'scope':'Decode-only CUDA graph replay; V52 token-major, V54 packed, branchless-mask, full-tile-fast-path, K16-loop; not serving evidence','records':len(xs),'rows':rows}
(d/'analysis.json').write_text(json.dumps(x,indent=2)+'\n')
for row in rows:print(row['context'],row['rows'],[round(v,3) for v in row['median_us']],[round(v,2) for v in row['vs_v52_percent']])
