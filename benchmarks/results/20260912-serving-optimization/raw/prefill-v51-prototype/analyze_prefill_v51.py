from pathlib import Path
import json,statistics,sys
r=Path('/tmp/riley-opt-260912')/(sys.argv[1] if len(sys.argv)>1 else 'prefill-projection-v51-matched')
xs=[json.loads(l) for l in (r/'timing.jsonl').read_text().splitlines()];assert len(xs)==672
result=[]
for key in sorted(set((x['N'],x['K'],x['I'],x['fused'],x['rows']) for x in xs)):
 N,K,I,F,rows=key;t={str(v):statistics.median(x['us'] for x in xs if (x['N'],x['K'],x['I'],x['fused'],x['rows'])==key and x['variant']==v) for v in range(4)}
 result.append({'N':N,'K':K,'interval':I,'fused':F,'rows':rows,'median_us':t,'change_percent':{str(v):100*(t[str(v)]/t['0']-1) for v in range(1,4)}})
(r/'analysis.json').write_text(json.dumps({'scope':'CUDA graph primitive; 4 reversed orders x100 replays,10warmup; production capacity512 for rows<=512 in matched directory; not serving proof','records':result},indent=2)+'\n')
for q in result:
 if q['rows'] in [16,128,398,512]:print(q['N'],q['K'],q['interval'],q['fused'],q['rows'],{k:round(v,3) for k,v in q['median_us'].items()},{k:round(v,2) for k,v in q['change_percent'].items()})
