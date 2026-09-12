from pathlib import Path
import json,re,statistics
r=Path('/tmp/riley-opt-260912');d=r/'mixed-attention-v49'
s=(d/'guards-timing.log').read_text();s,n=re.subn(r'mixed attention cases=32 three_geometries_exact=true finite_nonfinite_causal_owner_isolation=true inactive_guards=true\n','',s);assert n==1
rows=[json.loads(x) for x in s.splitlines() if x.startswith('{')];assert len(rows)==480
(d/'timing-clean.json').write_text(json.dumps({'note':'Removed exactly one interleaved stderr summary; original guards-timing.log retained.','rows':rows},indent=2)+'\n')
a=[]
for pattern,owners in sorted({(x['pattern'],x['owners']) for x in rows}):
 q=[x for x in rows if x['pattern']==pattern and x['owners']==owners];med={v:statistics.median(x['us'] for x in q if x['variant']==v) for v in range(4)}
 a.append({'pattern':pattern,'owners':owners,'tokens':q[0]['tokens'],'tiles':q[0]['tiles'],'reference':'V48 four-owner launch geometry' if owners<=4 else 'hypothetical rectangular32 launch geometry; not V48 serving','median_us':med,'compact_change_pct':{v:100*(med[v]/med[0]-1) for v in range(1,4)}})
(d/'analysis.json').write_text(json.dumps({'scope':'Attention component only, capacity512 if total<=512 otherwise1024; four reversed-order pairs,10 warmups and100 graph replays. Variants1/2/3 use64/128/256 CTAs per head. No model, scheduler, HTTP or serving claim.','cases':a},indent=2)+'\n')
for x in a:
 if x['owners'] in [4,32]:print(json.dumps(x))
