from pathlib import Path
import json,re,statistics
r=Path('/tmp/riley-opt-260912');d=r/'mixed-attention-map-v49'
s=(d/'timing.log').read_text();s,n=re.subn(r'mixed attention cases=32 four_geometries_exact=true finite_nonfinite_causal_owner_isolation=true inactive_guards=true\n','',s);assert n==1
rows=[json.loads(x) for x in s.splitlines() if x.startswith('{')];assert len(rows)==600
(d/'timing-clean.json').write_text(json.dumps({'note':'Removed exactly one interleaved stderr summary; original timing.log retained.','rows':rows},indent=2)+'\n')
a=[]
for pattern,owners in sorted({(x['pattern'],x['owners']) for x in rows}):
 q=[x for x in rows if x['pattern']==pattern and x['owners']==owners];med={v:statistics.median(x['us'] for x in q if x['variant']==v) for v in range(5)}
 a.append({'pattern':pattern,'owners':owners,'tokens':q[0]['tokens'],'tiles':q[0]['tiles'],'reference':'V48 four-owner launch geometry' if owners<=4 else 'hypothetical rectangular32 launch geometry; not V48 serving','median_us':med,'mapped_change_vs_rectangular_pct':100*(med[4]/med[0]-1),'mapped_change_vs_compact128_pct':100*(med[4]/med[2]-1)})
(d/'analysis.json').write_text(json.dumps({'scope':'Attention component only; variants0 rectangular,1/2/3 persistent64/128/256,4 direct mapped; capacity512 if total<=512 otherwise1024; four reversed pairs,10 warmups,100 graph replays.','cases':a},indent=2)+'\n')
for x in a:
 if x['owners'] in [4,32]:print(json.dumps(x))
