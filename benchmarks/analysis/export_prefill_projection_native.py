"""Validate native projection evidence. This does not qualify model or serving."""
import hashlib,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]); repo=pathlib.Path(__file__).resolve().parents[2]
receipt=json.loads((root/'sources.json').read_text())
for name,h in receipt.items(): assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h,name
assert json.loads((root/'blender-restored.json').read_text())['restored']
assert [r['exit_code'] for r in json.loads((root/'execution.json').read_text())]==[0,0,0]
def rows(name):return [json.loads(l) for l in (root/name).read_text().splitlines() if l.startswith('{')]
audit=rows('audit.log');checks=[r for r in audit if 'layers_exact' in r]
assert len(checks)==78 and all(r['layers_exact']==30 and r['graph_exact'] for r in checks)
for n,interval in [(576,192),(192,192),(576,128)]:
 assert {(r['rows'],r['repeat']) for r in checks if r['N']==n and r['interval']==interval}=={(m,i) for m in [0,1,8,16,17,31,32,64,128,398,512,1024,1025] for i in range(2)}
for name in ['memcheck.log','racecheck.log']:
 text=(root/name).read_text(); assert 'ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text
 assert len([r for r in rows(name) if 'layers_exact' in r])==12
for name in ['build.log','sm_90a-compile.log','sm_100a-compile.log']:
 text=(root/name).read_text();assert 'error:' not in text and 'ptxas info' in text
result=[]
for n,interval in [(576,192),(192,192),(576,128)]:
 for m in [32,128,398,512]:
  samples=[r for r in audit if r.get('timing_rows')==m and r['N']==n and r['interval']==interval];assert len(samples)==4
  estimates=[statistics.median(r['layer_us'] for r in samples if r['candidate']==c) for c in [0,1]]
  result.append({'N':n,'K':576,'interval':interval,'M':m,'prior_us':estimates[0],'candidate_us':estimates[1],'reduction_percent':100*(1-estimates[1]/estimates[0])})
(root/'comparison.json').write_text(json.dumps({'scope':'30-layer weight rotation native graph; two order estimates; not serving','audit_conditions':78,'per_layer_comparisons':2340,'memcheck_conditions':12,'racecheck_conditions':12,'resources':[r for r in audit if 'shared' in r],'comparison':result},indent=2)+'\n')
print('native audit, sanitizer, compilation and source checks passed')
