"""Recompute the isolated GQA staging gate; never a serving qualification."""
import hashlib,json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); logs=root/'final'
assert (logs/'complete.txt').read_text().strip()=='complete'
assert all(x['exit_code']==0 for x in json.loads((logs/'execution.json').read_text()))
assert json.loads((logs/'blender-restored.json').read_text())['restored']
for line in (logs/'source-sha256.txt').read_text().splitlines():
 digest,name=line.split();assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for name,count in [('check',171),('serving-shapes',48),('memcheck',45),('racecheck',45)]:
 text=(logs/(name+'.log')).read_text();checks=re.findall(r'^CHECK .+$',text,re.M);assert len(checks)==count,(name,len(checks))
 for row in checks:
  fields=dict(re.findall(r'(\w+)=(\d+)',row))
  assert all(fields[k]=='0' for k in ['mismatches','coverage_errors','release_mismatches','error']),row
 if name=='memcheck':assert 'ERROR SUMMARY: 0 errors' in text
 if name=='racecheck':assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in text
for arch in ['sm_90a','sm_100a']:
 text=(logs/(arch+'-build.log')).read_text();assert 'ptxas info' in text and 'error:' not in text.lower()
measurements={}
for row in (logs/'serving-shapes.log').read_text().splitlines():
 if not row.startswith('TIME '):continue
 f=dict(re.findall(r'(\w+)=([\d.]+)',row));key=tuple(int(f[k]) for k in ['prefill','decode','start','context','owners']);mode=int(f['mode'])
 measurements.setdefault(key,{}).setdefault(mode,[]).append(float(f['graph_us']))
assert len(measurements)==8
summary=[]
for key,m in measurements.items():
 assert set(m)=={0,1,2,3} and all(len(v)==2 for v in m.values())
 med=[statistics.median(m[i]) for i in range(4)]
 summary.append(dict(zip(['prefill','decode','start','context','owners'],key),original_us=med[0],compact_us=med[1],grouped_us=med[2],staged_us=med[3],staged_change_percent=100*(med[3]/med[0]-1)))
result={'scope':'native synthetic graph; two reversed-order timing samples; not serving or model qualification','checks':{'audit':171,'captured_replay':171,'bounded_memcheck':45,'bounded_racecheck':45},'timing':summary,'files_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(logs.iterdir()) if p.is_file()}}
(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print('| Prefill | Decode | Prefix | Context | Prefill owners | Original µs | Compact µs | Grouped µs | Staged µs | Change |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in summary:
 print('| '+' | '.join(str(r[k]) for k in ['prefill','decode','start','context','owners'])+' | '+' | '.join(f'{r[k]:.2f}' for k in ['original_us','compact_us','grouped_us','staged_us'])+f" | {r['staged_change_percent']:+.2f}% |")
