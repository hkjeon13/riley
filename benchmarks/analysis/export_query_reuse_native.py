"""Validate native-gate logs and derive paired graph timing, never serving speedup."""
import collections,hashlib,json,pathlib,re,statistics,sys
root=pathlib.Path(sys.argv[1]);e=root/'evidence';repo=pathlib.Path(__file__).resolve().parents[2]
assert (e/'complete.txt').read_text().strip()=='complete' and not (e/'compute-after.csv').read_text().strip()
for line in (e/'source-sha256.txt').read_text().splitlines():
 digest,path=line.split(maxsplit=1);assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==digest
checks=[]
for file,count in [('check.log',153),('timing.log',102),('serving-shapes.log',24),('memcheck.log',27),('racecheck.log',27)]:
 rows=[dict(re.findall(r'(\w+)=(\d+)',line)) for line in (e/file).read_text().splitlines() if line.startswith('CHECK ')]
 assert len(rows)==count
 assert all(all(int(r[k])==0 for k in ('mismatches','coverage_errors','release_mismatches','error')) for r in rows)
 checks.append({'file':file,'audit_and_replay_cases':len(rows)})
assert 'ERROR SUMMARY: 0 errors' in (e/'memcheck.log').read_text()
assert '0 errors, 0 warnings' in (e/'racecheck.log').read_text()
for arch in ('sm_90a','sm_100a'):assert (e/(arch+'-build.log')).stat().st_size>0
comparisons={}
for file in ('timing.log','serving-shapes.log'):
 groups=collections.defaultdict(lambda:collections.defaultdict(list))
 for line in (e/file).read_text().splitlines():
  if line.startswith('TIME '):
   d=dict(re.findall(r'(\w+)=([\d.]+)',line));key=tuple(int(d[k]) for k in ('prefill','decode','start','context'));groups[key][int(d['mode'])].append(float(d['graph_us']))
 rows=[]
 for shape,values in groups.items():
  assert set(values)==set(range(4)) and all(len(v)==2 for v in values.values())
  med={str(m):statistics.median(v) for m,v in values.items()}
  rows.append({'shape':dict(zip(('prefill','decode','start','context'),shape)),'median_graph_us':med,'compact16_change_percent':(med['3']/med['0']-1)*100})
 comparisons[file]=rows
out={'scope':'native attention graph timing, two reversed orders of 50 graph replays after 10 warmups; no serving or quality promotion','modes':{'0':'existing8','1':'compact8','2':'query16','3':'compact-query16'},'checks':checks,'comparisons':comparisons,'evidence_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(e.iterdir()) if p.is_file()}}
(root/'receipt.json').write_text(json.dumps(out,indent=2)+'\n')
for r in comparisons['serving-shapes.log']:print(r)
