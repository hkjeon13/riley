import pathlib,re,statistics,json,hashlib
r=pathlib.Path('/tmp/riley-opt-260912');report={}
for phase,suffix in [('before',''),('after','-after')]:
 p=r/f'v3-http-v25-host{suffix}-shared-final/server.log';d={}
 for name,rows,t in re.findall(r'HOST_(\w+) (?:rows=(\d+) )?us=(\d+)',p.read_text()):d.setdefault(name+('/rows'+rows if rows else ''),[]).append(int(t))
 assert d and d['SAMPLE/rows4'] and d['VALIDATE/rows4']
 report[phase]={'log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'stages':{k:{'n':len(v),'median_us':statistics.median(v),'total_us':sum(v)} for k,v in d.items()}}
x={'scope':'Nonexclusive instrumented actual HTTP serving; wall times include instrumentation/host interference, diagnostic not matched performance','phases':report}
(r/'host-v25-comparison.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
