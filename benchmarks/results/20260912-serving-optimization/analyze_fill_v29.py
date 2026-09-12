import pathlib,json,re,statistics,collections,subprocess,shutil,hashlib
r=pathlib.Path('/tmp/riley-opt-260912');out={}
for cap in (4,8):
 rows=[{'stage':s,'rows':int(n),'tokens':int(t),'us':int(us)} for s,n,t,us in re.findall(r'BATCH stage=(\w+) rows=(\d+) tokens=(\d+) us=(\d+)',(r/f'natural-fill-v29/c{cap}.log').read_text())]
 sections={}
 for label,data in [('all',rows),('middle80',rows[len(rows)//10:len(rows)*9//10])]:
  dec=[x for x in data if x['stage']=='decode'];pre=[x for x in data if x['stage']=='prefill'];assert dec and pre
  sections[label]={'iterations':len(data),'decode_iterations':len(dec),'prefill_iterations':len(pre),'decode_row_histogram':dict(collections.Counter(x['rows'] for x in dec)),'mean_decode_rows':statistics.mean(x['rows'] for x in dec),'full_decode_fraction':sum(x['rows']==cap for x in dec)/len(dec),'prefill_execution_time_fraction':sum(x['us'] for x in pre)/sum(x['us'] for x in data),'decode_median_execution_us':statistics.median(x['us'] for x in dec)}
 out[str(cap)]=sections
x={'scope':'Instrumented nonexclusive V28 natural serving, 132 requests each at C4 and C8. Middle80 excludes first/last10percent of iterations. Batch occupancy is request-row occupancy, not GPU occupancy. Execution includes transfer and validation.','results':out};(r/'fill-v29-analysis.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
d=r/'fill-diagnostic-v29';d.mkdir();shutil.copy2(r/'prefill-shapes-target-v11/release/riley',d/'riley');(d/'build.json').write_text(json.dumps({'base':'e181ae97e1202050654e887753893b609dd5ff44','binary_sha256':hashlib.sha256((d/'riley').read_bytes()).hexdigest(),'instrumentation':True},indent=2));(r/'fill-v29-instrumentation.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=r/'prefill-shapes-source-v11'));subprocess.run(['git','restore','crates/riley-runtime/src/llama/variable_session.rs'],cwd=r/'prefill-shapes-source-v11',check=True)
