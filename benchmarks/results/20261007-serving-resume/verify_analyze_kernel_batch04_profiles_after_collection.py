import json,subprocess,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch04-profile-independent-analysis-attempt01';O.mkdir()
C=R/'kernel-batch04-profile-terminal-collection-attempt01';failure=None
pins={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['verify_kernel_batch04_profiles.py','analyze_kernel_batch04_profile_matrix.py','analyze_profile_intervals.py','summarize_baseline.py','verify_kernel_batch01.py','kernel-batch04-controller/fixtures.json']}
(O/'preparation.json').write_text(json.dumps({'pins':pins,'scope':'32 diagnostic profiles only; no serving claim'},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not (C/'completion.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('collection terminal absent; no verification inferred')
  time.sleep(20)
 receipt=json.loads((C/'completion.json').read_text());assert receipt['failure'] is None, 'collection failed; no partial verification'
 assert len(receipt['records'])==1
 record=receipt['records'][0];archive=C/'kernel-batch04-profiling-attempt01.tar.gz';h=hashlib.sha256()
 with archive.open('rb') as source:
  for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==record['archive_sha256'],'archive changed since collection'
 (O/'source-archive-binding.json').write_text(json.dumps({'archive_sha256':h.hexdigest(),'files_verified':record['files_verified'],'source_pins':pins},indent=2)+'\n')
 assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==v for n,v in pins.items())
 argv=[sys.executable,str(R/'verify_kernel_batch04_profiles.py'),str(C/'kernel-batch04-profiling-attempt01.tar.gz'),'--fixtures',str(R/'kernel-batch04-controller/fixtures.json'),'--output',str(O/'independent32-profile-verification.json')]
 with (O/'verification.log').open('x') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0, 'independent32 failed; inspect original log'
 with (O/'analysis.log').open('x') as f:p=subprocess.run([sys.executable,str(R/'analyze_kernel_batch04_profile_matrix.py')],stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0, 'actual32 SQLite analysis failed'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행; diagnostic timings only','goal_achieved':False},indent=2)+'\n')
