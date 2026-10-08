import json,subprocess,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch03-profile-independent-analysis-attempt01';O.mkdir()
C=R/'kernel-batch03-profile-terminal-collection-attempt02';failure=None
pins={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['verify_kernel_batch03_profiles.py','analyze_kernel_batch03_profile_matrix.py','analyze_profile_intervals.py','kernel-batch03-controller/fixtures.json']}
(O/'preparation.json').write_text(json.dumps({'pins':pins,'scope':'32 diagnostic profiles only; no serving claim'},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not (C/'completion.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('collection terminal absent; no verification inferred')
  time.sleep(20)
 receipt=json.loads((C/'completion.json').read_text());assert receipt['failure'] is None, 'collection failed; no partial verification'
 assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==v for n,v in pins.items())
 argv=[sys.executable,str(R/'verify_kernel_batch03_profiles.py'),str(C/'kernel-batch03-profiling-attempt01.tar.gz'),'--fixtures',str(R/'kernel-batch03-controller/fixtures.json'),'--output',str(O/'independent32-profile-verification.json')]
 with (O/'verification.log').open('x') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0, 'independent32 failed; inspect original log'
 with (O/'analysis.log').open('x') as f:p=subprocess.run([sys.executable,str(R/'analyze_kernel_batch03_profile_matrix.py')],stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0, 'actual32 SQLite analysis failed'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행; diagnostic timings only','goal_achieved':False},indent=2)+'\n')
