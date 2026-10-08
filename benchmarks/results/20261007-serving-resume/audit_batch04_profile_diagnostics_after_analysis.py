import time,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'kernel-batch04-profile-diagnostics-local-dispatch-attempt01';O.mkdir();wait=R/'kernel-batch04-profile-independent-analysis-attempt02/completion.json';failure=None
script=R/'audit_kernel_batch04_profile_diagnostics.py';pin=hashlib.sha256(script.read_bytes()).hexdigest();(O/'preparation.json').write_text(json.dumps({'script_sha256':pin,'wait_for':str(wait),'scope':'diagnostics and overlaps; no serving or completeness qualification'},indent=2)+'\n')
try:
 deadline=time.monotonic()+21600
 while not wait.exists():
  if time.monotonic()>deadline:raise RuntimeError('full32 independent profile replay and matrix analysis not terminal')
  time.sleep(30)
 assert json.loads(wait.read_bytes())['failure'] is None
 assert hashlib.sha256(script.read_bytes()).hexdigest()==pin
 with (O/'audit.log').open('x') as log:subprocess.run([sys.executable,str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'goal_achieved':False},indent=2)+'\n')
