from pathlib import Path
import json,subprocess,time,hashlib
R=Path('/data/riley-serving-261007');O=R/'kernel-batch11-profile32-preparation-attempt01'
P=R/'kernel-batch11-source-owned-stability-preparation-attempt03'
failure=None
try:
 pid=2785246
 while True:
  p=Path('/proc')/str(pid);live=p.exists()
  if live:
   live=(p/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
   if live:assert str(P/'repair-and-compile.py') in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),'PID changed'
  (O/'compile-wait.json').write_text(json.dumps({'pid':pid,'confirmed_live':live,'time_ns':time.time_ns()}))
  if not live:break
  time.sleep(10)
 assert json.loads((P/'completion.json').read_text())['failure'] is None,'compile did not pass; preserve and stop'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'foreign GPU process'
 assert json.loads((R/'kernel-batch11-independent-analysis-attempt02/summary.json').read_text())['all_core_minimum_mean_screen'] is False
 argv=['sudo','-n','/usr/bin/python3','-B',str(O/'run.py'),'--attempt','1','--plan',str(O/'plan-after-build-repair03.json'),'--verified-baseline',str(R/'kernel-batch11-independent-analysis-attempt02/raw-replay.json')]
 with (O/'profile-controller.log').open('x') as log:
  child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with (O/'root-profile-dispatch.json').open('x') as f:json.dump({'pid':child.pid,'argv':argv,'time_ns':time.time_ns(),'scope':'matched all32 diagnostic CPU/CUDA profiles; no adoption or serving timing claim'},f,indent=2)
  exit=child.wait()
 assert exit==0,'profile controller failed; all observations preserved'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)};raise
finally:
 with (O/'dispatch-completion.json').open('x') as f:json.dump({'failure':failure,'goal_achieved':False},f,indent=2)
