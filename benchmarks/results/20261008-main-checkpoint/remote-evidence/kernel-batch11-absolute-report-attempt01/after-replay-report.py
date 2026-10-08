from pathlib import Path
import json,time,hashlib
from absolute_renderer import render
R=Path('/data/riley-serving-261007');O=Path(__file__).parent;A=R/'kernel-batch11-independent-analysis-attempt02';H=R/'kernel-batch11-host-independent-analysis-attempt02'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
failure=None
try:
 while True:
  states=[]
  for pid,expected in [(2368834,'kernel-batch11-controller-attempt02/run.py'),(2384242,'kernel-batch11-independent-analysis-attempt02/collect-and-verify.py'),(2384243,'kernel-batch11-independent-analysis-attempt02/host-after-replay.py')]:
   p=Path('/proc')/str(pid);live=p.exists()
   if live:
    live=(p/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    if live:assert expected in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),'PID identity changed'
   states.append({'pid':pid,'confirmed_live':live})
  (O/'verified-wait.json').write_text(json.dumps({'time_ns':time.time_ns(),'handles':states}))
  if not any(v['confirmed_live'] for v in states):break
  time.sleep(30)
 for p in [R/'kernel-batch11-quiet-attempt02/completion.json',A/'completion.json',H/'completion.json']:
  assert json.loads(p.read_text())['failure'] is None,str(p)
 summary=json.loads((A/'summary.json').read_text());replay=json.loads((A/'raw-replay.json').read_text());receipt=json.loads((A/'collection-receipt.json').read_text())
 assert replay['matrix_complete'] and replay['raw_replay_passed'] and len(replay['lanes'])==96
 archive=A/'kernel-batch11-quiet-attempt02.tar.gz'
 # Collector already computed archive SHA and checked unchanged raw; don't reread a large archive while build starts.
 evidence={'archive_path':str(archive),'archive_sha256':receipt['archive_sha256'],'summary_sha256':sha(A/'summary.json'),'replay_sha256':sha(A/'raw-replay.json'),'host_completion_sha256':sha(H/'completion.json'),'collector_receipt_sha256':sha(A/'collection-receipt.json')}
 with (O/'report.md').open('x') as f:f.write(render(summary,evidence))
 with (O/'report-receipt.json').open('x') as f:json.dump({'inputs':evidence,'report_sha256':sha(O/'report.md'),'formatter_sha256':sha(O/'absolute_renderer.py'),'all96_only':True,'relative_percentages_in_report':False,'stability':'미실행','goal_achieved':False},f,indent=2)
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)};raise
finally:
 with (O/'completion.json').open('x') as f:json.dump({'failure':failure,'goal_achieved':False},f,indent=2)
