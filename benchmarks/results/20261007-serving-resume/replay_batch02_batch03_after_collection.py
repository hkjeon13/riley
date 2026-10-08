"""Replay terminal archives locally; preserve failures and never adopt a candidate."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'batch02-batch03-independent-replay-attempt01';O.mkdir()
C=R/'batch02-batch03-terminal-collection-attempt01/completion.json'
BATCHES=[('kernel-batch03-quiet-attempt01','ab5486ebb6cd341bdd350f927807e7dd81590202'),('kernel-batch02-quiet-attempt01','3e9c979feff54b0525e198e569885e0ac0988e7f')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,obj):(O/name).write_text(json.dumps(obj,indent=2)+'\n')
pins={str(R/name):sha(R/name) for name in ['verify_kernel_batch01.py','summarize_kernel_batch01.py','summarize_baseline.py','audit_serving_candidate_contract.py','fixtures.json']}
write('preparation.json',{'script_sha256':sha(Path(__file__)),'input_pins':pins,'wait_for':str(C),'batches':BATCHES,'scope':'local independent raw SSE and frozen launch audit; all rows retained; no adoption or stability claim'})
records=[];failure=None
try:
 deadline=time.monotonic()+18000
 while not C.exists():
  write('wait-state.json',{'time_ns':time.time_ns(),'collection_terminal':False,'raw_replay_executed':False})
  if time.monotonic()>deadline:raise RuntimeError('collection terminal absent at deadline; no replay inferred')
  time.sleep(30)
 receipt=json.loads(C.read_text());assert receipt['failure'] is None,'raw collection failed; evidence preserved'
 write('collection-receipt-snapshot.json',receipt)
 archives={x['scope']:x for x in receipt['records']}
 for scope,commit in BATCHES:
  steps=[];batch_failure=None;archive_record=archives[scope]
  try:
   assert archive_record.get('files_verified',0)>0,'no verified terminal archive'
   archive=Path(archive_record['archive']);assert sha(archive)==archive_record['archive_sha256'],'archive changed after collection'
   assert all(sha(Path(p))==digest for p,digest in pins.items()),'verifier/fixture input changed after declaration'
   replay=O/(scope+'-independent-replay.json');summary=O/(scope+'-summary.json');audit=O/(scope+'-launch-audit.json')
   commands=[('raw-replay',[sys.executable,str(R/'verify_kernel_batch01.py'),str(archive),'--output',str(replay)]),('all8-summary',[sys.executable,str(R/'summarize_kernel_batch01.py'),str(replay),'--output',str(summary)])]
   for name,cmd in commands:
    with (O/(scope+'-'+name+'.log')).open('x') as log:exitcode=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    steps.append({'name':name,'argv':cmd,'exit':exitcode});assert exitcode==0,name+' failed; inspect retained log'
   result=json.loads(replay.read_text())
   if result['matrix_complete']:
    cmd=[sys.executable,str(R/'audit_serving_candidate_contract.py'),str(archive),'--source-commit',commit,'--output',str(audit)]
    with (O/(scope+'-launch-audit.log')).open('x') as log:exitcode=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    steps.append({'name':'frozen-launch-audit','argv':cmd,'exit':exitcode});assert exitcode==0,'launch contract failed'
   else:steps.append({'name':'frozen-launch-audit','execution':'미실행; whole matrix incomplete; completed raw lanes retained'})
   s=json.loads(summary.read_text());lines=['All retained repeats; independently reconstructed SSE timings. No adoption or stability qualification.','Archive SHA256 '+archive_record['archive_sha256'], 'Matrix complete '+str(s['matrix_complete']), 'Whole-attempt failure '+json.dumps(s['failure']), 'All-core minimum mean screen '+str(s['all_core_minimum_mean_screen']), 'Units: throughput token/s; latency ms; memory MiB. Four-repeat mean, sample SD, CV%, min, max.']
   for cell in s['cells']:
    lines.append('\n'+cell['cell']+' complete='+str(cell['complete_four_three_engine_repeats']))
    for engine,metrics in cell['all_available_repeat_statistics'].items():
     lines.append(engine)
     for metric,x in sorted(metrics.items()):lines.append('  '+metric+' '+json.dumps(x,sort_keys=True))
    for target,metrics in cell['complete_cell_comparisons'].items():
     lines.append(target)
     for metric,x in metrics.items():lines.append('  '+metric+' '+json.dumps(x,sort_keys=True))
   (O/(scope+'-all8-report.txt')).write_text('\n'.join(lines)+'\n')
  except Exception as e:batch_failure={'type':type(e).__name__,'message':str(e)}
  records.append({'scope':scope,'source_commit':commit,'archive_receipt':archive_record,'steps':steps,'failure':batch_failure,'adopted':False,'goal_achieved':False});write('progress.json',records)
  print(scope+' local replay terminal '+json.dumps(batch_failure),flush=True)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'records':records,'adopted':False,'goal_achieved':False})
