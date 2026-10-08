import pathlib,json,hashlib,subprocess,sys,time
R=pathlib.Path(__file__).resolve().parent;O=R/'kernel-batch07-independent-replay-attempt01';O.mkdir();C=R/'kernel-batch07-terminal-collection-attempt01/completion.json';failure=None;steps=[]
inputs=['verify_kernel_batch07.py','audit_kernel_batch07_serving_contract.py','summarize_kernel_batch01.py','summarize_baseline.py','kernel-batch07-controller/fixtures.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();pins={n:sha(R/n) for n in inputs}
(O/'preparation.json').write_text(json.dumps({'input_sha256':pins,'wait_for':str(C),'expected_lanes':96,'source_commit':'73003e2d8f1ce797b4e669c813c43261b30a46a2','no_exclusions':True},indent=2)+'\n')
try:
 deadline=time.monotonic()+43200
 while not C.exists():
  assert time.monotonic()<deadline,'collection still incomplete; no restart or inferred results'
  time.sleep(30)
 collection=json.loads(C.read_bytes());assert collection['failure'] is None
 assert all(sha(R/n)==h for n,h in pins.items())
 record=collection['records'][0];archive=pathlib.Path(record['archive']);assert sha(archive)==record['archive_sha256']
 replay=O/'raw-replay.json';summary=O/'summary.json';audit=O/'launch-audit.json'
 cmds=[('raw-replay',[sys.executable,str(R/'verify_kernel_batch07.py'),str(archive),'--fixtures',str(R/'kernel-batch07-controller/fixtures.json'),'--output',str(replay)]),('all8-summary',[sys.executable,str(R/'summarize_kernel_batch01.py'),str(replay),'--output',str(summary)])]
 for label,cmd in cmds:
  with (O/(label+'.log')).open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  steps.append({'step':label,'argv':cmd,'exit':proc.returncode});assert proc.returncode==0,label+' failed; original errors preserved'
 if json.loads(replay.read_bytes())['matrix_complete']:
  cmd=[sys.executable,str(R/'audit_kernel_batch07_serving_contract.py'),str(archive),'--source-commit','73003e2d8f1ce797b4e669c813c43261b30a46a2','--output',str(audit)]
  with (O/'launch-audit.log').open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  steps.append({'step':'launch-audit','exit':proc.returncode});assert proc.returncode==0
 else:steps.append({'step':'launch-audit','execution':'미실행; partial matrix cannot qualify'})
 s=json.loads(summary.read_bytes());lines=['All four repeats preserved; absolute throughput/TTFT/TPOT/E2E/tails/errors/memory and repeat SD/CV. No adoption or stability claim.','Archive SHA256 '+record['archive_sha256'],'Complete '+str(s['matrix_complete']),'All-core screen '+str(s['all_core_minimum_mean_screen'])]
 for c in s['cells']:
  lines.append('\n'+c['cell'])
  for e,metrics in c['all_available_repeat_statistics'].items():
   lines.append(e)
   for k,v in sorted(metrics.items()):lines.append(k+' '+json.dumps(v,sort_keys=True))
  for k,v in c['complete_cell_comparisons'].items():lines.append(k+' '+json.dumps(v,sort_keys=True))
 (O/'all8-report.txt').write_text('\n'.join(lines)+'\n');print('Batch07 actual terminal independent replay complete; adopted=false',flush=True)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'steps':steps,'adopted':False,'goal_achieved':False},indent=2)+'\n')
