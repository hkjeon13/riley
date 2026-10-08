import pathlib,json,time,hashlib,subprocess
r=pathlib.Path('/data/riley-serving-261007');o=r/'kernel-batch12-resume-after-host-gate-attempt02';a=r/'kernel-batch12-independent-analysis-attempt02'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
failure=None;executed=False
try:
 failed=json.loads((r/'kernel-batch12-quiet-attempt01/completion.json').read_text())
 assert failed['records']==[] and failed['failure']['message']=='whole attempt aborted: host start gate timeout c1-fixed-r0-v52-start'
 previous=r/'kernel-batch12-independent-analysis-attempt01'
 while not (previous/'completion.json').exists():time.sleep(10)
 receipt=json.loads((previous/'collection-receipt.json').read_text())
 assert receipt['source_before']==receipt['source_after'] and sha(previous/'kernel-batch12-quiet-attempt01.tar.gz')==receipt['archive_sha256']
 (o/'failed-attempt-preservation.json').write_text(json.dumps({'failed_terminal':failed,'archive_sha256':receipt['archive_sha256'],'archive_bytes':receipt['archive_bytes'],'measured_lanes':0,'all_failures_preserved':True,'no_sample_exclusions':True},indent=2))
 pins=json.loads((o/'pins.json').read_text())
 for p,h in pins.items():assert sha(pathlib.Path(p))==h,p
 plan=json.loads((r/'kernel-batch12-controller-attempt01/batch-plan.json').read_text());policy=plan['host_start_gate'];streak=0
 with (o/'host-admission-observations.jsonl').open('x') as log:
  while streak<policy['consecutive']:
   record={'time_ns':time.time_ns(),'pressure':{n:pathlib.Path('/proc/pressure',n).read_text() for n in ['cpu','io','memory']}}
   def avg(n,kind):return float(next(line for line in record['pressure'][n].splitlines() if line.startswith(kind+' ')).split()[1].split('=')[1])
   quiet=avg('cpu','some')<=policy['cpu_some_max'] and avg('io','full')<=policy['io_full_max'] and avg('memory','full')<=policy['memory_full_max']
   streak=streak+1 if quiet else 0;record.update({'quiet':quiet,'streak':streak});log.write(json.dumps(record)+chr(10));log.flush()
   (o/'progress.json').write_text(json.dumps(record))
   if streak<policy['consecutive']:time.sleep(policy['interval_seconds'])
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 for path,h in pins.items():assert sha(pathlib.Path(path))==h,path
 argv=[str(r/'vllm0271-venv/bin/python'),str(r/'kernel-batch12-controller-attempt01/run.py'),'--attempt','2','--mode','quiet']
 with (o/'matched96-controller.log').open('x') as log:p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (o/'matched96-dispatch.json').write_text(json.dumps({'pid':p.pid,'argv':argv,'plan_sha256':sha(r/'kernel-batch12-controller-attempt01/batch-plan.json'),'controller_sha256':sha(r/'kernel-batch12-controller-attempt01/run.py'),'independent_correctness_sha256':plan['independent_correctness_sha256'],'time_ns':time.time_ns()},indent=2))
 for script,name in [(a/'collect-and-verify.py','collector'),(a/'host-after-replay.py','host')]:
  argv=[str(r/'vllm0271-venv/bin/python'),str(script)]
  with (a/(name+'-dispatch.log')).open('x') as log:c=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (a/(name+'-dispatch.json')).write_text(json.dumps({'pid':c.pid,'argv':argv,'time_ns':time.time_ns()},indent=2))
 report=r/'kernel-batch12-absolute-report-attempt02';argv=['/usr/bin/python3','-B',str(report/'after-replay-report.py')]
 with (report/'dispatch.log').open('x') as log:c=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (report/'dispatch.json').write_text(json.dumps({'pid':c.pid,'argv':argv},indent=2))
 executed=True
 assert p.wait()==0,'second attempt failed; all evidence retained; no automatic further retry'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(o/'completion.json').write_text(json.dumps({'failure':failure,'executed':executed,'adopted':False,'goal_achieved':False},indent=2))
