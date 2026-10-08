import pathlib,json,hashlib,subprocess,time
r=pathlib.Path('/data/riley-serving-261007')
o=r/'kernel-batch08-profile-dispatch-attempt01'
analysis=r/'kernel-batch08-independent-analysis-attempt01'
host=r/'kernel-batch08-host-independent-analysis-attempt01'
failure=None;launched=False
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def live(pid,script):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return False
 if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return False
 assert str(script).encode() in (p/'cmdline').read_bytes().split(b'\0'),'wait PID identity changed'
 return True
try:
 for pid,script in [(3867261,analysis/'collect-and-verify.py'),(3873809,analysis/'host-after-replay.py')]:
  while live(pid,script):time.sleep(30)
 for d in [analysis,host]:assert json.loads((d/'completion.json').read_text())['failure'] is None,'independent evidence failed; no profile'
 summary=json.loads((analysis/'summary.json').read_text())
 assert summary['matrix_complete'] and summary['verified_lane_count']==96
 if summary['all_core_minimum_mean_screen'] is True:
  (o/'skipped.json').write_text(json.dumps({'reason':'all8 mean screen passed; sustained load/cancellation/re-request verification required next','stability':'미실행','goal_achieved':False},indent=2))
 else:
  assert summary['all_core_minimum_mean_screen'] is False
  p=json.loads((o/'profile-policy-before-serving-terminal.json').read_text())
  raw=pathlib.Path(p['baseline'])
  receipt=json.loads((analysis/'collection-receipt.json').read_text())
  names=['preparation.json','completion.json']+[f'c{c}-{w}-r0-{e}-launch.json' for c in [1,8,16,32] for w in ['fixed','natural'] for e in ['v52','candidate']]
  binding={n:sha(raw/n) for n in names}
  assert all(binding[n]==receipt['source_before'][n]['sha256'] for n in names)
  p.update({'baseline_source_file_shas':binding,'baseline_verification_file_sha256':sha(analysis/'raw-replay.json'),'verified_serving_archive_sha256':receipt['archive_sha256'],'remaining_serving_gaps':[{ 'cell':x['cell'],'mean_screen':x['candidate_minimum_mean_screen'],'absolute_means':{e:{k:v['mean'] for k,v in stats.items()} for e,stats in x['all_available_repeat_statistics'].items()}} for x in summary['cells'] if not x['candidate_minimum_mean_screen']]})
  plan=o/'kernel-batch08-cpu-capture-all8-profile-plan-attempt01.json'
  with plan.open('x') as f:json.dump(p,f,indent=2)
  driver=r/'run_kernel_batch08_cpu_capture_all8_profile_attempt01.py'
  assert sha(driver)==json.loads((o/'driver-preparation.json').read_text())['driver_sha256']
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'foreign GPU actor; no profile launch'
  argv=['sudo','-n',str(r/'vllm0271-venv/bin/python'),str(driver),'--attempt','1','--plan',str(plan),'--verified-baseline',str(analysis/'raw-replay.json')]
  (o/'launch.json').write_text(json.dumps({'argv':argv,'plan_sha256':sha(plan),'time_ns':time.time_ns()},indent=2))
  with (o/'profiler.log').open('x') as f:
   proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   launched=True
   (o/'profile-process.json').write_text(json.dumps({'pid':proc.pid,'argv':argv,'time_ns':time.time_ns()},indent=2))
   result=proc.wait()
  (o/'profile-process-exit.json').write_text(json.dumps({'pid':proc.pid,'exit':result},indent=2))
  assert result==0,'profile failed; preserve evidence and do not restart'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(o/'completion.json').write_text(json.dumps({'failure':failure,'launched':launched,'serving_performance':'미실행 (instrumented diagnostic only)','adopted':False,'goal_achieved':False},indent=2))
