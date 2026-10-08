"""Keep original diagnostic evidence and verify every all8 ABBA profile."""
import hashlib,json,pathlib,shlex,subprocess,sys,time
R=pathlib.Path(__file__).resolve().parent
O=R/'kernel-batch07-cpu-capture-all8-profile-analysis-queue-attempt01';O.mkdir()
WAIT=R/'kernel-batch07-cpu-capture-all8-profile-local-dispatch-attempt01/completion.json'
NAME='kernel-batch07-cpu-capture-all8-profile'
COLLECT=R/(NAME+'-independent-collection-attempt01')
programs=['collect_kernel_batch07_cpu_capture_all8_profile_raw_attempt01.py','verify_kernel_batch07_cpu_capture_all8_profile_attempt01.py','analyze_kernel_batch07_cpu_capture_all8_profile_phases_attempt01.py','analyze_kernel_batch07_cpu_stacks_attempt01.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
pins={n:sha(R/n) for n in programs};failure=None;steps=[]
write('preparation.json',{'source_pins':pins,'wait_for':str(WAIT),'expected_profiles':32,'expected_HTTP_requests':6144,'scope':'all original startup/warmup/retained/teardown and raw CPU/CUDA traces preserved; instrumented times never serving scores'})
try:
 deadline=time.monotonic()+43200
 while not WAIT.exists():
  assert time.monotonic()<deadline,'dispatcher remains live/pending; no restart'
  time.sleep(30)
 dispatch=json.loads(WAIT.read_bytes());write('dispatcher-terminal-snapshot.json',dispatch)
 if not dispatch['launched']:
  write('skipped.json',{'reason':dispatch['decision'] or dispatch['failure'],'actual_GPU_profiling':'미실행'})
 else:
  probe="import json,subprocess;from pathlib import Path;live=[]\nfor p in Path('/proc').iterdir():\n if not p.name.isdigit():continue\n try:a=p.joinpath('cmdline').read_bytes().split(bytes([0]))\n except (PermissionError,FileNotFoundError,ProcessLookupError):continue\n if b'/data/riley-serving-261007/run_kernel_batch07_cpu_capture_all8_profile_attempt01.py' in a:live.append(int(p.name))\nr=Path('/data/riley-serving-261007/kernel-batch07-cpu-capture-all8-profile-attempt01');print(json.dumps({'live':live,'terminal':(r/'completion.json').exists(),'GPU':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()}))"
  while True:
   state=json.loads(subprocess.check_output(['ssh','ai-assistant','sudo -n python3 -c '+shlex.quote(probe)],text=True));write('wait-state.json',state)
   if not state['live'] and not state['GPU']:
    assert state['terminal'],'actual controller absent without raw terminal; inspect original dispatcher log; no restart'
    break
   assert time.monotonic()<deadline,'original diagnostic or GPU actor remains live; no archive IO or restart'
   time.sleep(30)
  assert all(sha(R/n)==h for n,h in pins.items()),'analysis sources changed since declaration'
  cmds=[('collect',[sys.executable,str(R/programs[0])]),('independent32-replay',[sys.executable,str(R/programs[1]),str(COLLECT/(NAME+'-attempt01.tar.gz')),'--fixtures',str(R/'kernel-batch07-controller/fixtures.json'),'--output',str(COLLECT/'independent32-profile-verification.json')]),('phase-intervals',[sys.executable,str(R/programs[2])]),('CPU-stacks',[sys.executable,str(R/programs[3])])]
  for label,cmd in cmds:
   with (O/(label+'.log')).open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   steps.append({'step':label,'exit':p.returncode});write('progress.json',steps);assert p.returncode==0,label+' failed; original evidence preserved'
  proof=json.loads((COLLECT/'independent32-profile-verification.json').read_bytes());assert proof['complete_profiles']==32 and proof['HTTP_requests_replayed']==6144
  write('verified-scope.json',{'profiles':32,'HTTP_requests_replayed':6144,'phase_intervals_and_actual_CPU_stacks_analyzed':True,'serving_performance':'미실행; diagnostic only','goal_achieved':False})
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'steps':steps,'adopted':False,'goal_achieved':False})
