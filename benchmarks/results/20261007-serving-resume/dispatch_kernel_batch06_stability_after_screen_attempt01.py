"""Actual all8 GPU lifecycle gate only after an independently verified full screen."""
import hashlib,json,pathlib,subprocess,shlex,time
R=pathlib.Path(__file__).resolve().parent;O=R/'kernel-batch06-stability-local-dispatch-attempt01';O.mkdir();failure=None;launched=False;cases=[];decision=None
WAIT=R/'kernel-batch06-independent-replay-attempt02/completion.json';SCREEN=WAIT.with_name('summary.json');REMOTE='/data/riley-serving-261007';CODE=REMOTE+'/kernel-batch06-stability-controller-attempt01';OUT=REMOTE+'/kernel-batch06-stability-attempt01'
items=[(R/'run_serving_stability.py','run_serving_stability.py'),(R/'kernel-batch06-controller/serving_token_client_v2.py','serving_token_client_v2.py'),(R/'kernel-batch06-controller/mixed_phase_v34.py','mixed_phase_v34.py'),(R/'kernel-batch06-controller/fixtures.json','fixtures.json')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();pins={n:sha(p) for p,n in items}
def remote(code):return subprocess.check_output(['ssh','-o','ServerAliveInterval=20','ai-assistant','python3 -c '+shlex.quote(code)],text=True)
(O/'preparation.json').write_text(json.dumps({'wait_for':str(WAIT),'input_sha256':pins,'cells':[{'concurrency':c,'workload':w} for c in [1,8,16,32] for w in ['fixed','natural']],'steady_seconds_each':600,'cancel_cycles_each':5,'cancel_stages':['after_headers_before_consuming_tokens','after_first_token'],'rerequest_wave_each':96,'scope':'source-owned live metrics plus every raw HTTP SSE; native/CPU mock results never substitute','minimum_trigger':'full96 independent source/model/launch audit, quiet all8 mean screen PASS','serving_performance':'unexecuted until branch actually launched','goal_achieved':False},indent=2)+'\n')
try:
 deadline=time.monotonic()+43200
 while not WAIT.exists():
  assert time.monotonic()<deadline,'screen remains incomplete; no GPU lifecycle launch'
  time.sleep(30)
 terminal=json.loads(WAIT.read_bytes());assert terminal['failure'] is None,'independent screen pipeline failed; no lifecycle launch'
 screen=json.loads(SCREEN.read_bytes());require_source='e42cc345a08913b2dc1f9a0ea618dd73e3390f8e';assert screen['source_commits']['candidate']==require_source and screen['pressure_policy']=='quiet'
 if not screen['matrix_complete'] or screen['all_core_minimum_mean_screen'] is not True:
  decision={'launched':False,'reason':'all8 quiet full96 screen incomplete or failed; no stability or adoption qualification','all_core_minimum_mean_screen':screen['all_core_minimum_mean_screen'],'matrix_complete':screen['matrix_complete']};(O/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 else:
  audit=json.loads(WAIT.with_name('launch-audit.json').read_bytes());assert audit['requests_audited']==46080 and audit['launches_audited']==96 and audit['request_counts_equal_by_index'] is True and audit['frozen_cli_env_binary_model_tokenizer_software_pins_passed'] is True and audit['candidate_source_commit']==require_source and audit['candidate_binary_sha256']=='1ea04ca2b125e786cd925a57358cb7e245df98056913c4ebf73b4cc085b154a2','launch audit scope/source differs'
  assert all(sha(p)==pins[n] for p,n in items),'lifecycle inputs changed after declaration'
  state=json.loads(remote("import json,subprocess;from pathlib import Path;print(json.dumps({'GPU':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'existing':[n for n in "+repr([CODE,OUT])+" if Path(n).exists()]}))"));assert not state['GPU'] and not state['existing']
  subprocess.run(['ssh','ai-assistant','mkdir',CODE,OUT],check=True)
  for p,n in items:subprocess.run(['scp',str(p),'ai-assistant:'+CODE+'/'+n],check=True)
  subprocess.run(['scp',str(SCREEN),'ai-assistant:'+CODE+'/qualified-screen.json'],check=True)
  for c in [1,8,16,32]:
   for w in ['fixed','natural']:
    label=f'c{c}-{w}';launch=REMOTE+f'/kernel-batch06-quiet-attempt02/{label}-r0-candidate-launch.json';argv=[REMOTE+'/vllm0271-venv/bin/python',CODE+'/run_serving_stability.py','--launch',launch,'--fixtures',CODE+'/fixtures.json','--workload',w,'--screen',CODE+'/qualified-screen.json','--out',OUT+'/'+label,'--steady-seconds','600'];launched=True
    with (O/(label+'-remote.log')).open('x') as log:proc=subprocess.run(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant',shlex.join(argv)],stdout=log,stderr=subprocess.STDOUT)
    actual=json.loads(remote("from pathlib import Path;print(Path("+repr(OUT+'/'+label+'/completion.json')+").read_text())"));case={'case':label,'SSH_exit':proc.returncode,'actual_completion':actual};cases.append(case);(O/'progress.json').write_text(json.dumps(cases,indent=2)+'\n');assert proc.returncode==0 and actual['failure'] is None and actual['native_shutdown_reclaimed'],'actual lifecycle failed; preserve all preceding evidence, no retry'
  decision={'launched':True,'actual_GPU_cases_completed':len(cases),'independent_raw_replay':'pending; controller PASS alone is not qualification'};(O/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'launched':launched,'decision':decision,'cases':cases,'adopted':False,'goal_achieved':False},indent=2)+'\n')
