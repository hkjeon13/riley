"""Run all8 diagnostic profiles only after terminal, independently verified FAIL."""
import hashlib,json,pathlib,shlex,subprocess,time
R=pathlib.Path(__file__).resolve().parent
O=R/'kernel-batch07-cpu-capture-all8-profile-local-dispatch-attempt01';O.mkdir()
NAME='kernel-batch07-cpu-capture-all8-profile'
WAIT=R/'kernel-batch07-independent-replay-attempt02/completion.json'
STABILITY=R/'kernel-batch07-stability-local-dispatch-attempt01/completion.json'
POLICY=R/(NAME+'-policy-attempt01.json')
DRIVER=R/'run_kernel_batch07_cpu_capture_all8_profile_attempt01.py'
REMOTE='/data/riley-serving-261007'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
def remote(code):return subprocess.check_output(['ssh','-o','ServerAliveInterval=20','ai-assistant','python3 -c '+shlex.quote(code)],text=True)
pins={p.name:sha(p) for p in [POLICY,DRIVER]}
write('preparation.json',{'source_pins':pins,'wait_for':str(WAIT),'expected_profiles':32,'warmup_each':96,'retained_each':96,'no_exclusions':True,'trigger':'terminal full96 independent source/launch/raw audit and all8 minimum screen FAIL; otherwise no profile GPU launch'})
failure=None;launched=False;decision=None
try:
 deadline=time.monotonic()+43200
 while not WAIT.exists() or not STABILITY.exists():
  assert time.monotonic()<deadline,'original screen/decision still pending; no restart or profile launch'
  time.sleep(30)
 terminal=json.loads(WAIT.read_bytes());assert terminal['failure'] is None,'independent screen failed; preserve evidence, no profiling'
 screen=json.loads(WAIT.with_name('summary.json').read_bytes())
 assert screen['matrix_complete'] and screen['verified_lane_count']==96 and screen['pressure_policy']=='quiet'
 if screen['all_core_minimum_mean_screen'] is True:
  decision={'launched':False,'reason':'all8 screen PASS; sustained lifecycle has priority; no diagnostic GPU launch'}
 else:
  assert screen['all_core_minimum_mean_screen'] is False
  stability=json.loads(STABILITY.read_bytes());assert stability['failure'] is None and stability['launched'] is False,'lifecycle branch overlaps or failed'
  policy=json.loads(POLICY.read_bytes());source=policy['candidate_source_commit']
  assert screen['source_commits']['candidate']==source and all(sha(p)==pins[p.name] for p in [POLICY,DRIVER])
  audit=json.loads(WAIT.with_name('launch-audit.json').read_bytes())
  assert audit['launches_audited']==96 and audit['requests_audited']==46080 and audit['request_counts_equal_by_index'] and audit['frozen_cli_env_binary_model_tokenizer_software_pins_passed'] and audit['candidate_source_commit']==source and audit['candidate_binary_sha256']==policy['expected_candidate_binary_sha256']
  proof=WAIT.with_name('raw-replay.json');raw=json.loads(proof.read_bytes());assert raw['raw_replay_passed'] and raw['matrix_complete'] and len(raw['lanes'])==96
  collection=R/'kernel-batch07-terminal-collection-attempt01'
  collected=json.loads((collection/'completion.json').read_bytes());assert collected['failure'] is None and len(collected['records'])==1
  record=collected['records'][0];archive=pathlib.Path(record['archive']);assert sha(archive)==record['archive_sha256']
  before=json.loads((collection/'source-manifest-before.json').read_bytes());after=json.loads((collection/'source-manifest-after.json').read_bytes());assert before==after
  files=before['kernel-batch07-quiet-attempt01']['files']
  names=['preparation.json','completion.json']+[f'c{c}-{w}-r0-{e}-launch.json' for c in [1,8,16,32] for w in ['fixed','natural'] for e in ['v52','candidate']]
  policy.update(baseline_source_file_shas={n:files[n]['sha256'] for n in names},baseline_verification_file_sha256=sha(proof),verified_serving_archive_sha256=record['archive_sha256'])
  resolved=O/(NAME+'-plan-attempt01.json')
  with resolved.open('x') as f:json.dump(policy,f,indent=2)
  items=[(DRIVER,DRIVER.name),(resolved,resolved.name),(proof,NAME+'-independent-serving-screen.json')]
  targets=[REMOTE+'/'+n for p,n in items]+[REMOTE+'/'+NAME+'-attempt01']
  state=json.loads(remote("import json,subprocess;from pathlib import Path;print(json.dumps({'GPU':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'existing':[n for n in "+repr(targets)+" if Path(n).exists()]}))"))
  assert not state['GPU'] and not state['existing'],'GPU occupied or create-only destination exists; leave all actors/evidence unchanged'
  for p,n in items:subprocess.run(['scp',str(p),'ai-assistant:'+REMOTE+'/'+n],check=True)
  expected={n:sha(p) for p,n in items}
  remote("import hashlib;from pathlib import Path;expected="+repr(expected)+";assert all(hashlib.sha256((Path("+repr(REMOTE)+")/n).read_bytes()).hexdigest()==h for n,h in expected.items())")
  argv=['sudo','-n',REMOTE+'/vllm0271-venv/bin/python',REMOTE+'/'+DRIVER.name,'--attempt','1','--plan',REMOTE+'/'+resolved.name,'--verified-baseline',REMOTE+'/'+NAME+'-independent-serving-screen.json']
  launched=True;write('launch.json',{'argv':argv,'files_SHA256':expected,'serving_performance':'미실행; diagnostic only'})
  with (O/'remote-controller.log').open('x') as f:p=subprocess.run(['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','ai-assistant',shlex.join(argv)],stdout=f,stderr=subprocess.STDOUT)
  write('SSH-exit.json',{'exit':p.returncode,'actual_terminal_inferred':False});assert p.returncode==0,'inspect original remote process/terminal; never restart from observation failure'
  decision={'launched':True,'expected_profiles':32,'independent_replay':'pending'}
 write('decision.json',decision)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'launched':launched,'decision':decision,'adopted':False,'goal_achieved':False})
