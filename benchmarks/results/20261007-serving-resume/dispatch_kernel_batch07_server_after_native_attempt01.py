import pathlib,json,hashlib,subprocess,time,shlex
R=pathlib.Path(__file__).resolve().parent;O=R/'kernel-batch07-server-local-dispatch-attempt01';O.mkdir();failure=None
items=[(R/'run_kernel_batch07_server_attempt01.py','run_kernel_batch07_server_attempt01.py'),(R/'kernel_batch07_http_screen_attempt01.py','controller/kernel_batch07_http_screen_attempt01.py')]
try:
 assert json.loads((R/'kernel-batch07-local-materialization-attempt01.json').read_bytes())['verified_source_files']==468
 (O/'preparation.json').write_text(json.dumps({'inputs':{n:hashlib.sha256(p.read_bytes()).hexdigest() for p,n in items},'wait_for':'actual native attempt01 completion, not SSH observation','serving_performance':'미실행'},indent=2)+'\n')
 deadline=time.monotonic()+1800
 while True:
  script="import json;from pathlib import Path;p=Path('/data/riley-serving-261007/kernel-batch07-primitive-validation-attempt01/completion.json');print(p.read_text() if p.exists() else 'null')"
  state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(script)],text=True))
  if state is not None:break
  assert time.monotonic()<deadline,'native still active; leave original process, no restart'
  time.sleep(20)
 assert state['failure'] is None,'native failed; no build/HTTP launch'
 probe="import json,subprocess;from pathlib import Path;print(json.dumps({'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'existing':[n for n in %r if (Path('/data/riley-serving-261007')/n).exists()]}))"%[n for p,n in items]
 pre=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(probe)],text=True));assert not pre['gpu'] and not pre['existing']
 for p,n in items:subprocess.run(['scp',str(p),'ai-assistant:/data/riley-serving-261007/'+n],check=True)
 with (O/'remote-server.log').open('x') as f:proc=subprocess.run(['ssh','-o','ServerAliveInterval=20','ai-assistant','/data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/run_kernel_batch07_server_attempt01.py'],stdout=f,stderr=subprocess.STDOUT)
 (O/'SSH-exit.json').write_text(json.dumps({'exit':proc.returncode,'actual_terminal_inferred':False})+'\n');assert proc.returncode==0,'inspect actual controller terminal; never restart on observation errors'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','goal_achieved':False},indent=2)+'\n')
