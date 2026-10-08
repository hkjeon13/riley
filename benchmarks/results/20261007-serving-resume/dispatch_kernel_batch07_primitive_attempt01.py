import pathlib,json,hashlib,subprocess,time,shlex
R=pathlib.Path(__file__).resolve().parent;O=R/'kernel-batch07-primitive-local-dispatch-attempt01';O.mkdir();failure=None
items=[R/n for n in ['kernel-batch07-source-attempt01.tar','kernel-batch07-commit-attempt01.pack','kernel-batch07-source-receipt-attempt01.json','run_kernel_batch07_primitive_attempt01.py']]
try:
 assert json.loads((R/'kernel-batch07-local-materialization-attempt01.json').read_bytes())['verified_source_files']==468
 (O/'preparation.json').write_text(json.dumps({'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in items},'serving_performance':'미실행'},indent=2)+'\n')
 names=[p.name for p in items]+['kernel-batch07-primitive-validation-attempt01','kernel-batch07-source-attempt01']
 code="import json,subprocess;from pathlib import Path;print(json.dumps({'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'existing':[n for n in %r if (Path('/data/riley-serving-261007')/n).exists()]}))"%names
 state=json.loads(subprocess.check_output(['ssh','ai-assistant','python3 -c '+shlex.quote(code)],text=True));assert not state['gpu'] and not state['existing'],state
 for p in items:subprocess.run(['scp',str(p),'ai-assistant:/data/riley-serving-261007/'+p.name],check=True)
 code="import hashlib,json;from pathlib import Path;expected=%r;assert all(hashlib.sha256((Path('/data/riley-serving-261007')/n).read_bytes()).hexdigest()==h for n,h in expected.items())"%{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in items}
 subprocess.run(['ssh','ai-assistant','python3 -c '+shlex.quote(code)],check=True)
 with (O/'remote-native.log').open('x') as f:p=subprocess.run(['ssh','-o','ServerAliveInterval=20','ai-assistant','/data/riley-serving-261007/vllm0271-venv/bin/python /data/riley-serving-261007/run_kernel_batch07_primitive_attempt01.py'],stdout=f,stderr=subprocess.STDOUT)
 (O/'SSH-exit.json').write_text(json.dumps({'exit':p.returncode,'actual_terminal_inferred':False})+'\n');assert p.returncode==0,'inspect remote terminal; no restart solely for observation errors'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','goal_achieved':False},indent=2)+'\n')
