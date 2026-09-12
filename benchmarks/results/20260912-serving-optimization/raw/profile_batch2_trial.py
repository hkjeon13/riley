import importlib.util,json,os,subprocess,time,http.client
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--label',default='diagnostic-batch2');p.add_argument('--projection',choices=['off','q','gate','down'],default='off');a=p.parse_args();label=a.label;output_label=label+('-'+a.projection if a.projection!='off' else '')
root=Path('/tmp/riley-opt-260912')
spec=importlib.util.spec_from_file_location('old_runner','/tmp/riley-g04-gui-measurement-260911/measure-gui.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
plan=json.loads(Path('/tmp/riley-opt-260912/batch2-http-plan.json').read_text());lane=plan['http_lanes']['riley'];request=json.loads(Path('/tmp/riley-g04-vllm-profile-260911/request.json').read_text());lane['argv'][0]=str(root/(label+'-target/release/riley'))
env=os.environ.copy();env.update(lane['env']);env['RILEY_OWNED_GRAPH_PROFILE']='1';env['RILEY_OWNED_GRAPH_PROJECTION']=a.projection
telemetry=(root/(output_label+'-telemetry.csv')).open('x')
probe=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used','--format=csv','--loop-ms=200'],stdout=telemetry,stderr=subprocess.STDOUT)
with (root/(output_label+'-trial.log')).open('x') as log:
 proc=subprocess.Popen(lane['argv'],env=env,cwd=root/(label+'-source'),stdout=log,stderr=log)
 try:
  deadline=time.monotonic()+180
  while True:
   if proc.poll() is not None:raise RuntimeError('profile server exited')
   try:
    conn=http.client.HTTPConnection('127.0.0.1',lane['port'],timeout=2);conn.request('GET','/v1/models');resp=conn.getresponse();resp.read();conn.close()
    if resp.status==200:break
   except OSError:pass
   if time.monotonic()>deadline:raise TimeoutError('profile startup')
   time.sleep(.2)
  rows=[]
  for i in range(6):
   row=runner.http_request(lane['port'],request,i!=0)
   assert row['text']==lane['expected_output_text'] and row['finish_reason']=='length'
   rows.append({'request_index':i,'warmup':i<3,'text_exact':True,'finish_reason':row['finish_reason']})
  proc.terminate();proc.wait(timeout=30);assert proc.returncode==0
  (root/(output_label+'-validation.json')).write_text(json.dumps({'instrumented':True,'performance_claim_eligible':False,'requests':rows,'binary':lane['argv'][0],'exit_code':proc.returncode},indent=2))
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
  probe.terminate();probe.wait(timeout=10);telemetry.close()
