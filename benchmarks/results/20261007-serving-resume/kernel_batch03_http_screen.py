"""Correctness-only first HTTP screen of candidate e131915fd6648c7ce999b8ed9f807a36050063c0; not concurrent-load proof."""
import hashlib, http.client, json, os
from pathlib import Path
import signal, socket, subprocess, time
from serving_token_client_v2 import TokenHttpClient, TokenReference

root=Path('/data/riley-serving-261007');out=root/'kernel-batch03-http-screen-attempt01';out.mkdir()
fixtures=json.loads((root/'controller/fixtures.json').read_text())
binary=root/'kernel-batch03-target/release/riley'
model='/data/riley-serving-260913-recovery/runtime-assets-20260915/model'
env={'HOME':'/home/psyche','PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'0',
     'LD_LIBRARY_PATH':'/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13/lib'}
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
records=[];failure=None
try:
 for capacity in [1,8,16,32]:
  for workload in ['fixed','natural']:
   assert not gpu(),'foreign GPU compute process'
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   natural=workload=='natural';name=f'c{capacity}-{workload}'
   argv=[str(binary),'serve','--model',model,'--model-id','g04-smol','--bind',f'127.0.0.1:{port}',
     '--max-active-sequences',str(capacity),'--max-waiting-requests','64','--batch-token-budget','512',
     '--prefill-chunk-tokens',str(512 if natural else 128),'--max-sequence-tokens',str(1024 if natural else 160),
     '--max-output-tokens',str(128 if natural else 32),'--kv-blocks',str(capacity*(64 if natural else 10)),
     '--residual-rmsnorm','separate','--execution-completion','iteration-batch','--metadata-transport','synchronous',
     '--execution-graph-policy','require','--graph-numerics','variable-smol-v7','--sampling-backend','gpu-greedy']
   write(name+'-launch.json',{'argv':argv,'env':env,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()})
   with (out/(name+'.log')).open('w') as log:
    p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
     deadline=time.monotonic()+120
     while p.poll() is None and time.monotonic()<deadline:
      conn=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
      try:
       conn.request('GET','/v1/models');response=conn.getresponse();body=response.read()
       if response.status==200 and any(x.get('id')=='g04-smol' for x in json.loads(body).get('data',[])):break
      except (OSError,http.client.HTTPException):pass
      finally:conn.close()
      time.sleep(.1)
     else:raise RuntimeError('startup failed '+name)
     rows=[]
     with TokenHttpClient() as client:
      for c,b in zip(fixtures[workload]['corpus'],fixtures[workload]['responses']):
       choice=b['choices'][0]
       ref=TokenReference('g04-smol',tuple(choice.get('prompt_token_ids',c.get('prompt_token_ids',[]))),tuple(choice['token_ids']),choice['text'],choice['finish_reason'])
       rows.append(client.request(port,{'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0},ref,streaming=True,mode='observe'))
       write(name+'-rows.json',rows)
       assert rows[-1]['status']=='success' and rows[-1]['reference_match'],'exact HTTP reference mismatch '+name
     records.append({'case':name,'passed':len(rows),'strict_reference_pass':True});print(name+' correctness passed',flush=True)
    finally:
     if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=40)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
     write(name+'-exit.json',{'exit':p.returncode,'gpu_compute_after':gpu()})
   assert not gpu(),'GPU resource reclamation failed'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)};raise
finally:
 write('completion.json',{'records':records,'failure':failure,'requests':sum(x['passed'] for x in records),
       'correctness_only':True,'serving_performance':'미실행','performance_qualified':False,'goal_achieved':False})
