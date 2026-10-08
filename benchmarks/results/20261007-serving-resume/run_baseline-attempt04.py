"""Create-only baseline attempt using the historical token-aware HTTP client."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from serving_token_client_v2 import TokenHttpClient, TokenReference
from mixed_phase_v34 import mixed_phase

root = Path('/data/riley-serving-261007')
code = root/'controller'
plan = json.loads((code/'baseline-plan.json').read_text())
fixtures = json.loads((code/'fixtures.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--attempt', type=int, required=True)
args = parser.parse_args()
assert args.attempt > 0
out = root/f'baseline-attempt{args.attempt:02d}'
out.mkdir()
(out/'controller-snapshot.py').write_bytes(Path(__file__).read_bytes())
model = Path('/data/riley-serving-260913-recovery/runtime-assets-20260915/model')
tool = Path('/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13')
env = {'HOME': '/home/psyche',
       'PATH': str(root/'vllm0271-venv/bin')+':'+str(tool/'bin')+':/usr/bin:/bin',
       'CUDA_HOME': str(tool), 'LANG': 'C.UTF-8',
       'LC_ALL': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '0', 'VLLM_BATCH_INVARIANT': '0',
       'LD_LIBRARY_PATH': str(tool/'lib')}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write(name, value):
    (out/name).write_text(json.dumps(value, indent=2)+'\n')

def smi(query, kind='gpu'):
    return subprocess.check_output(['nvidia-smi', '--query-'+kind+'='+query,
                                   '--format=csv,noheader,nounits'], text=True).strip()

def host():
    return {'time_ns': time.time_ns(), 'pressure': {k: Path('/proc/pressure',k).read_text()
            for k in ['cpu','io','memory']}, 'loadavg': Path('/proc/loadavg').read_text(),
            'gpu': smi('uuid,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu')}

def host_gate(prefix):
    policy = plan['host_start_gate']; records=[]; streak=0
    deadline = time.monotonic()+policy['timeout_seconds']
    while time.monotonic()<deadline:
        h=host(); records.append(h)
        def avg(resource, kind):
            row=next(x for x in h['pressure'][resource].splitlines() if x.startswith(kind+' '))
            return float(row.split()[1].split('=')[1])
        quiet = (avg('cpu','some')<=policy['cpu_some_max'] and
                 avg('io','full')<=policy['io_full_max'] and
                 avg('memory','full')<=policy['memory_full_max'])
        streak=streak+1 if quiet else 0
        if streak>=policy['consecutive']:
            write(prefix+'-host-gate.json', {'passed':True,'policy':policy,'samples':records});return
        time.sleep(policy['interval_seconds'])
    write(prefix+'-host-gate.json', {'passed':False,'policy':policy,'samples':records})
    raise RuntimeError('whole attempt aborted: host start gate timeout '+prefix)

def ready(p, port):
    deadline=time.monotonic()+120
    while p.poll() is None and time.monotonic()<deadline:
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
        try:
            c.request('GET','/v1/models'); response=c.getresponse();body=response.read()
            if response.status==200 and any(x.get('id')=='g04-smol' for x in json.loads(body).get('data',[])):return
        except (OSError,http.client.HTTPException):pass
        finally:c.close()
        time.sleep(.1)
    raise RuntimeError('startup timeout or early exit')

def stop(p):
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=40)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)

binary={'v52':root/'v52-target/release/riley','vllm':root/'vllm0271-venv/bin/vllm'}
pins={str(p):sha(p) for p in [*binary.values(),*model.iterdir(),*code.glob('*.py'),
                                  code/'fixtures.json',code/'baseline-plan.json',root/'vllm0271-packages.txt'] if p.is_file()}
write('preparation.json', {'plan':plan,'pins':pins,'host':host(),
    'source_commit':'06302d8d8396b8f2f4996fec8595bbfa1dcd7450',
    'rebuilt_binary_sha256':pins[str(binary['v52'])], 'historical_binary_sha256':'d24783e7aab9e5ffbf7489e720a54122b967c5274131b385c3c0f405b3fbe07d',
    'software_packages':(root/'vllm0271-packages.txt').read_text(),
    'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True),
    'qualification':False})
records=[]; failure=None
try:
    for concurrency in plan['concurrency']:
        for workload in plan['workloads']:
            data=fixtures[workload];corpus=data['corpus']
            refs=[TokenReference('g04-smol',tuple(b['choices'][0]['prompt_token_ids']) if 'prompt_token_ids' in b['choices'][0] else tuple(c['prompt_token_ids']),
                   tuple(b['choices'][0]['token_ids']),b['choices'][0]['text'],b['choices'][0]['finish_reason'])
                   for c,b in zip(corpus,data['responses'])]
            for repeat,order in enumerate(plan['orders']):
                for name in order:
                    prefix=f'c{concurrency}-{workload}-r{repeat}-{name}'
                    assert not smi('pid','compute-apps'),'foreign GPU compute process'
                    host_gate(prefix+'-start')
                    for p,h in pins.items():assert sha(p)==h, 'immutable input changed '+p
                    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
                    if name=='v52':
                        argv=[str(binary[name]),'serve','--model',str(model),'--model-id','g04-smol','--bind',f'127.0.0.1:{port}',
                              '--max-active-sequences',str(concurrency),'--max-waiting-requests','64','--batch-token-budget','512',
                              '--prefill-chunk-tokens',str(plan['chunk_tokens'][workload]),'--max-sequence-tokens',str(1024 if workload=='natural' else 160),
                              '--max-output-tokens',str(128 if workload=='natural' else 32),'--kv-blocks',str((64 if workload=='natural' else 10)*concurrency),
                              '--residual-rmsnorm','separate','--execution-completion','iteration-batch','--metadata-transport','synchronous',
                              '--execution-graph-policy','require','--graph-numerics','variable-smol-v7','--sampling-backend','gpu-greedy']
                    else:
                        argv=[str(binary[name]),'serve',str(model),'--tokenizer',str(model),'--served-model-name','g04-smol','--host','127.0.0.1','--port',str(port),
                              '--dtype','bfloat16','--max-model-len',str(1024 if workload=='natural' else 160),'--max-num-seqs',str(concurrency),
                              '--max-num-batched-tokens','512','--gpu-memory-utilization','0.3','--no-enable-prefix-caching','--seed','0']
                    write(prefix+'-launch.json',{'argv':argv,'env':env,'host':host(),'binary_sha256':sha(binary[name])})
                    done=threading.Event();samples=[]
                    def monitor():
                        while not done.wait(1):samples.append(host())
                    with (out/(prefix+'.log')).open('w') as log:
                        p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        t=threading.Thread(target=monitor);t.start()
                        try:
                            ready(p,port)
                            with TokenHttpClient() as client:
                                def request_one(index):
                                    i=index%len(corpus);c=corpus[i]
                                    row=client.request(port,{'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0},refs[i],streaming=True,mode='observe')
                                    row['corpus_id']=c['id'];return row
                                for phase,count in [('warmup',plan['warmup']),('retained',plan['retained'])]:
                                    host_gate(prefix+'-'+phase)
                                    rows,account,summary=mixed_phase(client,request_one,concurrency=concurrency,count=count,phase=phase,corpus_ids=[c['id'] for c in corpus])
                                    write(prefix+'-'+phase+'-rows.json',rows);write(prefix+'-'+phase+'-accounting.json',account)
                                    assert account['completed'],'transport incomplete'
                                    if name=='v52':assert account['strict_reference_pass'],'Riley exact reference failed'
                                    if phase=='retained':
                                        write(prefix+'-summary.json',summary)
                                        records.append({'case':prefix,'summary':summary,'strict_reference_pass':account['strict_reference_pass']})
                        finally:
                            stop(p);done.set();t.join();write(prefix+'-host.json',samples)
                            write(prefix+'-exit.json',{'exit':p.returncode,'host':host()})
                    assert not smi('pid','compute-apps'),'owned GPU allocation not reclaimed'
                    assert all(float(h['gpu'].split(',')[4])*1024**2<=plan['gpu_operating_cap_bytes'] for h in samples),'GPU operating cap exceeded'
                    write('progress.json',records);print(prefix+' complete',flush=True)
except BaseException as e:
    failure={'type':type(e).__name__,'message':str(e)};raise
finally:
    write('completion.json',{'records':records,'failure':failure,'all_lanes_complete':len(records)==64,'goal_achieved':False,'performance_qualified':False})
