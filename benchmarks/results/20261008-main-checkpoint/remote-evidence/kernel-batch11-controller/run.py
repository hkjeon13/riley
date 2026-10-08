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
code = root/'kernel-batch11-controller'
plan = json.loads((code/'batch-plan.json').read_text())
fixtures = json.loads((code/'fixtures.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--attempt', type=int, required=True)
parser.add_argument('--mode',choices=['quiet','diagnostic'],required=True)
args = parser.parse_args()
assert args.attempt > 0
out = root/f'kernel-batch11-{args.mode}-attempt{args.attempt:02d}'
out.mkdir()
(out/'controller-snapshot.py').write_bytes(Path(__file__).read_bytes())
(out/'plan-snapshot.json').write_bytes((code/'batch-plan.json').read_bytes())
model = Path('/data/riley-serving-260913-recovery/runtime-assets-20260915/model')
tool = Path('/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13')
env = {'HOME': '/home/psyche',
       'PATH': str(root/'vllm0271-venv/bin')+':'+str(tool/'bin')+':/usr/bin:/bin',
       'CUDA_HOME': str(tool), 'LANG': 'C.UTF-8',
       'CPATH': str(root/'curand-only-include'),
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
    before=time.perf_counter_ns()
    epoch=time.time_ns();monotonic=time.monotonic_ns();raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    after=time.perf_counter_ns()
    clocks={'perf_counter_before_ns':before,'perf_counter_after_ns':after,'epoch_ns':epoch,'monotonic_ns':monotonic,'monotonic_raw_ns':raw,'perf_counter_implementation':time.get_clock_info('perf_counter').implementation}
    return {'time_ns': epoch, 'clocks':clocks, 'pressure': {k: Path('/proc/pressure',k).read_text()
            for k in ['cpu','io','memory']}, 'loadavg': Path('/proc/loadavg').read_text(),
            'gpu': smi('uuid,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu')}

def process_group_memory(group):
    rows=subprocess.check_output(['ps','-e','-o','pid=,pgid=,rss='],text=True)
    observed={}
    for row in rows.splitlines():
        pid,pgid,rss=map(int,row.split())
        if pgid==group:observed[str(pid)]=rss
    return {'rss_KiB_by_pid':observed,'rss_sum_KiB':sum(observed.values()),
            'scope':'instantaneous process-group RSS sum; shared pages may count more than once; sampled each second'}

def observed_host_gate(prefix):
    policy = plan['host_start_gate']; records=[]; streak=0
    deadline = time.monotonic()+policy['interval_seconds']*(policy['consecutive']+1)
    while time.monotonic()<deadline:
        h=host(); records.append(h)
        def avg(resource, kind):
            row=next(x for x in h['pressure'][resource].splitlines() if x.startswith(kind+' '))
            return float(row.split()[1].split('=')[1])
        quiet = (avg('cpu','some')<=policy['cpu_some_max'] and
                 avg('io','full')<=policy['io_full_max'] and
                 avg('memory','full')<=policy['memory_full_max'])
        streak=streak+1 if quiet else 0
        if len(records)>=policy['consecutive']:
            write(prefix+'-host-gate.json', {'passed':streak>=policy['consecutive'],'policy':policy,'samples':records,'pressure_policy':args.mode,'performance_claim_eligible':False});return
        time.sleep(policy['interval_seconds'])
    write(prefix+'-host-gate.json', {'passed':False,'policy':policy,'samples':records})
    raise RuntimeError('whole attempt aborted: host start gate timeout '+prefix)

def quiet_host_gate(prefix):
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

def host_gate(prefix):
    return quiet_host_gate(prefix) if args.mode=='quiet' else observed_host_gate(prefix)

def ready(p, port):
    deadline=time.monotonic()+plan['startup_http_ready_timeout_seconds']
    while p.poll() is None and time.monotonic()<deadline:
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
        try:
            c.request('GET','/v1/models'); response=c.getresponse();body=response.read()
            if response.status==200 and any(x.get('id')=='g04-smol' for x in json.loads(body).get('data',[])):return
        except (OSError,http.client.HTTPException):pass
        finally:c.close()
        time.sleep(.1)
    raise RuntimeError(f'server early exit {p.returncode}' if p.poll() is not None else f"HTTP readiness timeout after {plan['startup_http_ready_timeout_seconds']} seconds")

def stop(p):
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=40)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)

binary={'v52':root/'v52-target/release/riley','candidate':root/'kernel-batch11-target-attempt01/release/riley','vllm':root/'vllm0271-venv/bin/vllm'}
pins={str(p):sha(p) for p in [*binary.values(),*model.iterdir(),*code.glob('*.py'),
                                  code/'fixtures.json',code/'batch-plan.json',code/'kernel-batch11-native-http-independent-verification.json',root/'vllm0271-packages.txt',root/'curand-header-recovery.json',*sorted((root/'curand-only-include').glob('*.h'))] if p.is_file()}
validation=json.loads((root/'kernel-batch11-server-validation-attempt01/completion.json').read_text())
http_screen=json.loads((root/'kernel-batch11-http-screen-attempt01/completion.json').read_text())
binary_receipt=json.loads((root/'kernel-batch11-server-validation-attempt01/binary-receipt.json').read_text())
assert validation['failure'] is None, 'candidate server/build/HTTP validation incomplete'
primitive=json.loads((root/'kernel-batch11-primitive-validation-attempt01/completion.json').read_text())
native=json.loads((root/'kernel-batch11-primitive-validation-attempt01/native-process.json').read_text())
assert primitive['failure'] is None and native['exit']==0, 'candidate primitive validation failed'
proof=json.loads((code/'kernel-batch11-native-http-independent-verification.json').read_text())
assert sha(code/'kernel-batch11-native-http-independent-verification.json')==plan['independent_correctness_sha256'], 'independent proof changed'
assert proof['native_exact_cases']==3161 and proof['prefill_gate_exact_cases']==120 and proof['fusion_exact_cases']==272 and proof['HTTP_raw_requests_replayed']==60 and proof['source_commit']==plan['candidate_source_commit'], 'independent native/HTTP qualification incomplete'
assert proof['native_binary_sha256']==native['binary_after'], 'native qualifier binary mismatch'
write('primitive-validation-snapshot.json',primitive)
write('native-process-snapshot.json',native)
(out/'independent-correctness-snapshot.json').write_bytes((code/'kernel-batch11-native-http-independent-verification.json').read_bytes())
assert http_screen['failure'] is None and http_screen['requests']==60 and len(http_screen['records'])==8 and all(s['strict_reference_pass'] for s in http_screen['records']), 'candidate HTTP exact screen incomplete'
assert binary_receipt['source_commit']==plan['candidate_source_commit'] and binary_receipt['sha256']==pins[str(binary['candidate'])], 'validated candidate binary identity differs'
assert binary_receipt['sha256']==plan['candidate_binary_sha256'], 'predeclared candidate digest mismatch'
assert pins[str(binary['v52'])]=='6f424278437f82a462ee141c829914d02ccb8c02ed604fa166c70c3c70bb16f6', 'reviewed V52 digest mismatch'
write('validation-snapshot.json',validation)
write('http-screen-snapshot.json',http_screen)
write('binary-receipt-snapshot.json',binary_receipt)
write('preparation.json', {'plan':plan,'pins':pins,'host':host(),
    'source_commits':{'v52':'06302d8d8396b8f2f4996fec8595bbfa1dcd7450','candidate':plan['candidate_source_commit']},
    'controller_path':str(Path(__file__)), 'pressure_policy':args.mode, 'purpose':'matched three-engine candidate comparison; goal eligibility requires independent replay, quality and stability gates',
    'rebuilt_binary_sha256':pins[str(binary['v52'])], 'historical_binary_sha256':'d24783e7aab9e5ffbf7489e720a54122b967c5274131b385c3c0f405b3fbe07d',
    'software_packages':(root/'vllm0271-packages.txt').read_text(),
    'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True),
    'qualification':False, 'plan_path':str(code/'batch-plan.json')})
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
                    if name in ('v52','candidate'):
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
                        while not done.wait(1):
                            observation=host();observation['server_process_memory']=process_group_memory(p.pid);samples.append(observation)
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
                                    if name in ('v52','candidate'):assert account['strict_reference_pass'],'Riley exact reference failed'
                                    if phase=='retained':
                                        write(prefix+'-summary.json',summary)
                                        records.append({'case':prefix,'summary':summary,'strict_reference_pass':account['strict_reference_pass']})
                        finally:
                            before_stop_memory=process_group_memory(p.pid)
                            stop(p);done.set();t.join();write(prefix+'-host.json',samples)
                            write(prefix+'-exit.json',{'exit':p.returncode,'host':host(),'before_stop_process_memory':before_stop_memory})
                    assert not smi('pid','compute-apps'),'owned GPU allocation not reclaimed'
                    assert all(float(h['gpu'].split(',')[4])*1024**2<=plan['gpu_operating_cap_bytes'] for h in samples),'GPU operating cap exceeded'
                    write('progress.json',records);print(prefix+' complete',flush=True)
except BaseException as e:
    failure={'type':type(e).__name__,'message':str(e)};raise
finally:
    write('completion.json',{'records':records,'failure':failure,'all_lanes_complete':len(records)==96,'goal_achieved':False,'performance_qualified':False})
