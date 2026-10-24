"""Phase 0 same-device runner: v19B measure-matched.py plus a batch-invariant
`native` mode and token-level arrival times. Strict owned arms keep the
original same-device eager assertion; native arms record (not assert) eager,
HF and cross-concurrency agreement."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import sys
import socket
import statistics
import subprocess
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def support_process(pid,memory_mib=None):
    proc=Path('/proc')/str(pid)
    try:
        tick=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
        if proc.stat().st_uid!=os.getuid():return False
        if str(pid)==os.environ['RILEY_GPU_HOLDER_PID']:
            s=json.loads((Path(os.environ['RILEY_GPU_HOLDER_LEASE_DIR'])/('gpu'+str(6 if args_gpu.endswith('55ee1e6a7') else 7)+'.status.json')).read_text())
            return tick==os.environ['RILEY_GPU_HOLDER_START_TICKS']==s['start_ticks'] and s['status']=='running' and s['phase']=='active' and time.time_ns()-s['updated_ns']<10**10
        from holder import admitted_process
        return admitted_process(Path(os.environ['RILEY_GPU_HOLDER_LEASE_DIR']),6 if args_gpu.endswith('55ee1e6a7') else 7,pid,memory_mib)
    except FileNotFoundError:return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--vllm-token-budget', type=int, default=576)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--candidate-binary')
    parser.add_argument('--server-override')
    parser.add_argument('--baseline-server')
    parser.add_argument('--baseline-extra',default='[]')
    parser.add_argument('--candidate-extra',default='[]')
    parser.add_argument('--native-reference',required=True)
    parser.add_argument('--native-binary')
    parser.add_argument('--native-extra',default='[]')
    parser.add_argument('--native-after-binary')
    parser.add_argument('--native-after-extra')
    parser.add_argument('--native-after-env',default='{}')
    parser.add_argument('--model', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--gpu', default='GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0')
    parser.add_argument('--concurrency', default='1,8,16,32')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--repeat-start',type=int,default=0)
    parser.add_argument('--keep', type=int, default=48)
    parser.add_argument('--warm', type=int, default=8)
    parser.add_argument('--warm-buckets', action='store_true', help='Finite identical full-reference warm waves at power-of-two decode counts; measured controls unchanged')
    parser.add_argument('--modes', default='iteration-batch,owned-graph')
    parser.add_argument('--workload', choices=['mixed', 'fixed', 'natural'], default='mixed')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--windows',type=int,default=1)
    parser.add_argument('--arm-handshake',action='store_true')
    parser.add_argument('--batch-policy',choices=['disabled','grouped-v1','lt-mlp-v2','bounded-gateup-sm80-v1','bounded-projections-sm80-v2'],default='disabled')
    parser.add_argument('--candidate-mode',default='bounded-projections-sm80-v1')
    parser.add_argument('--graph-mode',choices=['off','segments-v1'],default='off')
    parser.add_argument('--completion-mode',choices=['per-operation','iteration-batch'],default='iteration-batch')
    parser.add_argument('--bootstrap-eager',action='store_true')
    parser.add_argument('--inspect',action='store_true')
    args = parser.parse_args()
    assert 1<=args.windows<=10 and (not args.profile or args.windows==1)
    assert args.gpu in ['GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7','GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0']
    global args_gpu
    args_gpu=args.gpu
    assert support_process(int(os.environ['RILEY_GPU_HOLDER_PID']))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    reference = Path(args.reference)
    items = json.loads((reference / 'items.json').read_text())
    expected = json.loads((reference / 'hf.json').read_text())['rows']
    if args.bootstrap_eager:
        assert args.concurrency=='1' and args.modes=='owned-before' and args.graph_mode=='off' and args.batch_policy=='disabled' and not args.profile
        native_reference=dict(binary_sha256=sha(args.binary),rows=[dict(id=x['id'],value=None)for x in items])
    else:
        native_reference=json.loads(Path(args.native_reference).read_text())
    assert native_reference['binary_sha256']==sha(args.binary),'eager reference must bind before-change binary'
    native_expected=native_reference['rows']
    assert len(items) == len(expected) == len(native_expected) == 15
    assert [x['id'] for x in native_expected]==[x['id'] for x in items]
    for a, b in zip(items, expected):
        assert a['id'] == b['id']
    measured_items = [(i, item) for i, item in enumerate(items)
                      if args.workload == 'mixed' or item['workload'] == args.workload]
    assert measured_items
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
               CUDA_DEVICE_ORDER='PCI_BUS_ID', LD_LIBRARY_PATH='/usr/local/cuda-12.4/lib64',
               OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('PYTHONPATH', None)
    env.pop('RILEY_DIAGNOSTIC_CUDA_EVENT_SPIN', None)
    for key in ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','LD_PRELOAD']:env.pop(key,None)
    env.update(TMPDIR=str(root/'tmp'),HF_HOME=str(root/'cache/hf'),TORCHINDUCTOR_CACHE_DIR=str(root/'cache/torchinductor'),TRITON_CACHE_DIR=str(root/'cache/triton'))
    for path in ['tmp','cache']: (root/path).mkdir()
    if args.completion_mode=='per-operation':assert args.graph_mode=='off' and args.batch_policy=='disabled'
    binary = Path(args.binary)
    plan = dict(vars(args), runner_sha256=sha(__file__), binary_sha256=sha(binary),
                candidate_sha256=sha(args.candidate_binary or binary), native_binary_sha256=sha(args.native_binary or args.baseline_server or binary), native_extra_args=json.loads(args.native_extra), native_after_binary_sha256=sha(args.native_after_binary) if args.native_after_binary else None, native_after_extra=args.native_after_extra, native_after_env=json.loads(args.native_after_env), server_override_sha256=sha(args.server_override) if args.server_override else None, baseline_server_sha256=sha(args.baseline_server) if args.baseline_server else None, candidate_extra_args=['--owned-decode-batch',args.candidate_mode],
                checkpoint_sha256=sha(Path(args.model) / 'riley-checkpoint.json'),
                reference_sha256=sha(reference / 'hf.json'), native_reference_sha256=None if args.bootstrap_eager else sha(args.native_reference),
                numerical_gate='exact same-A100 eager tokens/text/prompt/finish/usage; fresh same-A100 HF generation asserted in bootstrap; teacher full-vocab separately',
                items_sha256=sha(reference / 'items.json'),
                started_ns=time.time_ns(), library_path=env['LD_LIBRARY_PATH'],
                scope=f'Qwen3-4B Phase 0 (A/A calibration and numerics-contract cost); inspect actual per-arm argv for budget and kernel configuration; {args.repeats} fresh-process repeats with {args.warm} warm/{args.keep} measured; response-mean TPOT is not token ITL; token ITL from per-token arrivals is recorded separately',
                vllm_version=subprocess.check_output(['/home/nlp/riley-261013/vllm0271-venv/bin/python',
                    '-c', 'import importlib.metadata as m; print(m.version("vllm"))'],text=True).strip(),
                diagnostic_sha256=None,
                persistent_holder=dict(pid=os.environ['RILEY_GPU_HOLDER_PID'],start_ticks=os.environ['RILEY_GPU_HOLDER_START_TICKS'],active_allocation_bytes=64*1024**2),
                authorized_coexistence_policy=json.loads((Path(os.environ['RILEY_GPU_HOLDER_LEASE_DIR'])/('gpu'+str(6 if args.gpu.endswith('55ee1e6a7') else 7)+'.policy.json')).read_text()),
                matched=dict(dtype='BF16', token_budget=f'per-arm actual argv; stock={args.vllm_token_budget}', kv_block_size=16,
                             kv_blocks='concurrency * 64', sequence_bound=1024, process_GPU_memory_cap_GiB=22, vllm_allocation_fraction=0.5,
                             cpu_affinity=('24-31' if args.gpu=='GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7' else '32-39'), prefix_caching=False,
                             sampling='greedy, repetition_penalty1.0, top_p1.0, top_k disabled, zero frequency/presence penalties, original generation EOS'),
                graph_policy=dict(riley=args.graph_mode, candidate_extra_graph='inspect candidate-extra and actual per-arm argv',batch=args.batch_policy,candidate_batch=args.candidate_mode,completion=args.completion_mode,vllm='installed stock default compile/Graph, no event-spin changes'))
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))
    (root / 'hardware.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    records = []
    canonical = {}
    native_canonical = {}
    native_invariance = []
    for concurrency_index,conc in enumerate(map(int, args.concurrency.split(','))):
        for repeat in range(args.repeat_start,args.repeats):
            modes = args.modes.split(',')
            if (repeat+concurrency_index) % 2:
                modes.reverse()
            for mode in modes:
                arm = root / f'c{conc}-r{repeat}-{mode}'
                arm.mkdir()
                occupied = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
                    '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
                (arm / 'admission.txt').write_text(occupied)
                if any(not support_process(int(row.split(',')[0]),float(row.rsplit(',',1)[1].strip().split()[0])) for row in occupied.splitlines() if row.strip()):
                    raise RuntimeError('Selected GPU occupied; no process terminated: ' + occupied)
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                exe = args.server_override or ((args.baseline_server or args.binary) if mode == 'owned-before' else (args.candidate_binary or args.binary))
                assert mode in {'owned-before','owned-after','vllm','native','native-after'},'owned-HF-route, native or stock only'
                is_native = mode in ('native', 'native-after')
                if is_native: exe = args.native_binary or args.baseline_server or args.binary
                if mode == 'native-after': exe = args.native_after_binary
                argv = [exe, 'serve', '--bind', f'127.0.0.1:{port}', '--model', args.model,
                        '--model-id', 'm', '--max-active-sequences', str(conc),
                        '--max-waiting-requests', '64', '--max-sequence-tokens', '1024',
                        '--max-output-tokens', '128', '--max-weight-bytes', str(32 << 30),
                        '--kv-blocks', str(conc * 64)]
                if is_native:
                    argv += (json.loads(Path(args.native_after_extra).read_text()) if mode == 'native-after' and args.native_after_extra else json.loads(args.native_extra))
                else:
                    argv += ['--numerics-profile', 'owned-hf-eager-bf16-v1',
                             '--execution-completion', args.completion_mode]
                if mode in ['owned-before','owned-after'] and args.graph_mode!='off':
                    argv += ['--owned-decode-graph',args.graph_mode]
                if mode in ['owned-before','owned-after']:
                    argv += ['--owned-decode-batch',args.batch_policy]
                if mode=="owned-before":argv[argv.index("--owned-decode-batch")+1]="native-prefill-sm80-v5"
                if mode=="owned-after":argv[argv.index("--owned-decode-batch")+1]=args.candidate_mode
                if mode=='owned-before':argv += json.loads(args.baseline_extra)
                if mode=='owned-after':argv += json.loads(args.candidate_extra)
                arm_env = dict(env)
                arm_env['RILEY_PARITY_RUN_ID'] = str(arm)
                if mode == 'native-after': arm_env.update(json.loads(args.native_after_env))
                if args.inspect:
                    assert args.profile,'call inspection excluded from performance numbers'
                    arm_env['LD_PRELOAD']=str(Path(__file__).parent/'call-inspection.so')
                    arm_env['RILEY_W0_INSPECTION']=str(arm)
                is_vllm = mode.startswith('vllm')
                if is_vllm:
                    arm_env['PATH'] = '/home/nlp/riley-261013/vllm0271-venv/bin:/usr/local/cuda-12.4/bin:' + env.get('PATH','/usr/bin:/bin')
                    argv = ['/home/nlp/riley-261013/vllm0271-venv/bin/vllm', 'serve', args.model,
                            '--tokenizer', args.model, '--served-model-name', 'm',
                            '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                            '--max-model-len', '1024', '--max-num-seqs', str(conc),
                            '--max-num-batched-tokens', str(args.vllm_token_budget), '--block-size', '16',
                            '--num-gpu-blocks-override', str(conc*64),
                            '--gpu-memory-utilization', '0.5', '--no-enable-prefix-caching', '--seed', '0']
                    # Wheel libraries resolve from the installed vLLM environment.
                    arm_env.pop('LD_LIBRARY_PATH', None)
                    arm_env['VLLM_CACHE_ROOT'] = str(root/'vllm-cache')
                    arm_env['VLLM_RPC_BASE_PATH']=str(Path(__file__).parent/('ipc6' if args.gpu.endswith('55ee1e6a7') else 'ipc7'))
                    arm_env['VLLM_BATCH_INVARIANT'] = '0'
                    if mode == 'vllm-event-spin':
                        arm_env['RILEY_DIAGNOSTIC_CUDA_EVENT_SPIN'] = '1'
                        arm_env['PYTHONPATH'] = str(Path('/data/riley-graph-batch-opt-20261016-rtx4090')/'event-spin-diagnostic')
                (arm / 'argv.json').write_text(json.dumps(argv, indent=2))
                session = f'rileyopt{os.getpid()}c{conc}r{repeat}{mode.replace("-", "")}'
                nsys = '/mnt/efs/nlp/psyche/riley-w0-a100-20261019/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys'
                command = (['/usr/bin/taskset', '-c', ('24-31' if args.gpu=='GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7' else '32-39')] + argv)
                if args.profile:
                    command = [nsys, 'launch', '--session-new=' + session, '--trace=cuda','--cuda-graph-trace=node'] + command
                log = (arm / 'server.log').open('x')
                arm_launch_perf=time.perf_counter();arm_launch_ns=time.time_ns()
                process = subprocess.Popen(command, env=arm_env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                samples = []
                monitor_errors = []
                verified_owned_pids = {}
                observing = threading.Event()
                def observe():
                    while not observing.wait(.5):
                        try:
                            state = subprocess.check_output(['nvidia-smi','-i',args.gpu,
                                '--query-gpu=memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip()
                            processes = subprocess.check_output(['nvidia-smi','-i',args.gpu,
                                '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
                            for row in processes.splitlines():
                                if not row.strip(): continue
                                pid = int(row.split(',')[0]); proc = Path('/proc') / str(pid)
                                if support_process(pid,float(row.split(',')[1])):continue
                                if not proc.exists(): continue
                                try:
                                    same_uid = proc.stat().st_uid == os.getuid()
                                    start_ticks = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
                                    own = same_uid and (verified_owned_pids.get(pid) == start_ticks or (b'RILEY_PARITY_RUN_ID=' + str(arm).encode()) in (proc/'environ').read_bytes().split(b'\0'))
                                    # Stock vLLM can sanitize custom environment
                                    # variables when spawning EngineCore. Prove
                                    # ancestry back to this still-live server.
                                    current = pid
                                    seen = set()
                                    while same_uid and not own and current > 1 and current not in seen and len(seen) < 64:
                                        seen.add(current)
                                        if current == process.pid and process.poll() is None:
                                            own = True
                                            break
                                        status = (Path('/proc')/str(current)/'status').read_text()
                                        current = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))

                                except FileNotFoundError: continue
                                if own: verified_owned_pids[pid] = start_ticks
                                if not own and not any(e.get('pid') == pid for e in monitor_errors):
                                    cmd=(proc/'cmdline').read_bytes().split(b'\0')
                                    command_identity=[cmd[0].decode(errors='replace'),cmd[1].decode(errors='replace')] if len(cmd)>1 and cmd[1].endswith(b'.py') else [str((proc/'exe').resolve())]
                                    monitor_errors.append(dict(reason='foreign_GPU_process_preserved',pid=pid,start_ticks=start_ticks,uid=proc.stat().st_uid,command_identity=command_identity))
                            used = sum(float(row.split(',')[1]) for row in processes.splitlines() if row.strip())
                            if used > 22*1024: monitor_errors.append(dict(reason='22GiB_process_budget_exceeded',used_MiB=used))
                            device_used=float(state.split(',')[0])
                            if device_used > 22*1024: monitor_errors.append(dict(reason='22GiB_device_budget_exceeded',used_MiB=device_used))
                            samples.append(dict(time_ns=time.time_ns(), gpu=state, processes=processes, verified_owned_start_ticks=dict(verified_owned_pids)))
                            if monitor_errors:
                                (arm/'monitor-errors.json').write_text(json.dumps(monitor_errors,indent=2))
                                # Only this arm's verified own processes/session may be stopped.
                                for owned_pid,start_tick in list(verified_owned_pids.items()):
                                    proc=Path('/proc')/str(owned_pid)
                                    try:
                                        current_tick=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
                                        if proc.stat().st_uid==os.getuid() and current_tick==start_tick:
                                            os.kill(owned_pid,signal.SIGTERM)
                                    except ProcessLookupError:pass
                                    except FileNotFoundError:pass
                                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                                break
                        except (OSError,subprocess.CalledProcessError) as error:
                            samples.append(dict(time_ns=time.time_ns(), error=str(error)))
                observer = threading.Thread(target=observe)
                observer.start()
                def get(path):
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    try:
                        c.request('GET', path)
                        r = c.getresponse()
                        return r.status, r.read().decode()
                    finally:
                        c.close()

                def request(pair):
                    index, item = pair
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
                    started = time.perf_counter()
                    arrivals = []
                    ids, text, prompt, usage, finish, done = [], '', None, None, None, False
                    raw = []
                    try:
                        body = dict(model='m', prompt=item['prompt'], max_tokens=item['max_tokens'],
                                    temperature=0, stream=True, return_token_ids=True,
                                    stream_options=dict(include_usage=True))
                        if is_vllm:
                            # vLLM otherwise inherits repetition_penalty1.05
                            # from this checkpoint's generation_config.json.
                            # Retain its original EOS while explicitly matching
                            # native/reference neutral sampling controls.
                            body.update(repetition_penalty=1.0, top_p=1.0, top_k=-1,
                                        frequency_penalty=0.0, presence_penalty=0.0)
                        c.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
                        r = c.getresponse()
                        assert r.status == 200, (r.status, r.read())
                        while line := r.readline():
                            raw.append(line.decode())
                            if not line.startswith(b'data: '):
                                continue
                            payload = line[6:].strip()
                            if payload == b'[DONE]':
                                done = True
                                continue
                            event = json.loads(payload)
                            assert 'error' not in event, event
                            usage = event.get('usage') or usage
                            for choice in event.get('choices', []):
                                new = choice.get('token_ids') or []
                                if new:
                                    arrivals.extend([time.perf_counter()] * len(new))
                                ids.extend(new)
                                text += choice.get('text') or ''
                                prompt = choice.get('prompt_token_ids') or prompt
                                finish = choice.get('finish_reason') or finish
                        end = time.perf_counter()
                        assert done and ids and finish and usage, ''.join(raw)
                        ref = expected[index % len(items)]
                        if not is_vllm:
                            assert prompt == ref['prompt_ids'], (item['id'], 'HF tokenizer mismatch')
                        assert usage['completion_tokens'] == len(ids)
                        key = item['id']
                        value = dict(token_ids=ids, text=text, prompt_ids=prompt, finish_reason=finish, usage=usage)
                        if not is_vllm and not is_native:
                            if args.bootstrap_eager and native_expected[index % len(items)]['value'] is None:
                                assert ids==ref['token_ids'] and text==ref['text'],(item['id'],'fresh same-A100 HF generation mismatch')
                                native_expected[index % len(items)]['value']=value
                            if value!=native_expected[index % len(items)]['value']:
                                (arm/f'failed-response-{time.time_ns()}.json').write_text(json.dumps(dict(id=item['id'],actual=value,expected=native_expected[index % len(items)]['value'],raw_sse=''.join(raw)),indent=2))
                                raise AssertionError((item['id'],'same-device eager mismatch'))
                        return dict(id=key, value=value, start=started, end=end,
                                    hf_token_equal=ids == ref['token_ids'], hf_text_equal=text == ref['text'],
                                    eager_equal=None if args.bootstrap_eager else value == native_expected[index % len(items)]['value'],
                                    arrivals_ms=[(a - started) * 1000 for a in arrivals],
                                    ttft_ms=(arrivals[0] - started) * 1000,
                                    tpot_ms=(arrivals[-1] - arrivals[0]) * 1000 / (len(ids)-1) if len(ids)>1 else None,
                                    e2e_ms=(end-started)*1000, raw_sse=''.join(raw))
                    finally:
                        c.close()
                try:
                    deadline = time.monotonic() + 900
                    while True:
                        if monitor_errors:
                            raise RuntimeError('Selected GPU admission changed: '+json.dumps(monitor_errors))
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('Startup failed: ' + (arm/'server.log').read_text()[-2000:])
                        try:
                            if get('/v1/models')[0] == 200:
                                break
                        except (OSError, http.client.HTTPException):
                            pass
                        time.sleep(.25)
                    (arm/'startup-readiness.json').write_text(json.dumps(dict(elapsed_ms=(time.perf_counter()-arm_launch_perf)*1000,started_ns=arm_launch_ns,ready_ns=time.time_ns(),process_pid=process.pid,profile_launcher=args.profile),indent=2))
                    (arm / 'loaded-libraries.txt').write_text(Path(f'/proc/{process.pid}/maps').read_text() if not args.profile else 'See server child mapping; nsys launcher PID is not the model process.\n')
                    gpu_processes = subprocess.check_output(['nvidia-smi','-i',args.gpu,
                        '--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
                    for pid in gpu_processes:
                        if support_process(int(pid)):continue
                        proc=Path('/proc')/str(int(pid))
                        if not proc.exists():continue
                        if proc.stat().st_uid!=os.getuid():
                            raise RuntimeError('Foreign GPU PID appeared after admission; preserved PID '+str(pid))
                        try:
                            identity=(proc/'environ').read_bytes().split(b'\0')
                            marker=b'RILEY_PARITY_RUN_ID='+str(arm).encode()
                            if marker not in identity and int(pid) not in verified_owned_pids:
                                raise RuntimeError('Unverified same-user GPU PID; preserved PID '+str(pid))
                            (arm/f'gpu-process-{pid}-libraries.txt').write_text((proc/'maps').read_text())
                        except FileNotFoundError:continue
                        except PermissionError as error:
                            (arm/f'gpu-process-{pid}-maps-unavailable.txt').write_text(repr(error))
                            raise RuntimeError('GPU process identity/maps could not be verified; stop only this arm')
                    for endpoint in ['/v1/config', '/metrics']:
                        (arm / (endpoint.rsplit('/',1)[-1]+'-before.txt')).write_text(str(get(endpoint)))
                    with concurrent.futures.ThreadPoolExecutor(conc) as pool:
                        correctness = list(pool.map(request, enumerate(items * (1 if conc == 1 else 3)))) if repeat == 0 else []
                        for row in correctness:
                            if is_vllm:
                                continue
                            if is_native:
                                first = native_canonical.setdefault(row['id'], dict(value=row['value'], arm=str(arm)))
                                if first['value'] != row['value']:
                                    native_invariance.append(dict(id=row['id'], arm=str(arm), first_arm=first['arm']))
                                (root/'native-invariance.json').write_text(json.dumps(dict(
                                    checked_ids=sorted(native_canonical), mismatches=native_invariance), indent=2))
                                continue
                            if row['id'] in canonical:
                                assert canonical[row['id']] == row['value'], ('cross-arm or concurrency mismatch', row['id'])
                            else:
                                canonical[row['id']] = row['value']
                        (arm / 'correctness.json').write_text(json.dumps(correctness))
                        if repeat == 0 and not is_vllm:
                            stable = json.loads(get('/metrics')[1])
                            for _ in range(2):
                                c = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
                                c.request('POST', '/v1/completions', json.dumps(dict(
                                    model='m', prompt=items[-1]['prompt'], max_tokens=128,
                                    temperature=0, stream=True, return_token_ids=True)),
                                    {'Content-Type':'application/json'})
                                response = c.getresponse()
                                assert response.status == 200
                                while line := response.readline():
                                    if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                                        event = json.loads(line[6:])
                                        assert 'error' not in event, event
                                        if any(choice.get('token_ids') for choice in event.get('choices', [])):
                                            break
                                else:
                                    raise AssertionError('no token before disconnect')
                                if c.sock:
                                    c.sock.shutdown(socket.SHUT_RDWR)
                                response.close()
                                c.close()
                            deadline = time.monotonic() + 10
                            while True:
                                recovered = json.loads(get('/metrics')[1])
                                if recovered['active_requests'] == recovered['waiting_requests'] == recovered['kv_allocated_blocks'] == 0:
                                    break
                                assert time.monotonic() < deadline, recovered
                                time.sleep(.05)
                            retries = list(pool.map(request, enumerate(items[:3])))
                            final = json.loads(get('/metrics')[1])
                            assert final['allocation'] == stable['allocation'], (stable, final)
                            assert final['kv_allocated_blocks'] == 0
                            (arm/'recovery.json').write_text(json.dumps(dict(disconnects=2,
                                retries=retries, before=stable, after=final)))
                        if args.warm_buckets:
                            # Preserve the unmodified full eager numerical oracle for
                            # every warm request, including text/usage/finish. Both
                            # engines receive exactly the same waves. No token cap
                            # edits or prefix-only numerical checks are introduced.
                            anchors = [(i, item) for i, item in enumerate(items)
                                       if len(native_expected[i]['value']['token_ids']) >= 32
                                       and len(native_expected[i]['value']['prompt_ids']) <= 256]
                            assert anchors, 'No full-reference warm anchor with >=32 outputs and <=256 prompt tokens'
                            anchor = next((v for v in anchors if v[1]['id'] == 'natural-128'), anchors[0])
                            waves = []
                            width = 1
                            while width <= conc:
                                with concurrent.futures.ThreadPoolExecutor(width) as warming:
                                    warm_rows = list(warming.map(request, [anchor] * width))
                                waves.append(dict(width=width, anchor=anchor[1], requests=warm_rows))
                                (arm / 'bucket-warm.json').write_text(json.dumps(waves))
                                width += 1
                        list(pool.map(request, (measured_items[i % len(measured_items)] for i in range(args.warm))))
                        for window in range(args.windows):
                            window_dir=arm/f'w{window}'
                            window_dir.mkdir()
                            if args.profile:
                                subprocess.run([nsys, 'start', '--session=' + session, '--sample=none',
                                                '--cpuctxsw=none', '-o', str(arm/'trace')], env=env, check=True)
                            captures_before=(arm/'server.log').read_text().count('RILEY_OWNED_SEGMENT capture=true')
                            rows = list(pool.map(request, (measured_items[i % len(measured_items)] for i in range(args.keep))))
                            captures_after=(arm/'server.log').read_text().count('RILEY_OWNED_SEGMENT capture=true')
                            (window_dir / 'requests.json').write_text(json.dumps(rows))
                            if args.profile:
                                subprocess.run([nsys, 'stop', '--session=' + session], env=env, check=True)
                                exported = subprocess.run([nsys, 'export', '--type=sqlite', '--output', str(arm/'trace.sqlite'), str(arm/'trace.nsys-rep')], env=env)
                                (arm / 'export-status.txt').write_text(str(exported.returncode))
                            (window_dir/'metrics-after.txt').write_text(str(get('/metrics')))
                            wall = max(r['end'] for r in rows) - min(r['start'] for r in rows)
                            rec = dict(concurrency=conc, repeat=repeat, window=window, mode=mode, requests=len(rows),
                                       correctness_requests=(len(correctness) if window==0 else 0), output_tok_s=sum(len(r['value']['token_ids']) for r in rows)/wall)
                            rec.update(graph_captures_before_measurement=captures_before, graph_captures_during_measurement=captures_after-captures_before)
                            rec.update(bucket_warm_requested=args.warm_buckets, measured_graph_capture_free=(captures_after==captures_before))
                            rec.update(hf_correctness_tokens=(sum(r['hf_token_equal'] for r in correctness) if window==0 else 0),
                                       hf_correctness_text=(sum(r['hf_text_equal'] for r in correctness) if window==0 else 0),
                                       output_tokens=sum(len(r['value']['token_ids']) for r in rows),
                                       measured_hf_token_matches=sum(r['hf_token_equal'] for r in rows))
                            rec.update(measured_eager_matches=sum(bool(r['eager_equal']) for r in rows),
                                       measured_native_canonical_matches=(sum(r['value'] == native_canonical[r['id']]['value'] for r in rows if r['id'] in native_canonical) if is_native else None))
                            itl = [b - a for r in rows for a, b in zip(r['arrivals_ms'], r['arrivals_ms'][1:])]
                            for metric in ['ttft_ms', 'tpot_ms', 'e2e_ms']:
                                vals = [r[metric] for r in rows if r[metric] is not None]
                                for name, q in [('p50', .5), ('p95', .95), ('p99', .99)]:
                                    rec[metric+'_'+name] = quantile(vals, q)
                            for name, q in [('p50', .5), ('p95', .95), ('p99', .99)]:
                                rec['itl_ms_'+name] = quantile(itl, q)
                            rec['itl_samples'] = len(itl)
                            assert not monitor_errors, monitor_errors
                            records.append(rec)
                            (root / 'result.json').write_text(json.dumps(dict(plan=plan, records=records), indent=2))
                            print(json.dumps(rec), flush=True)

                finally:
                    if args.profile:
                        subprocess.run([nsys, 'shutdown', '--session='+session, '--kill=sigterm'], env=env)
                    elif process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    observing.set()
                    observer.join()
                    (arm/'gpu-samples.json').write_text(json.dumps(samples))
                    log.close()
                    # The nsys launcher can exit before its owned server has
                    # finished releasing the CUDA context. Do not misclassify
                    # that bounded teardown as a foreign process or kill it.
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        live = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
                            '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                        if not any(not support_process(int(pid)) for pid in live.split()):
                            break
                        time.sleep(.25)
                print('RILEY_ARM_COMPLETE '+json.dumps(dict(concurrency=conc,repeat=repeat,mode=mode)),flush=True)
                if args.arm_handshake:
                    assert sys.stdin.readline().strip()=='continue','controller did not acknowledge next arm'
    if args.bootstrap_eager:
        assert all(x['value'] is not None for x in native_expected)
        (root/'native-reference.json').write_text(json.dumps(native_reference,indent=2))
    (root/'complete.json').write_text(json.dumps(dict(arms=len(records),
        all_native_eager_reference_assertions_passed=True, vllm_hf_agreement_reported_separately=True)))


if __name__ == '__main__':
    main()
