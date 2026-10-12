"""Simple descriptive Qwen2.5-3B C1 comparison: P2048/O128 greedy, sequential requests. Not a qualifying benchmark."""
import hashlib, http.client, json, os, signal, socket, statistics as st, subprocess, sys, threading, time
from pathlib import Path
R = Path('/data/riley-serving-261007'); O = R / f'qwen-simple-perf-attempt{int(sys.argv[1]):02d}'; O.mkdir()
base = Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host'); teacher = base / 'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
model = '/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1'; riley = str(R / 'qwen-http-exact-target-attempt01/release/riley')
w = json.loads((base / 'inputs/qwen3b-c8-p2048-o128.json').read_text()); hf = json.loads((teacher / 'teacher-forced-oracle.json').read_text())['generation']['teacher_token_ids']
WARM, KEEP = 2, 10
plan = {'model': 'Qwen/Qwen2.5-3B-Instruct@aa8e7253 BF16', 'workload': 'qwen3b-c8-p2048-o128 prompt (2048 tokens), max_tokens 128, temperature 0, one request at a time', 'warmup': WARM, 'retained': KEEP,
        'orders': [['riley-hf-exact', 'riley-scheduler', 'vllm'], ['vllm', 'riley-scheduler', 'riley-hf-exact']], 'exclusions': 'none', 'scope': 'descriptive only; Riley Qwen routes are not performance-optimized', 'predeclared_time_ns': time.time_ns()}
(O / 'plan.json').write_text(json.dumps(plan, indent=1))
smi = lambda q, k='gpu': subprocess.check_output(['nvidia-smi', f'--query-{k}={q}', '--format=csv,noheader,nounits'], text=True).strip()
common = ['--model', model, '--model-id', 'qwen3b', '--max-active-sequences', '1', '--max-waiting-requests', '2', '--max-sequence-tokens', '2176', '--max-output-tokens', '128', '--max-weight-bytes', str(8 << 30)]
def launch(engine, port):
    env = {'HOME': '/home/psyche', 'PATH': str(R / 'vllm0271-venv/bin') + ':/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '0', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}
    if engine == 'vllm':
        tool = '/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13'; env.update(CUDA_HOME=tool, LD_LIBRARY_PATH=tool + '/lib', CPATH=str(R / 'curand-only-include'), VLLM_BATCH_INVARIANT='0')
        return [str(R / 'vllm0271-venv/bin/vllm'), 'serve', model, '--tokenizer', model, '--served-model-name', 'qwen3b', '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16', '--max-model-len', '2176', '--max-num-seqs', '1', '--gpu-memory-utilization', '0.6', '--no-enable-prefix-caching', '--seed', '0'], env
    env.update(LD_LIBRARY_PATH='/data/cuda-12.8.1/lib64', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if engine == 'riley-hf-exact':
        return [riley, 'serve', '--bind', f'127.0.0.1:{port}', *common, '--numerics-profile', 'hf-eager-qwen-p2048-o128-probe-v1'], env
    return [riley, 'serve', '--bind', f'127.0.0.1:{port}', *common, '--batch-token-budget', '32', '--prefill-chunk-tokens', '32', '--kv-blocks', '136', '--decode-attention-backend', 'native-bf16-paged-split-gqa-d128-two-stage',
            '--execution-graph-policy', 'disabled', '--reduction-profile', 'canonical-v1', '--projection-bias-backend', 'strict-staged-v1', '--batch-shape-policy', 'fixed-max', '--execution-completion', 'per-operation',
            '--metadata-transport', 'synchronous', '--residual-rmsnorm', 'separate'], env
def request(port):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=600); t0 = time.perf_counter(); arrivals = []; tokens = []; text = ''; usage = None; buf = b''
    try:
        c.request('POST', '/v1/completions', body=json.dumps({'model': 'qwen3b', 'prompt': w['prompt'], 'max_tokens': 128, 'temperature': 0, 'stream': True, 'return_token_ids': True, 'stream_options': {'include_usage': True}}), headers={'Content-Type': 'application/json'})
        r = c.getresponse(); assert r.status == 200, (r.status, r.read()[:300])
        while True:
            chunk = r.read1(65536)
            if not chunk: break
            buf += chunk
            while b'\n\n' in buf:
                frame, buf = buf.split(b'\n\n', 1)
                for line in frame.split(b'\n'):
                    if line.startswith(b'data: ') and line != b'data: [DONE]':
                        v = json.loads(line[6:]); usage = v.get('usage') or usage
                        for ch in v.get('choices', []):
                            ids = ch.get('token_ids') or []
                            if ids or ch.get('text'): arrivals.append((time.perf_counter(), len(ids)))
                            tokens += ids; text += ch.get('text') or ''
        t1 = time.perf_counter(); n = (usage or {}).get('completion_tokens') or len(tokens)
        return {'prompt_tokens': (usage or {}).get('prompt_tokens'), 'output_tokens': n, 'ttft_ms': (arrivals[0][0] - t0) * 1e3, 'tpot_ms': (arrivals[-1][0] - arrivals[0][0]) * 1e3 / max(1, n - 1), 'e2e_ms': (t1 - t0) * 1e3,
                'tokens_match_hf': tokens == hf if tokens else None, 'token_ids_returned': len(tokens), 'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
    finally: c.close()
records = []; failure = None
try:
    for repeat, order in enumerate(plan['orders']):
        for engine in order:
            assert not smi('pid', 'compute-apps'), 'GPU occupied'
            with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
            argv, env = launch(engine, port); name = f'r{repeat}-{engine}'; gpu = []; done = threading.Event()
            p = subprocess.Popen(argv, env=env, stdout=(O / (name + '.log')).open('x'), stderr=subprocess.STDOUT, start_new_session=True)
            def observe():
                while not done.wait(1): gpu.append(float(smi('memory.used')))
            th = threading.Thread(target=observe); th.start()
            try:
                deadline = time.monotonic() + 600
                while True:
                    assert p.poll() is None and time.monotonic() < deadline, 'startup failed ' + name
                    k = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    try:
                        k.request('GET', '/v1/models'); rr = k.getresponse(); rr.read()
                        if rr.status == 200: break
                    except (OSError, http.client.HTTPException): pass
                    finally: k.close()
                    time.sleep(.5)
                warm = [request(port) for _ in range(WARM)]; t0 = time.perf_counter(); rows = [request(port) for _ in range(KEEP)]; wall = time.perf_counter() - t0
                rec = {'case': name, 'engine': engine, 'repeat': repeat, 'argv': argv, 'rows': rows, 'warmup': warm, 'output_tok_per_s': sum(r['output_tokens'] for r in rows) / wall,
                       'ttft_ms_p50': st.median(r['ttft_ms'] for r in rows), 'tpot_ms_p50': st.median(r['tpot_ms'] for r in rows), 'e2e_ms_p50': st.median(r['e2e_ms'] for r in rows), 'e2e_ms_max': max(r['e2e_ms'] for r in rows),
                       'output_tokens': sorted({r['output_tokens'] for r in rows}), 'prompt_tokens': sorted({str(r['prompt_tokens']) for r in rows}), 'tokens_match_hf': sorted({str(r['tokens_match_hf']) for r in rows}), 'errors': 0}
            finally:
                if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=120)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                done.set(); th.join()
            rec['gpu_peak_MiB'] = max(gpu) if gpu else None; rec['exit_code'] = p.returncode; records.append(rec)
            print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k not in ('rows', 'warmup', 'argv')}, flush=True)
except BaseException as e:
    failure = {'type': type(e).__name__, 'message': str(e)}; raise
finally:
    (O / 'result.json').write_text(json.dumps({'plan': plan, 'failure': failure, 'records': records}, indent=1))
