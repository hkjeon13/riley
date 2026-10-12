"""Stability of the adopted Riley Qwen scheduler-route build: sustained load, cancellation under load, re-request, drain, reclamation.
No HF reference exists for these prompts, so correctness is that every completion of a prompt equals the first one (token ids and text)."""
import concurrent.futures, hashlib, http.client, json, os, signal, socket, statistics as st, subprocess, sys, threading, time
from pathlib import Path
R = Path('/data/riley-serving-261007'); CONC = int(sys.argv[2]); WL = sys.argv[3]; SUSTAIN = float(sys.argv[4])
O = R / f'qwen-scheduler-stability-c{CONC}-{WL}-attempt{int(sys.argv[1]):02d}'; O.mkdir()
corpus = json.loads((R / 'kernel-batch13-controller-attempt01/fixtures.json').read_text())[WL]['corpus']
model = '/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1'; riley = str(R / 'qwen-http-exact-target-attempt01/release/riley')
nat = WL == 'natural'; seq = 1024 if nat else 256; out_tokens = 128 if nat else 32
with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
argv = [riley, 'serve', '--bind', f'127.0.0.1:{port}', '--model', model, '--model-id', 'qwen3b', '--max-active-sequences', str(CONC), '--max-waiting-requests', '64', '--max-sequence-tokens', str(seq), '--max-output-tokens', str(out_tokens),
        '--max-weight-bytes', str(8 << 30), '--batch-token-budget', '256', '--prefill-chunk-tokens', '256', '--gemm-reduction-policy', 'first-heuristic-split-k-v1', '--kv-blocks', str(CONC * seq // 16),
        '--decode-attention-backend', 'native-bf16-paged-split-gqa-d128-two-stage', '--execution-graph-policy', 'disabled', '--reduction-profile', 'canonical-v1', '--projection-bias-backend', 'merged-qkv-strict-staged-v1',
        '--batch-shape-policy', 'power-of-two', '--execution-completion', 'iteration-batch', '--metadata-transport', 'synchronous', '--residual-rmsnorm', 'fused', '--sampling-backend', 'gpu-greedy', '--kv-cache-write', 'fused-split-v1', '--prompt-attention', 'tensor-core-v1', '--batch-shape-buckets', '1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256', '--decode-attention-merge', 'global-max-v1', '--decode-step-graph', 'capture-v1', '--mlp-activation', 'fused-silu-multiply-v1', '--query-key-rope', 'fused-split-v1', '--gemm-algorithm-selection', 'bit-equivalent-table-v1', '--mlp-residual-norm', 'fused-next-norm-v1', '--lm-head-rows', 'output-rows-v1']
env = {'HOME': '/home/psyche', 'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '0', 'LD_LIBRARY_PATH': '/data/cuda-12.8.1/lib64', 'RILEY_SHUTDOWN_METRICS_PATH': str(O / 'native-close.json')}
smi = lambda q, k='gpu': subprocess.check_output(['nvidia-smi', f'--query-{k}={q}', '--format=csv,noheader,nounits'], text=True).strip()
assert not smi('pid', 'compute-apps')
def metrics():
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
    try: c.request('GET', '/metrics'); return json.loads(c.getresponse().read())
    finally: c.close()
def drained(label):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        m = metrics()
        if m['active_requests'] == m['waiting_requests'] == m['kv_allocated_blocks'] == 0: return m
        time.sleep(.05)
    raise RuntimeError('did not drain ' + label)
first = {}; lock = threading.Lock()
def request(i, cancel_after=None):
    item = corpus[i % len(corpus)]; c = http.client.HTTPConnection('127.0.0.1', port, timeout=600); t0 = time.perf_counter(); arr = []; toks = []; text = ''; buf = b''
    try:
        c.request('POST', '/v1/completions', body=json.dumps({'model': 'qwen3b', 'prompt': item['prompt'], 'max_tokens': item['max_tokens'], 'temperature': 0, 'stream': True, 'return_token_ids': True}), headers={'Content-Type': 'application/json'})
        if cancel_after == 0: return {'cancelled': 'early'}
        r = c.getresponse(); assert r.status == 200, (r.status, r.read()[:200])
        while True:
            ch = r.read1(65536)
            if not ch: break
            buf += ch
            while b'\n\n' in buf:
                f, buf = buf.split(b'\n\n', 1)
                for line in f.split(b'\n'):
                    if line.startswith(b'data: ') and line != b'data: [DONE]':
                        for x in json.loads(line[6:]).get('choices', []):
                            ids = x.get('token_ids') or []; toks += ids; text += x.get('text') or ''
                            if ids: arr.append(time.perf_counter())
            if cancel_after and len(toks) >= cancel_after:
                with lock: ref = first.get(item['id'])
                return {'cancelled': 'decode', 'prefix_consistent': ref is None or toks == ref[0][:len(toks)]}
        with lock: ref = first.setdefault(item['id'], (toks, text))
        return {'consistent': ref == (toks, text), 'tokens': len(toks), 'start': t0, 'end': time.perf_counter(), 'ttft_ms': (arr[0] - t0) * 1e3, 'tpot_ms': (arr[-1] - arr[0]) * 1e3 / max(1, len(toks) - 1)}
    finally: c.close()
p = subprocess.Popen(argv, env=env, stdout=(O / 'server.log').open('x'), stderr=subprocess.STDOUT, start_new_session=True); samples = []; done = threading.Event(); failure = None; out = {'argv': argv, 'binary_sha256': hashlib.sha256(Path(riley).read_bytes()).hexdigest(), 'concurrency': CONC, 'workload': WL}
def observe():
    while not done.wait(1):
        rss = sum(int(x.split()[2]) for x in subprocess.check_output(['ps', '-e', '-o', 'pid=,pgid=,rss='], text=True).splitlines() if int(x.split()[1]) == p.pid)
        samples.append({'t': time.time(), 'gpu_temp_C': float(smi('temperature.gpu')), 'gpu_clock_MHz': float(smi('clocks.gr').split()[0]), 'gpu_MiB': float(smi('memory.used')), 'rss_MiB': rss / 1024})
th = threading.Thread(target=observe); th.start()
try:
    deadline = time.monotonic() + 600
    while True:
        assert p.poll() is None and time.monotonic() < deadline, 'startup failed'
        try: metrics(); break
        except (OSError, http.client.HTTPException, ValueError): time.sleep(.5)
    with concurrent.futures.ThreadPoolExecutor(CONC) as pool:
        warm = list(pool.map(request, range(len(corpus) * 2))); assert all(r['consistent'] for r in warm); drained('warmup')
        mark = len(samples); start = time.monotonic(); rounds = []
        while time.monotonic() - start < SUSTAIN:
            rows = list(pool.map(request, range(96))); assert all(r['consistent'] for r in rows), 'output changed between repeats'
            wall = max(r['end'] for r in rows) - min(r['start'] for r in rows)
            rounds.append({'requests': len(rows), 'tok_per_s': sum(r['tokens'] for r in rows) / wall, 'tpot_p50': st.median(r['tpot_ms'] for r in rows), 'ttft_p50': st.median(r['ttft_ms'] for r in rows), 'kv_after': metrics()['kv_allocated_blocks']})
        end = len(samples); out['sustained'] = {'round_tok_per_s': [round(r['tok_per_s'], 1) for r in rounds], 'round_tpot_p50': [round(r['tpot_p50'], 3) for r in rounds], 'seconds': time.monotonic() - start, 'rounds': len(rounds), 'requests': sum(r['requests'] for r in rounds), 'errors': 0, 'inconsistent': 0,
            'tok_per_s_first_last_min': [rounds[0]['tok_per_s'], rounds[-1]['tok_per_s'], min(r['tok_per_s'] for r in rounds)], 'tpot_p50_first_last': [rounds[0]['tpot_p50'], rounds[-1]['tpot_p50']], 'kv_after_rounds': sorted({r['kv_after'] for r in rounds})}
        drained('sustained'); win = samples[mark:end]; q = max(1, len(win) // 4)
        out['memory'] = {'gpu_MiB_first_last_quarter': [st.mean(x['gpu_MiB'] for x in win[:q]), st.mean(x['gpu_MiB'] for x in win[-q:])], 'rss_MiB_first_last_quarter': [st.mean(x['rss_MiB'] for x in win[:q]), st.mean(x['rss_MiB'] for x in win[-q:])], 'rss_MiB_max': max(x['rss_MiB'] for x in win)}
        idle = []
        for trial in range(20):
            r = request(trial, cancel_after=[0, 1, 3, 8][trial % 4]); assert r.get('prefix_consistent', True), r; drained(f'cancel {trial}')
            again = list(pool.map(request, range(len(corpus)))); assert all(x['consistent'] for x in again); drained(f'reuse {trial}'); idle.append(r.get('cancelled'))
        stop = threading.Event(); loaded = []; errors = []
        def canceller():
            try:
                for trial in range(40):
                    if stop.is_set(): break
                    r = request(trial, cancel_after=[0, 2][trial % 2]); assert r.get('prefix_consistent', True), r; loaded.append(r.get('cancelled')); time.sleep(.25)
            except BaseException as e: errors.append(repr(e))
        t = threading.Thread(target=canceller); t.start(); load_requests = 0
        while t.is_alive():
            rows = list(pool.map(request, range(48))); assert all(r['consistent'] for r in rows), 'output changed under cancellation load'; load_requests += len(rows)
        t.join(); assert not errors, errors; final = drained('loaded cancel')
        rows = list(pool.map(request, range(48))); assert all(r['consistent'] for r in rows); final = drained('final')
        out['cancellation'] = {'idle': len(idle), 'under_load': len(loaded), 'requests_during_loaded_cancellation': load_requests, 'server_counters': final['counters']}
except BaseException as e:
    failure = {'type': type(e).__name__, 'message': str(e)[:500]}; raise
finally:
    if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
    try: p.wait(timeout=180)
    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
    done.set(); th.join(); time.sleep(2)
    close = json.loads((O / 'native-close.json').read_text()) if (O / 'native-close.json').exists() else None
    out.update(failure=failure, exit_code=p.returncode, gpu_compute_after=smi('pid', 'compute-apps'), native_close=close and {k: close[k] for k in ['active_requests', 'waiting_requests', 'kv_allocated_blocks', 'allocation']})
    out['reclaimed'] = bool(p.returncode == 0 and not out['gpu_compute_after'] and close and close['kv_allocated_blocks'] == 0 and all(v == 0 for v in close['allocation'].values()))
    out['pass'] = failure is None and out['reclaimed']
    (O / 'result.json').write_text(json.dumps(out, indent=1)); (O / 'host-samples.json').write_text(json.dumps(samples))
    print(json.dumps({k: out.get(k) for k in ['pass', 'failure', 'sustained', 'memory', 'cancellation', 'reclaimed', 'exit_code']}))
