"""Sustained load, cancellation, re-request and reclamation for the real `riley serve` HF-exact Qwen profile.
Every completed request must return the 128 HF teacher tokens. Timing is descriptive only."""
import hashlib, http.client, json, os, signal, socket, statistics as st, subprocess, sys, threading, time
from pathlib import Path
R = Path('/data/riley-serving-261007'); O = R / f'qwen-http-exact-stability-attempt{int(sys.argv[1]):02d}'; O.mkdir()
SUSTAIN = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
base = Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host'); teacher = base / 'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
model = '/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1'; binary = R / 'qwen-http-exact-target-attempt01/release/riley'
w = json.loads((base / 'inputs/qwen3b-c8-p2048-o128.json').read_text()); hf = json.loads((teacher / 'teacher-forced-oracle.json').read_text())['generation']['teacher_token_ids']
with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
argv = [str(binary), 'serve', '--model', model, '--model-id', 'qwen3b', '--bind', f'127.0.0.1:{port}', '--max-active-sequences', '1', '--max-waiting-requests', '4',
        '--max-sequence-tokens', '2176', '--max-output-tokens', '128', '--max-weight-bytes', str(8 << 30), '--numerics-profile', 'hf-eager-qwen-p2048-o128-probe-v1']
env = {'HOME': '/home/psyche', 'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '0', 'LD_LIBRARY_PATH': '/data/cuda-12.8.1/lib64', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'RILEY_SHUTDOWN_METRICS_PATH': str(O / 'native-close.json')}
smi = lambda q, k='gpu': subprocess.check_output(['nvidia-smi', f'--query-{k}={q}', '--format=csv,noheader,nounits'], text=True).strip()
assert not smi('pid', 'compute-apps')
body = lambda n=128: json.dumps({'model': 'qwen3b', 'prompt': w['prompt'], 'max_tokens': n, 'temperature': 0, 'top_p': 1.0, 'stream': True, 'return_token_ids': True})
def metrics():
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
    try: c.request('GET', '/metrics'); return json.loads(c.getresponse().read())
    finally: c.close()
def drained(label):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        m = metrics()
        if m['active_requests'] == m['waiting_requests'] == m['kv_allocated_blocks'] == 0: return m
        time.sleep(.02)
    raise RuntimeError('did not drain ' + label)
def request(n=128, cancel_after=None):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=300); t0 = time.perf_counter(); arrivals = []; tokens = []; finish = None
    try:
        c.request('POST', '/v1/completions', body=body(n), headers={'Content-Type': 'application/json'})
        if cancel_after == 0: return {'cancelled': 'early'}
        r = c.getresponse(); assert r.status == 200, r.status; buf = b''
        while True:
            chunk = r.read1(65536)
            if not chunk: break
            buf += chunk
            while b'\n\n' in buf:
                frame, buf = buf.split(b'\n\n', 1)
                for line in frame.split(b'\n'):
                    if line.startswith(b'data: ') and line != b'data: [DONE]':
                        v = json.loads(line[6:])
                        for ch in v['choices']:
                            ids = ch.get('token_ids') or []
                            tokens += ids; arrivals += [time.perf_counter()] * len(ids); finish = ch.get('finish_reason') or finish
            if cancel_after and len(tokens) >= cancel_after: return {'cancelled': 'decode', 'tokens_seen': tokens, 'prefix_exact': tokens == hf[:len(tokens)]}
        return {'tokens_exact': tokens == hf[:n] and len(tokens) == n, 'finish': finish, 'ttft_ms': (arrivals[0] - t0) * 1e3, 'tpot_ms': (arrivals[-1] - arrivals[0]) * 1e3 / (len(tokens) - 1), 'e2e_ms': (time.perf_counter() - t0) * 1e3, 'tokens': len(tokens)}
    finally: c.close()
log = (O / 'server.log').open('x'); p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True); samples = []; done = threading.Event(); failure = None; out = {'argv': argv, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
def observe():
    while not done.wait(1):
        rss = sum(int(x.split()[2]) for x in subprocess.check_output(['ps', '-e', '-o', 'pid=,pgid=,rss='], text=True).splitlines() if int(x.split()[1]) == p.pid)
        samples.append({'t': time.time(), 'gpu_MiB': float(smi('memory.used')), 'rss_MiB': rss / 1024})
th = threading.Thread(target=observe); th.start()
try:
    deadline = time.monotonic() + 300
    while True:
        assert p.poll() is None and time.monotonic() < deadline, 'startup failed'
        try: metrics(); break
        except (OSError, http.client.HTTPException, ValueError): time.sleep(.2)
    rows = [request()]; assert rows[0]['tokens_exact']; drained('warmup')
    mark = len(samples); start = time.monotonic(); rows = []
    # sustained: two clients keep one request active and one waiting
    lock = threading.Lock(); errors = []
    def client():
        try:
            while time.monotonic() - start < SUSTAIN:
                r = request()
                with lock: rows.append(r)
                assert r['tokens_exact'] and r['finish'] == 'length', r
        except BaseException as e: errors.append(repr(e))
    cs = [threading.Thread(target=client) for _ in range(2)]; [c.start() for c in cs]; [c.join() for c in cs]
    assert not errors, errors
    end = len(samples); elapsed = time.monotonic() - start; drained('sustained')
    out['sustained'] = {'seconds': elapsed, 'requests': len(rows), 'errors': 0, 'exact': sum(r['tokens_exact'] for r in rows), 'output_tok_per_s': sum(r['tokens'] for r in rows) / elapsed,
        'tpot_ms_p50': st.median(r['tpot_ms'] for r in rows), 'tpot_ms_max': max(r['tpot_ms'] for r in rows), 'ttft_ms_p50_including_queue_wait': st.median(r['ttft_ms'] for r in rows), 'e2e_ms_p50': st.median(r['e2e_ms'] for r in rows),
        'tpot_first_last': [rows[0]['tpot_ms'], rows[-1]['tpot_ms']]}
    win = samples[mark:end]; q = max(1, len(win) // 4)
    out['memory'] = {'gpu_MiB_first_last_quarter': [st.mean(x['gpu_MiB'] for x in win[:q]), st.mean(x['gpu_MiB'] for x in win[-q:])], 'rss_MiB_first_last_quarter': [st.mean(x['rss_MiB'] for x in win[:q]), st.mean(x['rss_MiB'] for x in win[-q:])], 'rss_MiB_max': max(x['rss_MiB'] for x in win), 'samples': len(win)}
    cancels = []
    for trial in range(20):
        mode = [0, 1, 5, 40][trial % 4]; r = request(cancel_after=mode); assert r.get('prefix_exact', True), r
        drained(f'cancel {trial}'); again = request(8 if trial % 2 else 128); assert again['tokens_exact'], again; drained(f'reuse {trial}')
        cancels.append({'cancel_after_tokens': mode, 'rerequest_tokens': again['tokens'], 'rerequest_exact': True})
    # cancellation while another request is active and one is queued
    bg = []; t = threading.Thread(target=lambda: bg.append(request())); t.start(); time.sleep(.3)
    queued_cancel = request(cancel_after=0); t.join(); assert bg[0]['tokens_exact']; final = drained('queued cancel')
    out['cancellation'] = {'idle_trials': cancels, 'queued_cancel_during_active_request': {'background_exact': True}, 'server_counters': final['counters']}
    out['single_request_timing_ms'] = {k: request()[k] for k in ['ttft_ms', 'tpot_ms', 'e2e_ms']}; drained('final')
except BaseException as e:
    failure = {'type': type(e).__name__, 'message': str(e)}; raise
finally:
    if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
    p.wait(timeout=120); done.set(); th.join(); time.sleep(2)
    close = json.loads((O / 'native-close.json').read_text()) if (O / 'native-close.json').exists() else None
    out.update(failure=failure, exit_code=p.returncode, gpu_compute_after=smi('pid', 'compute-apps'), native_close=close and {k: close[k] for k in ['active_requests', 'waiting_requests', 'kv_allocated_blocks', 'allocation']})
    out['reclaimed'] = bool(p.returncode == 0 and not out['gpu_compute_after'] and close and close['kv_allocated_blocks'] == 0 and all(v == 0 for v in close['allocation'].values()))
    out['pass'] = failure is None and out['reclaimed']
    (O / 'result.json').write_text(json.dumps(out, indent=1)); (O / 'host-samples.json').write_text(json.dumps(samples))
    print(json.dumps({k: out.get(k) for k in ['pass', 'failure', 'sustained', 'memory', 'reclaimed', 'single_request_timing_ms']}, indent=1))
