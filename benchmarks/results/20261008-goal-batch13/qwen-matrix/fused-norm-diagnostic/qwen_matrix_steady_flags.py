"""Descriptive Qwen2.5-3B 8-cell baseline: Riley legacy scheduler route vs vLLM 0.27.1 on the SmolLM2 fixture prompts.
No HF reference exists for these prompts under Qwen; correctness here is only per-engine determinism across repeats."""
import concurrent.futures, hashlib, http.client, json, os, signal, socket, statistics as st, subprocess, sys, threading, time
from pathlib import Path
R = Path('/data/riley-serving-261007'); O = R / f'qwen-matrix-steady-flags-{sys.argv[2]}-diagnostic-attempt{int(sys.argv[1]):02d}'; O.mkdir()
fixtures = json.loads((R / 'kernel-batch13-controller-attempt01/fixtures.json').read_text())
model = '/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1'; riley = str(R / 'qwen-algo-diagtarget/release/riley')
WARM, KEEP = 32, 192
plan = {'model': 'Qwen/Qwen2.5-3B-Instruct@aa8e7253 BF16', 'workloads': 'SmolLM2 benchmark fixture prompts and max_tokens (fixed: 12 prompts, natural: 3 prompts), temperature 0, original EOS', 'concurrency': [1, 8, 16, 32], 'warmup': WARM, 'retained': KEEP,
        'engines': {'riley-scheduler': 'branch build: batch-executor route, iteration-batch completion, 256-token budget, power-of-two shapes, first-heuristic GEMM policy, worker pool sized to active sequences, one-pass greedy selection, GPU greedy, shared-KV D128 producer', 'vllm': '0.27.1, prefix cache off, gpu-memory-utilization 0.6'},
        'orders': [['riley-scheduler'], ['riley-scheduler']], 'exclusions': 'none', 'scope': 'steady-state comparison (warmup 32, retained 192 per lane so C32 holds full concurrency for 6 waves); no host gate; Riley = branch build with shared-KV D128 producer + bounded GPU greedy; pass rule per cell: Riley tok/s >= vLLM, TTFT/TPOT p50 <= vLLM, p95/p99 not worse; C1 treated as tie range per user decision 2026-10-09', 'predeclared_time_ns': time.time_ns()}
(O / 'plan.json').write_text(json.dumps(plan, indent=1))
smi = lambda q, k='gpu': subprocess.check_output(['nvidia-smi', f'--query-{k}={q}', '--format=csv,noheader,nounits'], text=True).strip()
def launch(engine, port, c, natural):
    seq = 1024 if natural else 256; out = 128 if natural else 32
    env = {'HOME': '/home/psyche', 'PATH': str(R / 'vllm0271-venv/bin') + ':/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '0', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}
    if engine == 'vllm':
        tool = '/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13'; env.update(CUDA_HOME=tool, LD_LIBRARY_PATH=tool + '/lib', CPATH=str(R / 'curand-only-include'), VLLM_BATCH_INVARIANT='0')
        return [str(R / 'vllm0271-venv/bin/vllm'), 'serve', model, '--tokenizer', model, '--served-model-name', 'qwen3b', '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16', '--max-model-len', str(seq), '--max-num-seqs', str(c), '--gpu-memory-utilization', '0.6', '--no-enable-prefix-caching', '--seed', '0'], env
    env.update(LD_LIBRARY_PATH='/data/cuda-12.8.1/lib64')
    argv = [riley, 'serve', '--bind', f'127.0.0.1:{port}', '--model', model, '--model-id', 'qwen3b', '--max-active-sequences', str(c), '--max-waiting-requests', '64', '--max-sequence-tokens', str(seq), '--max-output-tokens', str(out),
            '--max-weight-bytes', str(8 << 30), '--batch-token-budget', '256', '--prefill-chunk-tokens', '256', '--gemm-reduction-policy', 'first-heuristic-split-k-v1', '--kv-blocks', str(c * seq // 16), '--decode-attention-backend', 'native-bf16-paged-split-gqa-d128-two-stage',
            '--execution-graph-policy', 'disabled', '--reduction-profile', 'canonical-v1', '--projection-bias-backend', 'strict-staged-v1', '--batch-shape-policy', 'power-of-two', '--execution-completion', 'iteration-batch',
            '--metadata-transport', 'synchronous', '--residual-rmsnorm', 'separate', '--sampling-backend', 'gpu-greedy']
    for a in sys.argv[3:]:
        k, v = a.split('=', 1); argv[argv.index(k) + 1] = v
    return argv, env
def request(port, item):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=600); t0 = time.perf_counter(); arr = []; toks = []; text = ''; usage = None; buf = b''
    try:
        c.request('POST', '/v1/completions', body=json.dumps({'model': 'qwen3b', 'prompt': item['prompt'], 'max_tokens': item['max_tokens'], 'temperature': 0, 'stream': True, 'return_token_ids': True, 'stream_options': {'include_usage': True}}), headers={'Content-Type': 'application/json'})
        r = c.getresponse()
        if r.status != 200: return {'error': r.status, 'detail': r.read()[:200].decode('utf8', 'replace')}
        while True:
            ch = r.read1(65536)
            if not ch: break
            buf += ch
            while b'\n\n' in buf:
                f, buf = buf.split(b'\n\n', 1)
                for line in f.split(b'\n'):
                    if line.startswith(b'data: ') and line != b'data: [DONE]':
                        v = json.loads(line[6:]); usage = v.get('usage') or usage
                        for x in v.get('choices', []):
                            ids = x.get('token_ids') or []; toks += ids; text += x.get('text') or ''
                            if ids or x.get('text'): arr.append(time.perf_counter())
        t1 = time.perf_counter(); n = (usage or {}).get('completion_tokens') or len(toks)
        return {'id': item['id'], 'start': t0, 'end': t1, 'prompt_tokens': (usage or {}).get('prompt_tokens'), 'output_tokens': n, 'ttft_ms': (arr[0] - t0) * 1e3 if arr else None, 'tpot_ms': (arr[-1] - arr[0]) * 1e3 / (n - 1) if n > 1 and arr else None, 'e2e_ms': (t1 - t0) * 1e3,
                'out_sha': hashlib.sha256((json.dumps(toks) + text).encode()).hexdigest()}
    except Exception as e: return {'error': repr(e)}
    finally: c.close()
def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; lo = int(k); hi = min(lo + 1, len(v) - 1); return v[lo] + (v[hi] - v[lo]) * (k - lo)
records = []; failure = None
try:
    for conc in plan['concurrency']:
        for workload in ['fixed', 'natural']:
            corpus = fixtures[workload]['corpus']
            for repeat, order in enumerate(plan['orders']):
                for engine in order:
                    assert not smi('pid', 'compute-apps'), 'GPU occupied'
                    with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
                    argv, env = launch(engine, port, conc, workload == 'natural'); name = f'c{conc}-{workload}-r{repeat}-{engine}'; gpu = []; done = threading.Event()
                    p = subprocess.Popen(argv, env=env, stdout=(O / (name + '.log')).open('x'), stderr=subprocess.STDOUT, start_new_session=True)
                    def observe():
                        while not done.wait(1): gpu.append(float(smi('memory.used')))
                    th = threading.Thread(target=observe); th.start(); rec = {'case': name, 'cell': f'c{conc}-{workload}', 'engine': engine, 'repeat': repeat, 'argv': argv}
                    try:
                        deadline = time.monotonic() + 1800
                        while True:
                            assert p.poll() is None and time.monotonic() < deadline, 'startup failed ' + name
                            k = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                            try:
                                k.request('GET', '/v1/models'); rr = k.getresponse(); rr.read()
                                if rr.status == 200: break
                            except (OSError, http.client.HTTPException): pass
                            finally: k.close()
                            time.sleep(.5)
                        with concurrent.futures.ThreadPoolExecutor(conc) as pool:
                            list(pool.map(lambda i: request(port, corpus[i % len(corpus)]), range(WARM)))
                            rows = list(pool.map(lambda i: request(port, corpus[i % len(corpus)]), range(KEEP)))
                        ok = [r for r in rows if 'error' not in r]; wall = max(r['end'] for r in ok) - min(r['start'] for r in ok) if ok else None
                        tp = [r['tpot_ms'] for r in ok if r['tpot_ms'] is not None]
                        rec.update(rows=rows, errors=len(rows) - len(ok), first_error=next((r for r in rows if 'error' in r), None))
                        if ok: rec.update(output_tok_per_s=sum(r['output_tokens'] for r in ok) / wall, output_tokens=sum(r['output_tokens'] for r in ok), prompt_tokens=sum(r['prompt_tokens'] or 0 for r in ok),
                            ttft_p50=st.median(r['ttft_ms'] for r in ok), ttft_p95=pct([r['ttft_ms'] for r in ok], .95), tpot_p50=st.median(tp), tpot_p95=pct(tp, .95), tpot_p99=pct(tp, .99), ttft_p99=pct([r['ttft_ms'] for r in ok], .99), e2e_p99=pct([r['e2e_ms'] for r in ok], .99), e2e_p50=st.median(r['e2e_ms'] for r in ok), e2e_p95=pct([r['e2e_ms'] for r in ok], .95),
                            outputs_by_prompt={i: sorted({r['out_sha'][:12] for r in ok if r['id'] == i}) for i in {r['id'] for r in ok}})
                    finally:
                        if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
                        try: p.wait(timeout=120)
                        except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                        done.set(); th.join()
                    rec['gpu_peak_MiB'] = max(gpu) if gpu else None; rec['exit_code'] = p.returncode; records.append(rec)
                    print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k in ('output_tok_per_s', 'ttft_p50', 'tpot_p50', 'e2e_p50', 'errors', 'gpu_peak_MiB', 'output_tokens', 'prompt_tokens', 'first_error')}, flush=True)
                    (O / 'result.json').write_text(json.dumps({'plan': plan, 'failure': None, 'complete': False, 'records': records}, indent=1))
except BaseException as e:
    failure = {'type': type(e).__name__, 'message': str(e)}; raise
finally:
    (O / 'result.json').write_text(json.dumps({'plan': plan, 'failure': failure, 'complete': failure is None and len(records) == 16, 'records': records}, indent=1))
