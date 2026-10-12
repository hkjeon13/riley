"""Matched production-binary stability lane: sustained load, cancellation (idle and under load),
exact re-request, KV drain/reuse, memory trend and shutdown reclamation. Never promotes performance.

Derived from kernel-batch12-stability-controller-draft-attempt01/stability.py; the launch argv/env is the
exact matched serving launch of the bound quiet attempt.
"""
import argparse, hashlib, http.client, json, os, signal, socket, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from serving_token_client_v2 import TokenHttpClient, TokenReference, SSEFramer
from mixed_phase_v34 import mixed_phase

ROOT = Path('/data/riley-serving-261007')
CODE = Path(__file__).parent
PLAN = json.loads((CODE / 'stability-plan.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=1, allow_nan=False); f.write('\n')


def smi(query, kind='gpu'):
    return subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + query, '--format=csv,noheader,nounits'], text=True).strip()


def host():
    return {'time_ns': time.time_ns(), 'pressure': {k: Path('/proc/pressure', k).read_text() for k in ['cpu', 'io', 'memory']},
            'loadavg': Path('/proc/loadavg').read_text(), 'gpu': smi('uuid,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu')}


def group_memory(group):
    observed = {}
    for row in subprocess.check_output(['ps', '-e', '-o', 'pid=,pgid=,rss='], text=True).splitlines():
        pid, pgid, rss = map(int, row.split())
        if pgid == group:
            observed[str(pid)] = rss
    return {'rss_sum_KiB': sum(observed.values()), 'pids': len(observed)}


def gate(out, label):
    """Gate v2 of the matched plan: CPU/memory pressure abort, IO pressure awaited then recorded."""
    policy = PLAN['host_start_gate']; records = []; streak = 0; hard = 0; start = time.monotonic()
    while time.monotonic() - start < policy['timeout_seconds']:
        h = host(); records.append(h)
        avg = lambda r, kind: float(next(x for x in h['pressure'][r].splitlines() if x.startswith(kind + ' ')).split()[1].split('=')[1])
        hard_quiet = avg('cpu', 'some') <= policy['cpu_some_max'] and avg('memory', 'full') <= policy['memory_full_max']
        io_quiet = avg('io', 'full') <= policy['io_full_max']
        hard = hard + 1 if hard_quiet else 0; streak = streak + 1 if hard_quiet and io_quiet else 0
        if streak >= policy['consecutive'] or (hard >= policy['consecutive'] and time.monotonic() - start >= policy['io_soft_wait_seconds']):
            write(out / (label + '-host-gate.json'), {'passed': True, 'io_elevated': streak < policy['consecutive'], 'samples': records}); return
        time.sleep(policy['interval_seconds'])
    write(out / (label + '-host-gate.json'), {'passed': False, 'samples': records})
    raise RuntimeError('host gate timeout ' + label)


def metrics(port):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    try:
        c.request('GET', '/metrics'); r = c.getresponse(); b = r.read(); assert r.status == 200
        return {'time_ns': time.time_ns(), 'json': json.loads(b)}
    finally:
        c.close()


def drained(port, out, label, sink):
    observations = []; deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        m = metrics(port); observations.append(m); x = m['json']
        if all(x[k] == 0 for k in ['active_requests', 'waiting_requests', 'kv_allocated_blocks']):
            sink.append({'label': label, 'polls': len(observations), 'elapsed_ms': (m['time_ns'] - observations[0]['time_ns']) / 1e6, 'final': x}); return x
        time.sleep(.02)
    write(out / (label + '-drain-failure.json'), observations)
    raise RuntimeError('ownership did not drain ' + label)


def cancel(port, body, after_token, expected_prefix):
    """Open a streaming completion and drop the connection before the response (early) or after the first token (decode)."""
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    seen = False; tokens_seen = []; status = None; response = None; framer = SSEFramer(); raw = 0
    try:
        c.request('POST', '/v1/completions', body=encoded, headers={'Content-Type': 'application/json', 'Connection': 'close'})
        if after_token:
            response = c.getresponse(); status = response.status
            assert status == 200 and 'text/event-stream' in response.getheader('Content-Type', ''), 'cancel response head'
            while not seen:
                chunk = response.read1(4096)
                if not chunk:
                    raise RuntimeError('EOF before cancellation token')
                raw += len(chunk)
                for payload, _ in framer.feed(chunk, time.perf_counter_ns()):
                    assert payload != b'[DONE]', 'DONE before cancellation token'
                    value = json.loads(payload); assert 'error' not in value
                    for choice in value.get('choices', []):
                        tokens_seen += choice.get('token_ids', [])
                    if tokens_seen:
                        if expected_prefix is not None:
                            assert tuple(tokens_seen) == tuple(expected_prefix[:len(tokens_seen)]), 'cancellation prefix reference mismatch'
                        seen = True; break
    finally:
        if response:
            response.close()
        c.close()
    return {'after_first_token': seen, 'http_status': status, 'tokens_seen': tokens_seen, 'body_bytes_before_close': raw, 'closed_ns': time.time_ns(),
            'phase_classification': 'after-first-token' if seen else 'closed right after request write; server phase (queued/prefill) not observable from the client'}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--case', required=True); a.add_argument('--engine', required=True); a.add_argument('--attempt', type=int, required=True)
    args = a.parse_args()
    assert PLAN['status'] == 'FROZEN_FOR_EXECUTION' and args.case in PLAN['cases'] and args.engine in PLAN['engines']
    for path, digest in PLAN['pins'].items():
        assert sha(path) == digest, path
    assert not smi('pid', 'compute-apps'), 'foreign GPU compute process'
    riley = args.engine != 'vllm'
    out = ROOT / f"{PLAN['output_prefix']}-{args.case}-{args.engine}-attempt{args.attempt:02d}"; out.mkdir()
    failure = None; p = None; observer = None; done = threading.Event(); samples = []; drains = []; summary = {}
    try:
        launch = json.loads((ROOT / PLAN['matched_attempt'] / f"{args.case}-r0-{args.engine}-launch.json").read_text())
        argv = launch['argv'].copy(); env = launch['env'].copy(); assert sha(argv[0]) == launch['binary_sha256'] == PLAN['binaries'][args.engine]
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        if riley:
            argv[argv.index('--bind') + 1] = f'127.0.0.1:{port}'; env['RILEY_SHUTDOWN_METRICS_PATH'] = str(out / 'native-close.json')
        else:
            argv[argv.index('--port') + 1] = str(port)
        write(out / 'preparation.json', {'argv': argv, 'env': env, 'matched_launch': launch, 'plan_sha256': sha(CODE / 'stability-plan.json'), 'performance_qualification': False})
        concurrency = int(args.case.split('-')[0][1:]); workload = args.case.split('-')[1]
        data = json.loads((CODE / 'fixtures.json').read_text())[workload]; corpus = data['corpus']; ids = [x['id'] for x in corpus]
        refs = [TokenReference('g04-smol', tuple(b['choices'][0]['prompt_token_ids']) if 'prompt_token_ids' in b['choices'][0] else tuple(c['prompt_token_ids']), tuple(b['choices'][0]['token_ids']), b['choices'][0]['text'], b['choices'][0]['finish_reason']) for c, b in zip(corpus, data['responses'])]
        gate(out, 'startup')
        log = (out / 'server.log').open('x')
        p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

        def observe():
            while not done.wait(1):
                h = host(); h['rss'] = group_memory(p.pid); samples.append(h)
        observer = threading.Thread(target=observe); observer.start()
        deadline = time.monotonic() + 300
        while True:
            assert p.poll() is None and time.monotonic() < deadline, 'startup failed'
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
            try:
                c.request('GET', '/v1/models'); r = c.getresponse(); r.read()
                if r.status == 200:
                    break
            except (OSError, http.client.HTTPException):
                pass
            finally:
                c.close()
            time.sleep(.1)
        with TokenHttpClient() as client:
            def request(index):
                i = index % len(corpus); c = corpus[i]
                row = client.request(port, {'model': 'g04-smol', 'prompt': c['prompt'], 'max_tokens': c['max_tokens'], 'temperature': 0}, refs[i], streaming=True, mode='observe')
                row['corpus_id'] = c['id']; return row

            def check(rows, account, label):
                assert account['completed'] and account['failed'] == 0, label + ' incomplete'
                assert all(x['protocol_valid'] and x['transport_complete'] and x['http_status'] == 200 for x in rows), label + ' transport'
                if riley:
                    assert account['strict_reference_pass'], label + ' exact reference failed'
                return {'requests': len(rows), 'reference_matches': account['reference_matches'], 'output_tokens': sum(len(x['token_ids']) for x in rows)}

            gate(out, 'warmup')
            rows, account, _ = mixed_phase(client, request, concurrency=concurrency, count=96, corpus_ids=ids, phase='warmup')
            summary['warmup'] = check(rows, account, 'warmup')
            if riley:
                drained(port, out, 'after-warmup', drains)
            gate(out, 'sustained')
            mark = {'sustained_first_sample': len(samples)}
            start = time.monotonic(); rounds = []
            while time.monotonic() - start < PLAN['sustained_seconds']:
                rows, account, s = mixed_phase(client, request, concurrency=concurrency, count=384, corpus_ids=ids, phase='retained')
                r = check(rows, account, f'sustained round {len(rounds)}')
                r.update(round=len(rounds), elapsed_s=time.monotonic() - start, throughput=s['successful_output_tokens_per_wall_second'],
                         ttft_p50=s['token_ttft_ms']['median'], tpot_p50=s['token_tpot_ms']['median'], tpot_p99=s['token_tpot_ms']['p99'], e2e_p99=s['e2e_ms']['p99'])
                if riley:
                    r['kv_allocated_blocks_after_round'] = metrics(port)['json']['kv_allocated_blocks']
                rounds.append(r)
            mark['sustained_last_sample'] = len(samples)
            summary['sustained'] = {'seconds': time.monotonic() - start, 'rounds': rounds, 'requests': sum(r['requests'] for r in rounds), 'errors': 0}
            if riley:
                drained(port, out, 'after-sustained', drains)
            # Idle cancellation: early and after-first-token, each followed by exact re-request of the whole corpus and a full drain.
            cancels = []
            body = lambda c: {'model': 'g04-smol', 'prompt': c['prompt'], 'max_tokens': c['max_tokens'], 'temperature': 0, 'stream': True, 'return_token_ids': True, 'stream_options': {'include_usage': True}}
            for after in [False, True]:
                for trial in range(PLAN['cancellation_trials_each_mode']):
                    i = trial % len(corpus)
                    result = cancel(port, body(corpus[i]), after, data['responses'][i]['choices'][0]['token_ids'] if riley else None)
                    result.update(mode='decode' if after else 'early', trial=trial, context='idle')
                    if riley:
                        drained(port, out, f"idle-{result['mode']}-{trial:02d}", drains)
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(pool.map(request, range(len(corpus))))
                    assert all(x['protocol_valid'] and x['transport_complete'] and x['http_status'] == 200 for x in rows), 're-request transport'
                    if riley:
                        assert all(x['reference_match'] for x in rows), 're-request exact reference failed'
                        drained(port, out, f"idle-{result['mode']}-{trial:02d}-reuse", drains)
                    result['re_requests_exact'] = sum(x['reference_match'] for x in rows); cancels.append(result)
            # Cancellation under load: cancels interleave with a full-concurrency exact workload, whose rows must stay exact.
            gate(out, 'cancel-under-load')
            loaded = []; stop = threading.Event(); errors = []

            def canceller():
                trial = 0
                try:
                    while not stop.is_set() and trial < PLAN['loaded_cancellations']:
                        i = trial % len(corpus); after = trial % 2 == 1
                        result = cancel(port, body(corpus[i]), after, data['responses'][i]['choices'][0]['token_ids'] if riley else None)
                        result.update(mode='decode' if after else 'early', trial=trial, context='under-load'); loaded.append(result); trial += 1
                        time.sleep(PLAN['loaded_cancel_interval_seconds'])
                except BaseException as e:
                    errors.append({'type': type(e).__name__, 'message': str(e)})
            t = threading.Thread(target=canceller); t.start(); load_rounds = []
            try:
                while t.is_alive():
                    rows, account, s = mixed_phase(client, request, concurrency=concurrency, count=PLAN['loaded_round_requests'], corpus_ids=ids, phase='retained')
                    load_rounds.append(check(rows, account, f'loaded round {len(load_rounds)}'))
            finally:
                stop.set(); t.join()
            assert not errors, 'cancellation under load failed: ' + json.dumps(errors)
            assert len(loaded) == PLAN['loaded_cancellations']
            if riley:
                drained(port, out, 'after-cancel-under-load', drains)
            rows, account, _ = mixed_phase(client, request, concurrency=concurrency, count=96, corpus_ids=ids, phase='retained')
            summary['after_cancel_exact_round'] = check(rows, account, 'post-cancel round')
            if riley:
                final = drained(port, out, 'final', drains); summary['final_metrics'] = final
                summary['server_counters'] = final['counters']
                assert final['counters']['cancellations'] + final['counters']['disconnects'] >= 1, 'server observed no cancellation'
            summary['cancellations'] = {'idle': cancels, 'under_load': loaded, 'load_rounds': load_rounds}
            summary['sample_marks'] = mark
    except BaseException as e:
        failure = {'type': type(e).__name__, 'message': str(e)}; raise
    finally:
        exit_code = None
        if p:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=10)
            exit_code = p.returncode
        done.set()
        if observer:
            observer.join(timeout=15)
        time.sleep(2)
        after = {'gpu_compute_apps': smi('pid', 'compute-apps'), 'gpu': smi('memory.used'), 'port_listening': None}
        if p:
            with socket.socket() as s:
                after['port_listening'] = s.connect_ex(('127.0.0.1', port)) == 0
        close = json.loads((out / 'native-close.json').read_text()) if (out / 'native-close.json').exists() else None
        reclaimed = None
        if failure is None:
            reclaimed = exit_code == 0 and not after['gpu_compute_apps'] and after['port_listening'] is False
            if riley:
                reclaimed = reclaimed and close is not None and close['active_requests'] == close['waiting_requests'] == close['kv_allocated_blocks'] == 0 \
                    and all(close['allocation'][k] == 0 for k in ['device_live_count', 'device_live_bytes', 'pinned_live_count', 'pinned_live_bytes'])
            if not reclaimed:
                failure = {'type': 'ReclamationFailure', 'message': 'process/GPU/port/native allocation not reclaimed'}
        write(out / 'host-samples.json', samples); write(out / 'drains.json', drains)
        write(out / 'completion.json', {'failure': failure, 'case': args.case, 'engine': args.engine, 'exit_code': exit_code, 'after_stop': after, 'native_close': close,
                                        'reclaimed': reclaimed, 'summary': summary, 'no_exclusions': True, 'performance_qualification': False})
    assert failure is None, failure


if __name__ == '__main__':
    main()
