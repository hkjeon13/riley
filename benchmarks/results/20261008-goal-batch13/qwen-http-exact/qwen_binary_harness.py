"""Real `riley serve` process check of the HF-exact profile: one streaming HTTP request, tokens and observed logit rows vs the HF reference."""
import hashlib, http.client, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
R = Path('/data/riley-serving-261007'); O = R / f'qwen-http-exact-binary-attempt{int(sys.argv[1]):02d}'; O.mkdir(); rows = O / 'rows'; rows.mkdir()
sha = lambda b: hashlib.sha256(b).hexdigest()
base = Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host'); teacher = base / 'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
model = '/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1'
binary = R / 'qwen-http-exact-target-attempt01/release/riley'
w = json.loads((base / 'inputs/qwen3b-c8-p2048-o128.json').read_text()); hf = json.loads((teacher / 'teacher-forced-oracle.json').read_text())['generation']['teacher_token_ids']
side = (teacher / 'cache-on-logits.safetensors').read_bytes(); n = int.from_bytes(side[:8], 'little'); expected = side[8 + n:]; RB = 151936 * 2; assert len(expected) == 128 * RB
with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
argv = [str(binary), 'serve', '--model', model, '--model-id', 'qwen3b', '--bind', f'127.0.0.1:{port}', '--max-active-sequences', '1', '--max-waiting-requests', '2',
        '--max-sequence-tokens', '2176', '--max-output-tokens', '128', '--max-weight-bytes', str(8 << 30), '--numerics-profile', 'hf-eager-qwen-p2048-o128-probe-v1', '--logit-observation-dir', str(rows)]
env = {'HOME': '/home/psyche', 'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '0', 'LD_LIBRARY_PATH': '/data/cuda-12.8.1/lib64', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'RILEY_SHUTDOWN_METRICS_PATH': str(O / 'native-close.json')}
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
log = (O / 'server.log').open('x'); p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True); result = {'argv': argv, 'binary_sha256': sha(binary.read_bytes())}
try:
    deadline = time.monotonic() + 300
    while True:
        assert p.poll() is None and time.monotonic() < deadline, 'startup failed'
        c = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
        try:
            c.request('GET', '/v1/models'); r = c.getresponse(); r.read()
            if r.status == 200: break
        except (OSError, http.client.HTTPException): pass
        finally: c.close()
        time.sleep(.2)
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=300)
    c.request('POST', '/v1/completions', body=json.dumps({'model': 'qwen3b', 'prompt': w['prompt'], 'max_tokens': 128, 'temperature': 0, 'top_p': 1.0, 'stream': True, 'return_token_ids': True}), headers={'Content-Type': 'application/json'})
    r = c.getresponse(); body = r.read().decode(); c.close(); tokens = []; rid = None; finish = None
    for line in body.splitlines():
        if line.startswith('data: ') and line != 'data: [DONE]':
            v = json.loads(line[6:]); rid = v['id']
            for ch in v['choices']:
                tokens += ch.get('token_ids') or []; finish = ch.get('finish_reason') or finish
    exact = [(rows / f'{rid}-{i:03d}.bf16').read_bytes() == expected[i * RB:(i + 1) * RB] for i in range(128)]
    result.update(http_status=r.status, tokens_match_hf=tokens == hf, finish_reason=finish, logit_rows_exact=sum(exact), first_non_exact_row=next((i for i, e in enumerate(exact) if not e), None), row_files=len(list(rows.iterdir())))
finally:
    os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=120)
    close = json.loads((O / 'native-close.json').read_text()) if (O / 'native-close.json').exists() else None
    result.update(exit_code=p.returncode, native_close={k: close[k] for k in ['active_requests', 'waiting_requests', 'allocation']} if close else None)
    result['pass'] = bool(result.get('tokens_match_hf') and result.get('logit_rows_exact') == 128 and p.returncode == 0 and close and all(v == 0 for v in close['allocation'].values()))
    (O / 'result.json').write_text(json.dumps(result, indent=1)); print(json.dumps({k: v for k, v in result.items() if k != 'argv'}))
