"""Token-level reference for the Qwen scheduler route on the 15 benchmark prompts.
Predeclared: reference = HF transformers eager (attn_implementation=eager, use_cache=True), BF16, greedy, batch 1, default EOS,
max_new_tokens = fixture max_tokens. Riley = adopted scheduler-route flags over HTTP at C1 (sequential) and C32 (all prompts
repeated concurrently). Metric per prompt: full token-sequence equality, else first differing index. No exclusions.
This is NOT the pinned-image byte-exact logit contract; it is a separate, weaker (token-level) check with this venv's torch."""
import concurrent.futures, hashlib, http.client, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
R = Path('/data/riley-serving-261007'); O = R / f'qwen-hf-token-reference-attempt{int(sys.argv[1]):02d}'; O.mkdir(); import shutil; shutil.copy(R / 'qwen-hf-token-reference-attempt01/hf.json', O / 'hf.json')
fixtures = json.loads((R / 'kernel-batch13-controller-attempt01/fixtures.json').read_text())
model = '/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1'; riley = str(R / 'qwen-riley-rowwise1-candidate')
items = [dict(c, workload=w) for w in ('fixed', 'natural') for c in fixtures[w]['corpus']]
(O / 'plan.json').write_text(json.dumps({'doc': __doc__, 'prompts': [(i['workload'], i['id'], i['max_tokens']) for i in items], 'riley_sha256': hashlib.sha256(Path(riley).read_bytes()).hexdigest(), 'time_ns': time.time_ns()}, indent=1))
if not (O / 'hf.json').exists():
    code = r'''
import json, sys, torch, transformers
model, out = sys.argv[1], sys.argv[2]; items = json.load(sys.stdin)
tok = transformers.AutoTokenizer.from_pretrained(model); m = transformers.AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
res = {'transformers': transformers.__version__, 'torch': torch.__version__, 'eos': m.generation_config.eos_token_id, 'rows': []}
for it in items:
    ids = tok(it['prompt'], return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    with torch.no_grad(): g = m.generate(ids, do_sample=False, max_new_tokens=it['max_tokens'], use_cache=True, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    res['rows'].append({'workload': it['workload'], 'id': it['id'], 'prompt_tokens': ids.shape[1], 'tokens': g[0, ids.shape[1]:].tolist()})
json.dump(res, open(out, 'w'))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1')
    p = subprocess.run([str(R / 'vllm0271-venv/bin/python'), '-c', code, model, str(O / 'hf.json')], input=json.dumps(items), text=True, env=env, stdout=(O / 'hf.log').open('w'), stderr=subprocess.STDOUT)
    assert p.returncode == 0, 'HF reference failed'
hf = json.loads((O / 'hf.json').read_text())
def request(port, item):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=600); toks = []; buf = b''
    try:
        c.request('POST', '/v1/completions', body=json.dumps({'model': 'qwen3b', 'prompt': item['prompt'], 'max_tokens': item['max_tokens'], 'temperature': 0, 'stream': True, 'return_token_ids': True}), headers={'Content-Type': 'application/json'})
        r = c.getresponse(); assert r.status == 200, (r.status, r.read()[:200])
        while True:
            ch = r.read1(65536)
            if not ch: break
            buf += ch
            while b'\n\n' in buf:
                f, buf = buf.split(b'\n\n', 1)
                for line in f.split(b'\n'):
                    if line.startswith(b'data: ') and line != b'data: [DONE]':
                        for x in json.loads(line[6:]).get('choices', []): toks += x.get('token_ids') or []
        return toks
    finally: c.close()
out = {'hf': {k: hf[k] for k in ('transformers', 'torch', 'eos')}, 'runs': []}
for conc in (1, 32):
    with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    argv = [riley, 'serve', '--bind', f'127.0.0.1:{port}', '--model', model, '--model-id', 'qwen3b', '--max-active-sequences', str(conc), '--max-waiting-requests', '64', '--max-sequence-tokens', '1024', '--max-output-tokens', '128',
            '--max-weight-bytes', str(8 << 30), '--batch-token-budget', '256', '--prefill-chunk-tokens', '256', '--gemm-reduction-policy', 'first-heuristic-split-k-v1', '--kv-blocks', str(conc * 64), '--decode-attention-backend', 'native-bf16-paged-split-gqa-d128-two-stage',
            '--execution-graph-policy', 'disabled', '--reduction-profile', 'canonical-v1', '--projection-bias-backend', 'merged-qkv-strict-staged-v1', '--batch-shape-policy', 'power-of-two', '--execution-completion', 'iteration-batch',
            '--metadata-transport', 'synchronous', '--residual-rmsnorm', 'fused', '--sampling-backend', 'gpu-greedy', '--kv-cache-write', 'fused-split-v1', '--decode-attention-producer', 'rowwise-tensor-core-v1', '--batch-shape-buckets', '1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256', '--decode-attention-merge', 'global-max-v1', '--decode-step-graph', 'capture-v1', '--mlp-activation', 'fused-silu-multiply-v1', '--query-key-rope', 'fused-split-v1', '--gemm-algorithm-selection', 'bit-equivalent-table-v1', '--mlp-residual-norm', 'fused-next-norm-v1', '--lm-head-rows', 'output-rows-v1']
    p = subprocess.Popen(argv, env={'HOME': '/home/psyche', 'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '0', 'LD_LIBRARY_PATH': '/data/cuda-12.8.1/lib64'}, stdout=(O / f'riley-c{conc}.log').open('w'), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 900
        while True:
            assert p.poll() is None and time.monotonic() < deadline, 'startup failed'
            k = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
            try:
                k.request('GET', '/v1/models'); rr = k.getresponse(); rr.read()
                if rr.status == 200: break
            except (OSError, http.client.HTTPException): pass
            finally: k.close()
            time.sleep(.5)
        n = len(items) * (1 if conc == 1 else 4)
        with concurrent.futures.ThreadPoolExecutor(conc) as pool: got = list(pool.map(lambda i: request(port, items[i % len(items)]), range(n)))
    finally:
        if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=120)
    rows = []
    for i, toks in enumerate(got):
        ref = hf['rows'][i % len(items)]; first = next((j for j, (a, b) in enumerate(zip(toks, ref['tokens'])) if a != b), None)
        if first is None and len(toks) != len(ref['tokens']): first = min(len(toks), len(ref['tokens']))
        rows.append({'workload': ref['workload'], 'id': ref['id'], 'riley_len': len(toks), 'hf_len': len(ref['tokens']), 'first_difference': first, 'riley_tokens': toks})
    out['runs'].append({'concurrency': conc, 'requests': n, 'exact': sum(r['first_difference'] is None for r in rows), 'rows': rows})
    print('C', conc, 'exact', out['runs'][-1]['exact'], '/', n, [(r['id'], r['first_difference'], r['hf_len']) for r in rows[:len(items)] if r['first_difference'] is not None], flush=True)
(O / 'result.json').write_text(json.dumps(out, indent=1))
