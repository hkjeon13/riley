"""Same-run Riley vs vLLM table and the 2026-10-12 goal checks: python3 qwen_final_table.py RESULT_DIR"""
import json, statistics as st, sys
d = json.load(open(sys.argv[1] + '/result.json')); print('complete', d['complete'], 'failure', d['failure'], 'records', len(d['records']))
cells = []
for r in d['records']:
    if r['cell'] not in cells: cells.append(r['cell'])
def m(cell, eng, k): return st.mean(r[k] for r in d['records'] if r['cell'] == cell and r['engine'] == eng)
def spread(cell, eng):
    v = [r['output_tok_per_s'] for r in d['records'] if r['cell'] == cell and r['engine'] == eng]; return (max(v) - min(v)) / st.mean(v) * 100
R, V = 'riley-scheduler', 'vllm'
print('errors', {e: sum(r['errors'] for r in d['records'] if r['engine'] == e) for e in (R, V)})
by = {}
for r in d['records']:
    if r['engine'] == R:
        for k, o in r['outputs_by_prompt'].items(): by.setdefault(k, set()).update(o)
print('Riley prompts with one output across all cells and repeats: %d/%d' % (sum(len(x) == 1 for x in by.values()), len(by)))
print('| 조건 | 처리량 tok/s | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | E2E p50 / p99 ms | GPU MiB |'); print('|---|---|---|---|---|---|')
for c in cells:
    f = lambda e, ks: ' / '.join(('%.1f' if 'ttft' in k or 'e2e' in k else '%.2f') % m(c, e, k) for k in ks)
    print('| %s | %.1f / %.1f | %s vs %s | %s vs %s | %s vs %s | %.0f / %.0f |' % (c, m(c, R, 'output_tok_per_s'), m(c, V, 'output_tok_per_s'), f(R, ('ttft_p50', 'ttft_p95', 'ttft_p99')), f(V, ('ttft_p50', 'ttft_p95', 'ttft_p99')),
          f(R, ('tpot_p50', 'tpot_p95', 'tpot_p99')), f(V, ('tpot_p50', 'tpot_p95', 'tpot_p99')), f(R, ('e2e_p50', 'e2e_p99')), f(V, ('e2e_p50', 'e2e_p99')), max(r['gpu_peak_MiB'] for r in d['records'] if r['cell'] == c and r['engine'] == R), max(r['gpu_peak_MiB'] for r in d['records'] if r['cell'] == c and r['engine'] == V)))
print(); print('| 조건 | 처리량 R/V | TPOT p50 R/V | TTFT p50 R/V | 필수 판정 | TTFT p95 R/V | TTFT p99 R/V | TPOT p99 R/V | 반복 간 처리량 편차 R / V |'); print('|---|---:|---:|---:|---|---:|---:|---:|---|')
ok_all = True
for c in cells:
    rt = lambda k: m(c, R, k) / m(c, V, k)
    conc = int(c[1:].split('-')[0])
    need = [('TTFT p50', rt('ttft_p50') <= 1.0)] + ([('처리량', rt('output_tok_per_s') >= 1.0), ('TPOT p50', rt('tpot_p50') <= 1.0)] if conc >= 8 else [])
    miss = [n for n, ok in need if not ok]; ok_all &= not miss
    print('| %s | %.3f | %.3f | %.3f | %s | %.3f | %.3f | %.3f | %.1f%% / %.1f%% |' % (c, rt('output_tok_per_s'), rt('tpot_p50'), rt('ttft_p50'), '충족' if not miss else '미달: ' + ', '.join(miss), rt('ttft_p95'), rt('ttft_p99'), rt('tpot_p99'), spread(c, R), spread(c, V)))
print('mandatory items all met:', ok_all)
