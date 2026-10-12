"""Markdown trade-off table for Riley-only steady runs: python3 qwen_variant_table.py label=DIR[+DIR...] ...  (first label is the base)."""
import json, statistics as st, sys
variants = []
for a in sys.argv[1:]:
    label, dirs = a.split('=', 1); recs = []
    for d in dirs.split('+'):
        j = json.load(open(d + '/result.json')); assert j['failure'] is None, (d, j['failure']); recs += [r for r in j['records'] if r['engine'] == 'riley-scheduler']
    variants.append((label, recs))
cells = []
for _, recs in variants:
    for r in recs:
        if r['cell'] not in cells: cells.append(r['cell'])
cells.sort(key=lambda c: (int(c[1:].split('-')[0]), c))
outs = []
for label, recs in variants:
    by = {}
    for r in recs:
        for k, o in r['outputs_by_prompt'].items(): by.setdefault(k, set()).update(o)
    outs.append(by)
print('errors:', {label: sum(r['errors'] for r in recs) for label, recs in variants})
print('prompts with one output:', {variants[i][0]: '%d/%d' % (sum(len(x) == 1 for x in outs[i].values()), len(outs[i])) for i in range(len(variants))}, '| same outputs as base:', {variants[i][0]: '%d/%d' % (sum(outs[0].get(k) == v for k, v in outs[i].items()), len(outs[i])) for i in range(1, len(variants))})
f = lambda recs, c, k: st.mean(r[k] for r in recs if r['cell'] == c)
print('| 조건 | 구성 | tok/s | 변화 | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 / p99 ms | E2E p50 / p99 ms | GPU MiB |'); print('|---|---|---:|---:|---|---|---|---:|')
for c in cells:
    base = f(variants[0][1], c, 'output_tok_per_s') if any(r['cell'] == c for r in variants[0][1]) else None
    for label, recs in variants:
        if not any(r['cell'] == c for r in recs): continue
        t = f(recs, c, 'output_tok_per_s')
        print('| %s | %s | %.1f | %s | %.2f / %.2f / %.2f | %.1f / %.1f / %.1f | %.1f / %.1f | %.0f |' % (c, label, t, '—' if label == variants[0][0] or base is None else '%+.1f%%' % ((t / base - 1) * 100), f(recs, c, 'tpot_p50'), f(recs, c, 'tpot_p95'), f(recs, c, 'tpot_p99'), f(recs, c, 'ttft_p50'), f(recs, c, 'ttft_p95'), f(recs, c, 'ttft_p99'), f(recs, c, 'e2e_p50'), f(recs, c, 'e2e_p99'), max(r['gpu_peak_MiB'] for r in recs if r['cell'] == c)))
