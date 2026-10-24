"""Step-level timeline from an Nsight node-traced SQLite export.

A step ends with the engine's sampling kernel. Each step is classified as
mixed (prompt tokens present) or decode-only and summarised by span, kernel
sum, inter-step gap and kernel categories.
"""
import json, sqlite3, statistics as st, sys

def category(name):
    n = name.lower()
    if 'prefill_mma' in n: return 'attn_prompt'
    if 'partial_state' in n or 'global_merge' in n or 'flash_fwd' in n or 'segment_state' in n: return 'attn'
    if 'reshape_and_cache' in n or 'kv_write' in n: return 'kv_write'
    if 'gemm' in n or n.startswith('kernel2') or 'splitkreduce' in n or 'cutlass' in n or 'gemv' in n: return 'gemm'
    if 'rms' in n or 'norm' in n: return 'norm'
    if 'silu' in n or 'mul_silu' in n: return 'act'
    if 'argmax' in n or 'sample' in n: return 'sample'
    return 'other'

def analyse(path, engine):
    c = sqlite3.connect(path)
    rows = c.execute('select k.start, k.end, s.value, k.graphId, k.gridX, k.gridY, k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start').fetchall()
    sampler = 'bf16_argmax_kernel' if engine == 'riley' else '_gumbel_sample_kernel'
    steps, current = [], []
    for row in rows:
        current.append(row)
        if row[2] == sampler:
            steps.append(current); current = []
    out = []
    previous_end = None
    for kernels in steps:
        start = kernels[0][0]; end = max(k[1] for k in kernels)
        names = [k[2] for k in kernels]
        if engine == 'riley':
            mixed = any('prefill_mma' in n for n in names)
        else:
            mixed = any(k[2] == 'flash_fwd_splitkv_kernel' and k[3] is None for k in kernels)
        cats = {}
        for k in kernels:
            cats[category(k[2])] = cats.get(category(k[2]), 0) + (k[1] - k[0]) / 1e3
        flash = [k for k in kernels if k[2] == 'flash_fwd_splitkv_kernel']
        out.append(dict(mixed=mixed, span_us=(end - start) / 1e3, kernel_us=sum((k[1]-k[0]) for k in kernels) / 1e3,
                        gap_us=(start - previous_end) / 1e3 if previous_end else None, kernels=len(kernels), cats=cats,
                        flash_grid=(flash[0][4], flash[0][5], flash[0][6]) if flash else None))
        previous_end = end
    return out

def summary(steps):
    res = {}
    for label, sel in (('mixed', True), ('decode', False)):
        s = [x for x in steps if x['mixed'] == sel]
        if not s: continue
        cats = {}
        for x in s:
            for k, v in x['cats'].items(): cats[k] = cats.get(k, 0) + v
        gaps = [x['gap_us'] for x in s if x['gap_us'] is not None]
        res[label] = dict(steps=len(s), span_ms_total=sum(x['span_us'] for x in s)/1e3, span_us_median=st.median(x['span_us'] for x in s),
                          kernel_ms_total=sum(x['kernel_us'] for x in s)/1e3, gap_ms_total=sum(gaps)/1e3, gap_us_median=st.median(gaps) if gaps else None,
                          kernels_per_step=st.median(x['kernels'] for x in s), cats_ms={k: round(v/1e3, 2) for k, v in sorted(cats.items())})
    return res

if __name__ == '__main__':
    result = {}
    for engine, path in (('riley', sys.argv[1]), ('vllm', sys.argv[2])):
        steps = analyse(path, engine)
        result[engine] = dict(summary=summary(steps), steps=steps)
        print(engine, json.dumps(result[engine]['summary'], indent=1))
    json.dump(result, open(sys.argv[3], 'w'))
