# perf_extract.py OUT.json: per-model, per-cell means of the two repeats (as a100_verdict.py) for the A100 perf docs.
import json, sys, statistics as st
E = '/mnt/efs/nlp/psyche/riley-261013/evidence/'
SOURCES = [('Qwen2.5-0.5B', 'qual-q25-0-5b-attempt01'), ('Qwen2.5-1.5B', 'd1-qual-1-5b-attempt05'), ('Qwen2.5-3B', 'd1-qual-3b-attempt05'),
    ('Qwen2.5-7B', 'qual-7b-attempt06'), ('Qwen2.5-14B', 'qual-nv-14b-vmem09-attempt02'), ('Qwen2.5-Coder-1.5B', 'qual-nv-coder-1-5b-attempt01'),
    ('Qwen2.5-Coder-3B', 'qual-nv-coder-3b-attempt01'), ('Qwen2.5-Coder-7B', 'qual-nv-coder-7b-attempt01'), ('Qwen2.5-Coder-14B', 'qual-nv-coder-14b-attempt01'),
    ('DeepSeek-R1-Distill-Qwen-1.5B', 'qual-nv-r1-qwen-1-5b-attempt04'), ('DeepSeek-R1-Distill-Qwen-7B', 'qual-nv-r1-qwen-7b-attempt04'),
    ('DeepSeek-R1-Distill-Qwen-14B', 'qual-nv-r1-qwen-14b-attempt04'), ('DeepSeek-R1-Distill-Llama-8B', 'qual-nv-r1-llama-8b-attempt03'),
    ('Qwen3-0.6B', 'nv53-qwen3-0-6b-attempt01'), ('Qwen3-1.7B', 'qual-nv-qwen3-1-7b-attempt02'), ('Qwen3-4B', 'qual-nv-qwen3-4b-attempt01'),
    ('Qwen3-8B', 'qual-nv-qwen3-8b-attempt01'), ('Qwen3-14B', 'qual-nv-qwen3-14b-attempt04'), ('SmolLM2-135M', 'qual-smol-135m-attempt01'),
    ('SmolLM2-360M', 'qual-smol-360m-attempt01'), ('SmolLM2-1.7B', 'qual-nv-smol-1-7b-attempt01'), ('Llama-3.2-1B', 'qual-nv-llama32-1b-attempt02'),
    ('Llama-3.2-3B', 'qual-nv-llama32-3b-attempt01'), ('Llama-3.1-8B', 'qual-nv-llama31-8b-attempt01')] + [(n, r) for n, r in [a.split('=') for a in sys.argv[2:]]]
out = {}
for name, run in SOURCES:
    d = json.load(open(E + run + '/result.json')); cells = {}
    for r in d['records']: cells.setdefault(r['cell'], {}).setdefault(r['engine'], []).append(r)
    out[name] = {'run': run, 'riley_sha256': d['plan'].get('riley_sha256', '')[:8], 'cells': {c: {eng: {k: st.mean(x[k] for x in v) for k in ('output_tok_per_s', 'ttft_p50', 'tpot_p50', 'e2e_p50')} for eng, v in e.items()} for c, e in cells.items()}}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(len(out), 'models')
