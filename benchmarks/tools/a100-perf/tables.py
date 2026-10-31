"""tables.py A100-PERF.json DOC.md...: rewrite the metric tables of the A100 perf docs from extract.py output.

docs/model-coverage-a100-perf-table.md has one table per metric (row = model, column = cell);
docs/model-coverage-a100-perf-by-model.md has one table (8 rows per model). A cell is bold where Riley
misses the criterion of a100_verdict.py: C8-C32 throughput below or TPOT above vLLM, C1 throughput or
TPOT outside +-3 %, TTFT p50 above vLLM. E2E, and the by-model table, are never bold.
"""
import json, re, sys

CELLS = [f'c{c}-{w}' for c in (1, 8, 16, 32) for w in ('fixed', 'natural')]
FORMAT = {'output_tok_per_s': '{:.0f}', 'ttft_p50': '{:.1f}', 'tpot_p50': '{:.2f}', 'e2e_p50': '{:.0f}'}
HEADINGS = {'## 처리량': 'output_tok_per_s', '## TTFT': 'ttft_p50', '## TPOT': 'tpot_p50', '## E2E': 'e2e_p50'}

def miss(cell, key, r, v):
    if key == 'ttft_p50': return r > v
    if key == 'e2e_p50': return False
    ratio = r / v
    if cell.startswith('c1-'): return not (0.97 <= ratio <= 1.03) and (ratio < 1 if key == 'output_tok_per_s' else ratio > 1)
    return ratio < 1 if key == 'output_tok_per_s' else ratio > 1

def value(data, name, cell, key, bold=True):
    r, v = (data[name]['cells'][cell][e][key] for e in ('riley-scheduler', 'vllm'))
    text = f'{FORMAT[key].format(r)} / {FORMAT[key].format(v)}'
    return f'**{text}**' if bold and miss(cell, key, r, v) else text

def metric_tables(data, lines):
    out, key, index = [], None, 0
    while index < len(lines):
        line = lines[index]
        key = next((k for h, k in HEADINGS.items() if line.startswith(h)), key if not line.startswith('## ') else None)
        if key and line.startswith('| 모델 | c1-fixed'):
            out += [line, lines[index + 1]]; index += 2
            while index < len(lines) and lines[index].startswith('| '): index += 1
            out += ['| ' + ' | '.join([name] + [value(data, name, c, key) for c in CELLS]) + ' |' for name in data]
            key = None; continue
        out.append(line); index += 1
    return out

def by_model_table(data, lines):
    out, index = [], 0
    while index < len(lines):
        line = lines[index]
        if line.startswith('| 모델 | 셀 |'):
            out += [line, lines[index + 1]]; index += 2
            while index < len(lines) and lines[index].startswith('| '): index += 1
            for name in data:
                for position, cell in enumerate(CELLS):
                    out.append('| ' + ' | '.join([name if position == 0 else '', cell] + [value(data, name, cell, k, bold=False) for k in FORMAT]) + ' |')
            continue
        out.append(line); index += 1
    return out

if __name__ == '__main__':
    data = json.load(open(sys.argv[1]))
    for path in sys.argv[2:]:
        lines = open(path).read().split('\n')
        lines = by_model_table(data, lines) if 'by-model' in path else metric_tables(data, lines)
        open(path, 'w').write('\n'.join(lines))
