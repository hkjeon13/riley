"""Apply the frozen G1-G5 gates to all retained qualification arms."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--partial', action='store_true')
args = parser.parse_args()
root = args.root
groups = [('qual-fixed-c1-c8-c16', 'fixed', [1, 8, 16]),
          ('qual-fixed-c32', 'fixed', [32]), ('qual-natural', 'natural', [8, 32])]
data = {}
for label, workload, concs in groups:
    path = root / label / 'result.json'
    if path.exists():
        data[label] = json.loads(path.read_text())
if args.partial:
    candidate = [r for d in data.values() for r in d['records'] if r['mode'] == 'native-after']
    print(json.dumps({
        'completed_arms': sum(len(d['records']) for d in data.values()),
        'planned_arms': 90,
        'groups': {label: len(data.get(label, {}).get('records', [])) for label, _, _ in groups},
        'candidate_arms': len(candidate),
        'candidate_canonical_failures': sum(r['measured_native_canonical_matches'] != 192 for r in candidate),
        'candidate_correctness_failures': sum(
            r['hf_correctness_tokens'] * 15 < 11 * r['correctness_requests']
            for r in candidate if r['correctness_requests']),
        'exit_codes': {label: (root / (label + '.exit')).read_text().strip()
                       for label, _, _ in groups if (root / (label + '.exit')).exists()},
    }))
    raise SystemExit(0)

manifest_path = root / 'qualification-preregistration.json'
manifest = json.loads(manifest_path.read_text())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
    root / 'qualification-preregistration.sha256').read_text().split()[0]
cells = []
canonicals = {}
cross_root_mismatches = []
g2 = g3 = True
all_normal_arms = 0
for label, workload, concs in groups:
    assert (root / (label + '.exit')).read_text().strip() == '0', label
    d = data[label]
    assert d['plan']['native_after_binary_sha256'] == manifest['candidate_sha256']
    assert d['plan']['runner_sha256'] == manifest['runner_sha256']
    assert d['plan']['repeats'] == 5
    assert len(d['records']) == len(concs) * 15
    assert json.loads((root / label / 'complete.json').read_text())['collection_complete']
    inv = json.loads((root / label / 'native-invariance.json').read_text())
    g2 &= not inv['mismatches']
    all_normal_arms += len(d['records'])
    for conc in concs:
        engines = {}
        for mode in ['native', 'native-after', 'vllm']:
            rows = sorted((r for r in d['records'] if r['concurrency'] == conc and r['mode'] == mode),
                          key=lambda r: r['repeat'])
            assert [r['repeat'] for r in rows] == list(range(5)), (label, conc, mode)
            assert all(r['requests'] == 192 for r in rows)
            engines[mode] = rows
        before, after, stock = [engines[k] for k in ['native', 'native-after', 'vllm']]
        paired = {metric: statistics.median((a[metric] / b[metric] - 1) * 100
                                            for a, b in zip(after, before))
                  for metric in ['output_tok_s', 'tpot_ms_p50', 'tpot_ms_p99',
                                 'ttft_ms_p95', 'ttft_ms_p99', 'itl_ms_p99']}
        medians = {mode: {metric: statistics.median(r[metric] for r in rows)
                          for metric in ['output_tok_s', 'tpot_ms_p50', 'tpot_ms_p99',
                                         'ttft_ms_p95', 'ttft_ms_p99', 'itl_ms_p99']}
                   for mode, rows in engines.items()}
        g2 &= all(r['measured_native_canonical_matches'] == 192 for r in after)
        correctness = [r for r in after if r['correctness_requests']]
        assert len(correctness) == 1
        g3 &= all(r['hf_correctness_tokens'] * 15 >= 11 * r['correctness_requests'] for r in correctness)
        canonical_file = root / label / f'c{conc}-r0-native-after' / 'correctness.json'
        for row in json.loads(canonical_file.read_text()):
            if row['id'] in canonicals and canonicals[row['id']] != row['value']:
                cross_root_mismatches.append({'group': label, 'concurrency': conc, 'id': row['id']})
            canonicals.setdefault(row['id'], row['value'])
        cells.append({
            'workload': workload, 'concurrency': conc, 'gpu': d['plan']['gpu'],
            'medians': medians, 'median_paired_change_percent': paired,
            'candidate_over_stock': medians['native-after']['output_tok_s'] / medians['vllm']['output_tok_s'],
            'G5': (paired['output_tok_s'] >= -1.3 and paired['tpot_ms_p50'] <= 1.5
                   and paired['tpot_ms_p99'] <= 10.0),
            'candidate_canonical_matches': [r['measured_native_canonical_matches'] for r in after],
            'hf_correctness': [[r['hf_correctness_tokens'], r['correctness_requests']] for r in correctness],
            'output_token_counts': {mode: [r['output_tokens'] for r in rows]
                                    for mode, rows in engines.items()},
        })
assert all_normal_arms == 90
g2 &= not cross_root_mismatches
g1 = all((root / 'build' / ('gpu-' + name + '-p2g-no-fma.exit')).read_text().strip() == '0'
         for name in ['fi', 'pd', 'fu', 'bench'])
primary = next(c for c in cells if (c['workload'], c['concurrency']) == ('natural', 32))
g4 = (primary['candidate_over_stock'] >= 1.0
      and primary['median_paired_change_percent']['output_tok_s'] >= 5.0)
gates = {'G1': g1, 'G2': bool(g2), 'G3': bool(g3), 'G4': g4, 'G5': all(c['G5'] for c in cells)}
result = {'preregistration_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'normal_arms': all_normal_arms, 'cells': cells, 'gates': gates,
          'accepted': all(gates.values()), 'cross_campaign_mismatches': cross_root_mismatches,
          'candidate_sha256': manifest['candidate_sha256']}
(root / 'qualification-analysis.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'gates': gates, 'accepted': result['accepted'], 'normal_arms': all_normal_arms}))
for cell in cells:
    print(cell['workload'], cell['concurrency'], 'candidate/vLLM', round(cell['candidate_over_stock'], 4),
          'paired', cell['median_paired_change_percent'], 'G5', cell['G5'])
