"""Reconcile compact request evidence and report pooled empirical percentiles."""
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
case = sys.argv[2] if len(sys.argv) > 2 else 'c32-natural'
candidate = sys.argv[3] if len(sys.argv) > 3 else 'flashinfer'
report = {'qualification': False, 'percentile_method': 'nearest rank over pooled retained requests', 'lanes': {}}
for lane in ('baseline', candidate, 'vllm'):
    rows, summaries = [], []
    for pair in range(2):
        prefix = f'{case}-p{pair}-{lane}'
        part = json.loads(gzip.decompress((root / 'compact' / (prefix + '-retained-rows.json.gz')).read_bytes()))
        summary = json.loads((root / (prefix + '-summary.json')).read_text())
        assert summary['completed'] and summary['failed'] == 0
        assert len(part) == summary['requested']
        assert sum(len(r['token_ids']) for r in part) == summary['successful_output_tokens']
        assert sum(r['reference_match'] for r in part) == summary['reference_matches']
        for row in part:
            assert row['status'] == 'success' and row['transport_complete'] and row['protocol_valid']
            arrivals, start, metrics = row['token_arrival_ns'], row['started_ns'], row['metrics']
            assert metrics['e2e_ns'] == row['finished_ns'] - start
            assert metrics['token_ttft_ns'] == arrivals[0] - start
            assert math.isclose(metrics['token_tpot_ns'], (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1))
        rows.extend(part)
        summaries.append(summary)
    wall = sum(s['common_wall_ns'] for s in summaries) / 1e9
    def stats(metric):
        values = sorted(r['metrics'][metric] / 1e6 for r in rows)
        return {'p50': statistics.median(values), 'p95': values[math.ceil(.95 * len(values)) - 1],
                'p99': values[math.ceil(.99 * len(values)) - 1]}
    report['lanes'][lane] = {
        'requests': len(rows), 'failures': 0, 'reference_matches': sum(r['reference_match'] for r in rows),
        'output_tokens': sum(len(r['token_ids']) for r in rows),
        'tokens_per_second': sum(s['successful_output_tokens'] for s in summaries) / wall,
        'requests_per_second': len(rows) / wall,
        'run_tokens_per_second': [s['successful_output_tokens_per_wall_second'] for s in summaries],
        'ttft_ms': stats('token_ttft_ns'), 'tpot_ms': stats('token_tpot_ns'), 'e2e_ms': stats('e2e_ns'),
    }
(root / (f'comparison-{case}.json' if len(sys.argv)>2 else 'comparison.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
