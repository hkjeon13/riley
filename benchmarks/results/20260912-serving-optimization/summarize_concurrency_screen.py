"""Recompute the completed seven-setting screen from locally mirrored raw rows."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'scripts'))
import run_serving_concurrency as load


def read(path):
    return json.loads(path.read_text())


def evidence(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    raw = ROOT / 'raw'
    matrix_path = raw / 'concurrency-screen-plans/matrix.json'
    matrix = read(matrix_path)
    screen = raw / 'concurrency-screen'
    completed = read(screen / 'completion.json')
    assert completed['completed'] and completed['settings'] == 7
    assert completed['retained_requests'] == 3584
    assert completed['screening_only'] and not completed['tail_stability_claim']
    results = []
    for item in matrix['plans']:
        label = item['label']
        plan_path = raw / 'concurrency-screen-plans' / (label + '.json')
        assert evidence(plan_path)['sha256'] == item['plan']['sha256']
        plan = read(plan_path)
        directory = screen / label
        summary_path = directory / 'summary.json'
        summary = read(summary_path)
        assert summary['completed'] and len(summary['pair_results']) == 1
        assert summary['workload'] == plan['workload']
        assert summary['inputs'][item['plan']['path']] == item['plan']['sha256']
        pair = summary['pair_results'][0]
        result = {'label': label, 'workload': plan['workload'],
                  'riley_capacity': plan['http_lanes']['riley']['active_capacity'],
                  'vllm_capacity': plan['http_lanes']['vllm']['active_capacity'],
                  'vllm_token_budget': plan['http_lanes']['vllm']['token_budget'],
                  'summary': evidence(summary_path), 'plan': evidence(plan_path)}
        for role in ('riley', 'vllm'):
            lane = directory / ('pair-01-' + role)
            rows_path = lane / 'retained.jsonl'
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            accounting = read(lane / 'retained-accounting.json')
            assert len(rows) == plan['workload']['retained_requests_per_process'] == 256
            recalculated = load.summarize(rows, accounting, 32)
            assert recalculated == pair[role], (label, role, 'raw/summary mismatch')
            assert load.overlap_peak(rows) == plan['workload']['offered_concurrency']
            assert read(lane / 'execution-complete.json')['completed']
            assert read(lane / 'process-exit.json')['owned_session_cleanup_finished']
            result[role] = {**recalculated, 'retained_rows': evidence(rows_path)}
        result['riley_over_vllm_wall_throughput'] = (
            result['riley']['output_tokens_per_wall_second'] /
            result['vllm']['output_tokens_per_wall_second'])
        assert result['riley_over_vllm_wall_throughput'] == pair['riley_over_vllm_wall_throughput']
        results.append(result)
    output = {'schema_version': 'riley.concurrency-screen-comparison.v1',
              'completed': True, 'screening_only': True, 'high_concurrency_stability_claim': False,
              'matrix': evidence(matrix_path), 'completion': evidence(screen / 'completion.json'),
              'analyzer': evidence(Path(__file__).resolve()), 'settings': results,
              'limitations': ['One process pair and 256 retained requests per lane and setting.',
                              'Fixed repeated P128/O32 prompt; closed-loop offered load only.',
                              'HTTP first-text is not token TTFT; HTTP TPOT is unmeasured.',
                              'Riley active capacity remains one; vLLM active capacity equals offered concurrency.',
                              'Token count is checked through exact reference text and available usage; raw HTTP token IDs are unavailable.',
                              'Per-process nearest-rank tails are descriptive screening evidence.']}
    (ROOT / 'concurrency-screen-comparison.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print('| C | vLLM budget | Riley tok/s | vLLM tok/s | R/V | Riley E2E ms | vLLM E2E ms | Riley first-text ms | vLLM first-text ms |')
    print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in results:
        r, v = row['riley'], row['vllm']
        print(f"| {row['workload']['offered_concurrency']} | {row['vllm_token_budget']} | "
              f"{r['output_tokens_per_wall_second']:.3f} | {v['output_tokens_per_wall_second']:.3f} | "
              f"{row['riley_over_vllm_wall_throughput']:.4f} | "
              f"{r['e2e_ms']['median']:.3f} | {v['e2e_ms']['median']:.3f} | "
              f"{r['first_text_event_ms']['median']:.3f} | {v['first_text_event_ms']['median']:.3f} |")


if __name__ == '__main__':
    main()
