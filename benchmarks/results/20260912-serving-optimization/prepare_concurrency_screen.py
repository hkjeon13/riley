"""Freeze seven exploratory queued-c1 HTTP load configurations; never launch GPU work."""
import copy
import json
import os
from pathlib import Path

import run_serving_concurrency as load

ROOT = Path('/tmp/riley-opt-260912')
PARENT = ROOT / 'batch7-http-plan.json'
EXPECTED_SOURCE = '1a2be0df01fe49daa4d4db155ad5c44f34ead6df'
MATRIX = [(1, 128), *((c, budget) for c in (2, 4, 8) for budget in (128, 128*c))]


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def replace(argv, flag, value):
    assert argv.count(flag) <= 1
    if flag not in argv:
        argv.extend([flag, str(value)])
    else:
        argv[argv.index(flag)+1] = str(value)


def main():
    parent = load.read(PARENT)
    assert parent['source_commit'] == EXPECTED_SOURCE
    for mode in ('http', 'engine'):
        result = load.read(ROOT / ('batch7-'+mode) / 'summary.json')
        assert result['completed'] is True and result['source_commit'] == EXPECTED_SOURCE
    request_path = Path(parent['request_path'])
    binding_path = Path(parent['binding_path'])
    request, binding = load.read(request_path), load.read(binding_path)
    output = ROOT / 'concurrency-screen-plans'
    assert not output.exists()
    base_env = {key: os.environ[key] for key in ('HOME', 'PATH', 'LANG', 'LC_ALL', 'LC_CTYPE') if key in os.environ}
    assert base_env.get('HOME') and base_env.get('PATH')
    immutable = {str(Path(path).resolve()): load.shared.digest(path) for path in
                 (PARENT, Path(load.__file__), Path(load.shared.__file__), Path(__file__))}
    prepared = []
    for concurrency, budget in MATRIX:
        label = f'c{concurrency}-vllm-budget{budget}'
        lanes = copy.deepcopy(parent['http_lanes'])
        lanes['riley'].update(active_capacity=1, waiting_capacity=64, http_workers=8, token_budget=128)
        replace(lanes['riley']['argv'], '--max-waiting-requests', 64)
        lanes['vllm'].update(active_capacity=concurrency, token_budget=budget)
        replace(lanes['vllm']['argv'], '--max-num-seqs', concurrency)
        replace(lanes['vllm']['argv'], '--max-num-batched-tokens', budget)
        plan = {
            'schema_version': load.PLAN_SCHEMA,
            'parent_c1_plan': load.evidence(PARENT), 'immutable_files': immutable,
            'workload': {'id': f'queued-c1-{label}-screen-v1', 'offered_concurrency': concurrency,
                         'arrival_policy': 'closed-loop-refill', 'retained_requests_per_process': 256,
                         'warmups_per_worker_per_transport': 5, 'pairs': 1, 'purpose': 'screening'},
            'base_environment': base_env, 'http_lanes': lanes,
            'vllm_runtime': {'engine_version': '0.27.1', 'enforce_eager': False,
                             'enable_chunked_prefill': True, 'compilation_mode': 'VLLM_COMPILE',
                             'cudagraph_mode': 'FULL_AND_PIECEWISE'},
            'startup_timeout_seconds': 180, 'request_timeout_seconds': 120,
        }
        load.validate_manifest(plan, request_path, binding_path, request, binding)
        prepared.append((label, plan))
    output.mkdir()
    for label, plan in prepared:
        write(output / (label+'.json'), plan)
    write(output / 'matrix.json', {
        'schema_version': 'riley.concurrency-screen-matrix.v1',
        'source_commit': EXPECTED_SOURCE,
        'plans': [{'label': label, 'plan': load.evidence(output / (label+'.json'))} for label, _ in prepared],
        'request': load.evidence(request_path), 'binding': load.evidence(binding_path),
        'screening_only': True, 'requests_per_process': 256, 'pairs_per_configuration': 1,
        'tail_stability_claim': False, 'measurement_started': False,
        'purpose': 'Explore queued-c1 throughput and vLLM active/token-budget settings; no statistical winner or P99 stability claim.',
        'next_step': 'Repeat viable settings with at least 1000 requests per process and fresh paired processes before a concurrency baseline decision.',
    })
    print(json.dumps({'prepared': len(prepared), 'directory': str(output), 'measurement_started': False}))


if __name__ == '__main__':
    main()
