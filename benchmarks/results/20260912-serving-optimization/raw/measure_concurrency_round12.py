"""Run the seven prepared exploratory HTTP load settings, then restore Blender."""
import json
from pathlib import Path
import subprocess
import time

import run_serving_concurrency as load

ROOT = Path('/tmp/riley-opt-260912')
SESSION = ROOT / 'remote_session_round12.py'
OUTPUT = ROOT / 'concurrency-screen'
MATRIX = ROOT / 'concurrency-screen-plans/matrix.json'
EXPECTED = [(1, 128), *((c, budget) for c in (2, 4, 8) for budget in (128, 128*c))]


def run(argv):
    subprocess.run(argv, check=True)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert load.read(ROOT / 'decode7-operator-screen/completion.json')['completed'] is True
    matrix = load.read(MATRIX)
    assert matrix['schema_version'] == 'riley.concurrency-screen-matrix.v1'
    assert matrix['screening_only'] is True and matrix['tail_stability_claim'] is False
    assert matrix['requests_per_process'] == 256 and matrix['pairs_per_configuration'] == 1
    assert [item['label'] for item in matrix['plans']] == [f'c{c}-vllm-budget{b}' for c, b in EXPECTED]
    request_path, binding_path = Path(matrix['request']['path']), Path(matrix['binding']['path'])
    assert load.evidence(request_path) == matrix['request'] and load.evidence(binding_path) == matrix['binding']
    request, binding = load.read(request_path), load.read(binding_path)
    for item, (c, budget) in zip(matrix['plans'], EXPECTED):
        assert load.evidence(item['plan']['path']) == item['plan']
        plan = load.read(item['plan']['path'])
        assert plan['workload']['offered_concurrency'] == c and plan['workload']['pairs'] == 1
        assert plan['workload']['retained_requests_per_process'] == 256 and plan['workload']['purpose'] == 'screening'
        assert plan['http_lanes']['vllm']['active_capacity'] == c and plan['http_lanes']['vllm']['token_budget'] == budget
        load.validate_manifest(plan, request_path, binding_path, request, binding)
        for lane in plan['http_lanes'].values():
            load.shared.check_port(lane['port'])
    assert not OUTPUT.exists() and not (ROOT / 'blender-round12').exists()
    OUTPUT.mkdir()
    preps = OUTPUT / 'preparation-only'
    preps.mkdir()
    pins = {str(p): load.shared.digest(p) for p in (
        MATRIX, SESSION, ROOT / 'remote_session.py',
        ROOT / 'decode7-operator-screen/completion.json',
        Path(__file__), Path(load.__file__), Path(load.shared.__file__))}
    write(OUTPUT / 'screen-plan.json', {'matrix': load.evidence(MATRIX), 'inputs': pins,
          'settings': len(EXPECTED), 'requests_per_process': 256, 'pairs_per_setting': 1,
          'screening_only': True, 'tail_stability_claim': False})
    # These explicit preparations cannot query GPU or start a server.
    for item in matrix['plans']:
        run(['python3', str(ROOT / 'run_serving_concurrency.py'), '--plan', item['plan']['path'],
             '--request', str(request_path), '--binding', str(binding_path),
             '--output', str(preps / item['label']), '--prepare-only'])
    run(['python3', str(SESSION), 'check'])
    try:
        run(['python3', str(SESSION), 'stop'])
        stopped = load.read(ROOT / 'blender-round12/stopped.json')['pids']
        deadline = time.monotonic()+30
        while True:
            sample = load.shared.gpu_snapshot(binding)
            assert set(sample['compute_pids']) <= {str(pid) for pid in stopped}
            if not sample['compute_pids'] and sample['memory_used_mib'] <= 512:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('terminated Blender CUDA contexts did not drain')
            time.sleep(.2)
        for item in matrix['plans']:
            assert all(load.shared.digest(p) == digest for p, digest in pins.items())
            assert load.evidence(item['plan']['path']) == item['plan']
            run(['python3', str(SESSION), 'extend'])
            print(json.dumps({'starting': item['label'], 'screening_only': True}), flush=True)
            run(['python3', str(ROOT / 'run_serving_concurrency.py'), '--plan', item['plan']['path'],
                 '--request', str(request_path), '--binding', str(binding_path),
                 '--output', str(OUTPUT / item['label']), '--measure'])
            result = load.read(OUTPUT / item['label'] / 'summary.json')
            assert result['completed'] is True and result['workload']['purpose'] == 'screening'
    except BaseException as error:
        write(OUTPUT / 'failure.json', {'completed': False, 'error': str(error), 'type': type(error).__name__})
        raise
    finally:
        if (ROOT / 'blender-round12/session.json').exists():
            run(['python3', str(SESSION), 'restore'])
    restored = load.read(ROOT / 'blender-round12/verified.json')
    assert restored['alive_and_listening'] is True and restored['commands_and_gui_environment_match'] is True
    assert all(load.shared.digest(p) == digest for p, digest in pins.items())
    write(OUTPUT / 'completion.json', {'completed': True, 'settings': 7, 'processes': 14,
          'retained_requests': 3584, 'screening_only': True, 'tail_stability_claim': False,
          'restoration': load.evidence(ROOT / 'blender-round12/verified.json')})


if __name__ == '__main__':
    main()
