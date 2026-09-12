"""Verify the screened attention bottleneck over all thirty layers."""
import json
import time

import run_operator_profile as base

ROOT = base.ROOT
SESSION = ROOT / 'remote_session_round5.py'
OUTPUT = ROOT / 'attention-full-profile'


def main():
    assert json.loads((ROOT / 'operator-screen/completion.json').read_text())['completed']
    build = json.loads((ROOT / 'operator-profile-build.json').read_text())
    plan = json.loads((ROOT / 'batch4-http-plan.json').read_text())
    binding = json.loads((ROOT / 'batch4-binding.json').read_text())
    request = json.loads(base.REQUEST_PATH.read_text())
    base.verify(build)
    base.shared.validate_artifacts(plan, base.REQUEST_PATH, ROOT / 'batch4-binding.json', request, binding)
    base.shared.check_port(plan['http_lanes']['riley']['port'])
    OUTPUT.mkdir()
    try:
        base.run(['python3', str(SESSION), 'check'])
        base.run(['python3', str(SESSION), 'stop'])
        stopped = json.loads((ROOT / 'blender-round5/stopped.json').read_text())['pids']
        deadline = time.monotonic() + 30
        while True:
            sample = base.shared.gpu_snapshot(binding)
            assert set(sample['compute_pids']) <= {str(pid) for pid in stopped}
            if not sample['compute_pids'] and sample['memory_used_mib'] <= 512:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('terminated GPU contexts did not drain')
            time.sleep(.2)
        baseline = None
        for label, family in [('off-before', 'off'), ('attention-all', 'attention'), ('off-after', 'off')]:
            base.run(['python3', str(SESSION), 'extend'])
            print(json.dumps({'starting': label, 'diagnostic': True}), flush=True)
            retained = base.case(OUTPUT / label, build, plan, binding, request, family, layers='all')
            if baseline is None:
                baseline = retained
            command = ['python3', str(base.TOOL), 'summarize', '--log', str(retained)]
            if baseline != retained:
                command += ['--baseline-log', str(baseline)]
            with (retained.parent / 'summary.json').open('x') as stream:
                base.run(command, stdout=stream)
            report = json.loads((retained.parent / 'summary.json').read_text())
            assert report['whole_graph']['failed_replays'] == 0
            for phase, expected in [('prefill', 3), ('decode', 93)]:
                group = report['whole_graph']['groups'][phase]
                assert group['replays'] == group['successful_replays'] == expected
                assert group['metrics']['cuda_graph_span_ns']['measured_count'] == expected
            assert not report['whole_graph']['projection_groups']
            groups = report['operator_groups']
            assert [g['operator'] for g in groups] == ([] if family == 'off' else ['attention'])
            for group in groups:
                assert group['replays'] == 93
                assert group['sum_across_selected_layers_ns']['unmeasured_count'] == 0
                assert [row['layer'] for row in group['layers']] == list(range(30))
            print(json.dumps({'completed': label}), flush=True)
        base.write(OUTPUT / 'completion.json', {'completed': True, 'cases': 3, 'performance_claim_eligible': False})
    finally:
        if (ROOT / 'blender-round5/session.json').exists():
            base.run(['python3', str(SESSION), 'restore'])


if __name__ == '__main__':
    main()
