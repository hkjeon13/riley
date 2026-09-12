"""Run twelve fresh batch7 diagnostic captures with two off baselines.

Only an explicit script run pauses the verified round10 Blender successors.
Each case validates six P128/O32 HTTP requests and retains the last three.
Captured timing events are diagnostics, never serving performance evidence.
"""
import json
import math
import time

import run_operator_profile_batch7 as base

ROOT = base.ROOT
SESSION = ROOT / 'remote_session_round11.py'
SESSION_ROOT = ROOT / 'blender-round11'
OUTPUT = ROOT / 'decode7-operator-screen'
CASES = (('off-before', 'off'), *((family, family) for family in base.FAMILIES), ('off-after', 'off'))


def complete_metric(metric, count):
    assert metric['measured_count'] == count and metric.get('unmeasured_count', 0) == 0
    assert type(metric['median']) in (int, float) and math.isfinite(metric['median']) and metric['median'] >= 0


def validate_report(report, family, requests, log):
    assert report['schema_version'] == 'riley.decode-operator-summary.v1'
    assert report['performance_claim_eligible'] is False and report['log_sha256'] == base.shared.digest(log)
    whole = report['whole_graph']
    assert whole['failed_replays'] == 0 and not whole['projection_groups']
    for phase, expected in (('prefill', requests), ('decode', requests * 31)):
        group = whole['groups'][phase]
        assert group['replays'] == group['successful_replays'] == expected
        complete_metric(group['metrics']['cuda_graph_span_ns'], expected)
    captures = report['captures']
    assert len(captures) == 2 and {row['phase'] for row in captures} == {'prefill', 'decode'}
    assert len({row['capture_id'] for row in captures}) == 2
    count = 0 if family == 'off' else 1 if family == 'head' else 30
    mask = 0 if family == 'off' else 1 << 30 if family == 'head' else (1 << 30) - 1
    inventory = {row['capture_id']: row for row in report['external_event_inventories']}
    assert len(report['external_event_inventories']) == len(inventory) == 2
    for capture in captures:
        assert capture['operator'] == family and capture['layer_mask'] == mask
        assert capture['interval_count'] == count and capture['projection_events_compiled'] is False
        assert capture['source_sha256'] == base.builder.GRAPH_SHA
        assert capture['tool_sha256'] == base.builder.TOOL_SHA
        assert capture['historical_tool_sha256'] == base.builder.HISTORICAL_SHA
        edges = 2 * count if capture['phase'] == 'decode' else 0
        assert capture['expected_event_records'] == capture['recorded_event_records'] == edges
        row = inventory[capture['capture_id']]
        assert row['inventory_status'] == 0 and row['event_record_nodes'] == edges and row['event_wait_nodes'] == 0
    groups = report['operator_groups']
    assert [group['operator'] for group in groups] == ([] if family == 'off' else [family])
    if family == 'off':
        assert not report['replay_totals']
    for group in groups:
        assert group['replays'] == requests * 31
        complete_metric(group['sum_across_selected_layers_ns'], requests * 31)
        assert [row['layer'] for row in group['layers']] == ([None] if family == 'head' else list(range(30)))
        for row in group['layers']:
            complete_metric(row['cuda_span_ns'], requests * 31)
        assert [row['position'] for row in group['positions']] == list(range(128, 159))
        for row in group['positions']:
            complete_metric(row['cuda_sum_ns'], requests)
        assert len(report['replay_totals']) == requests * 31
    if 'whole_graph_perturbation' in report:
        bracket = report['whole_graph_perturbation']
        assert bracket['performance_claim_eligible'] is False
        assert [row['phase'] for row in bracket['groups']] == ['prefill', 'decode']
        for row in bracket['groups']:
            assert all(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] > 0
                       for key in ('off_median_ns', 'selected_median_ns', 'ratio'))


def summarize(log, output, family, requests=3, baseline=None):
    command = ['python3', str(base.TOOL), 'summarize', '--log', str(log)]
    if baseline is not None:
        command += ['--baseline-log', str(baseline)]
    with output.open('x') as stream:
        base.run(command, stdout=stream)
    report = json.loads(output.read_text())
    validate_report(report, family, requests, log)
    if baseline is not None:
        assert report['whole_graph_perturbation']['baseline_log_sha256'] == base.shared.digest(baseline)
    return report


def main():
    base.builder.campaign_gate()
    build = json.loads((ROOT / 'decode7-profile-build.json').read_text())
    plan = json.loads((ROOT / 'batch7-http-plan.json').read_text())
    binding = json.loads((ROOT / 'batch7-binding.json').read_text())
    request = json.loads(base.REQUEST_PATH.read_text())
    base.verify(build)
    assert plan['source_commit'] == base.builder.BASE_COMMIT
    base.shared.validate_artifacts(plan, base.REQUEST_PATH, ROOT / 'batch7-binding.json', request, binding)
    base.shared.check_port(plan['http_lanes']['riley']['port'])
    assert len(CASES) == 12 and not SESSION_ROOT.exists(), 'round11 must be a new session'
    OUTPUT.mkdir()
    paths = [ROOT / 'decode7-profile-build.json', ROOT / 'batch7-http-plan.json', ROOT / 'batch7-binding.json',
             base.REQUEST_PATH, SESSION, ROOT / 'remote_session.py', base.TOOL,
             base.TOOL.with_name('profile_owned_graph.py'), base.__file__, base.shared.__file__,
             base.builder.__file__, __file__]
    pins = {str(path): base.shared.digest(path) for path in paths}
    base.write(OUTPUT / 'preparation.json', {
        'source_commit': base.builder.BASE_COMMIT, 'base_build_sha256': base.builder.BASE_BUILD_SHA,
        'inputs': pins, 'cases': list(CASES), 'requests_per_case': 6, 'retained_requests_per_case': 3,
        'all_layers': True, 'head_layer': None, 'historical_projection_events': False,
        'performance_claim_eligible': False,
    })
    reports = {}
    try:
        base.run(['python3', str(SESSION), 'check'])
        base.run(['python3', str(SESSION), 'stop'])
        stopped = json.loads((SESSION_ROOT / 'stopped.json').read_text())['pids']
        assert len(stopped) == len(set(stopped)) == 3
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
        for label, family in CASES:
            assert all(base.shared.digest(path) == digest for path, digest in pins.items()), 'diagnostic inputs changed'
            base.run(['python3', str(SESSION), 'extend'])
            print(json.dumps({'starting': label, 'diagnostic': True}), flush=True)
            retained = base.case(OUTPUT / label, build, plan, binding, request, family, layers='all')
            summarize(retained.parent / 'raw.log', retained.parent / 'raw-summary.json', family, requests=6)
            reports[label] = summarize(retained, retained.parent / 'summary.json', family, baseline=baseline)
            if baseline is None:
                baseline = retained
            print(json.dumps({'completed': label}), flush=True)
        after = OUTPUT / 'off-after/retained.log'
        bracketed = []
        for family in base.FAMILIES:
            base.run(['python3', str(SESSION), 'extend'])
            directory = OUTPUT / family
            report = summarize(directory / 'retained.log', directory / 'summary-vs-off-after.json', family, baseline=after)
            bracketed.append({
                'operator': family,
                'sum_across_selected_layers_ns': report['operator_groups'][0]['sum_across_selected_layers_ns'],
                'whole_graph_vs_off_before': reports[family]['whole_graph_perturbation'],
                'whole_graph_vs_off_after': report['whole_graph_perturbation'],
                'summary_before_sha256': base.shared.digest(directory / 'summary.json'),
                'summary_after_sha256': base.shared.digest(directory / 'summary-vs-off-after.json'),
            })
        assert all(base.shared.digest(path) == digest for path, digest in pins.items()), 'diagnostic inputs changed'
        base.verify(build)
        base.shared.validate_artifacts(plan, base.REQUEST_PATH, ROOT / 'batch7-binding.json', request, binding)
        base.write(OUTPUT / 'summary.json', {
            'diagnostic_cases_completed': 12, 'operators': bracketed,
            'source_commit': base.builder.BASE_COMMIT, 'binary_sha256': build['binary_sha256'],
            'performance_claim_eligible': False,
            'interpretation': 'Per-family intervals include event perturbation. Families come from separate processes; do not add their medians as a whole-graph time.',
        })
    except BaseException as error:
        base.write(OUTPUT / 'failure.json', {'completed': False, 'error': str(error), 'type': type(error).__name__})
        raise
    finally:
        if (SESSION_ROOT / 'session.json').exists():
            base.run(['python3', str(SESSION), 'restore'])
    restored = json.loads((SESSION_ROOT / 'verified.json').read_text())
    assert restored['alive_and_listening'] is True and restored['commands_and_gui_environment_match'] is True
    assert len(restored['processes']) == 3
    base.write(OUTPUT / 'completion.json', {
        'completed': True, 'cases': 12, 'requests': 72, 'retained_requests': 36,
        'restoration_verified_sha256': base.shared.digest(SESSION_ROOT / 'verified.json'),
        'performance_claim_eligible': False,
    })


if __name__ == '__main__':
    main()
