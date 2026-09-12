#!/usr/bin/env python3
"""Describe fully completed predeclared cells without accepting a partial campaign.

The frozen V2 analyzer runs unchanged first. A separate cleanup/restoration gate
then permits descriptive ratios only for a setting/comparison with all five
planned pairs and at least 1000 strict retained responses in every process.
Missing repeats are never selected, pooled, retried or imputed. This is offline
analysis, not numerical/performance acceptance or a new measurement campaign.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import analyze_serving_token_optimization_v2 as v2

SCHEMA = 'riley.serving-token-completed-cell-analysis.v3'
V2_SHA = '776c1dcb45c2ba676ea9e795df32d46aa043cef26c25069a55945f9f9b5d7a61'
CONTROLLER_SHA = '4b51034247b4301e390897788ffa3f7c0f1ce26bb313a9fc718afebc6524ee0e'
require = v2.require


def cleanup_envelope(campaign, analysis, evidence):
    """Validate complete cleanup, even for the lane that stopped the campaign.

This reads saved evidence only. It does not execute the remote session helper,
inspect current processes, or claim the previously restored processes live now.
    """
    require(analysis['status'] != 'invalid', 'a conflicting campaign completion claim cannot yield completed-cell statistics')
    prepared = evidence.json(campaign / 'preparation.json')
    require(prepared['plan'] == analysis['plan'], 'V2 preparation/plan changed')
    plan = evidence.json_ref(prepared['plan'])
    require(plan['workload'] == analysis['workload'], 'V2 workload changed')
    require((campaign / 'finalization.json').is_file(), 'full campaign finalization is required')
    final = evidence.json(campaign / 'finalization.json')
    restoration_ref = final['restoration']
    require(restoration_ref is not None, 'verified Blender restoration is required')
    restored = evidence.json_ref(restoration_ref)
    require(restored['alive_and_listening'] is True and restored['commands_and_gui_environment_match'] is True
            and restored['all_relaunched_processes_have_pinned_vendor_maps'] is True
            and restored['host_or_global_configuration_modified'] is False,
            'restoration is incomplete or changed host configuration')
    originals = {row['pid']: row for row in prepared['sessions']}
    rows = restored['processes']
    require(len(originals) == len(rows) == 3
            and {row['original_pid'] for row in rows} == set(originals)
            and len({row['new_pid'] for row in rows}) == 3
            and all(type(row['new_pid']) is int and row['new_pid'] > 0
                    and row['port'] == originals[row['original_pid']]['port'] for row in rows),
            'restored session identity/port scope differs')
    require(evidence.json_ref(restored['runtime_manifest']) == prepared['runtime'], 'restored private runtime differs')

    allowed, pair_paths, lane_paths = {}, [], []
    for setting in plan['settings']:
        for comparison in plan['comparisons']:
            for index in range(1, plan['workload']['pairs'] + 1):
                directory = campaign / f"{setting['id']}-{comparison['id']}-pair{index:02d}"
                pair_paths.append(directory / 'pair.json')
                order = [comparison['left'], comparison['right']]
                if index % 2 == 0:
                    order.reverse()
                for name in order:
                    allowed[directory / name] = (setting, name)
                    lane_paths.append(directory / name)
    launches = {path.parent for pattern in ('*/*/launch.json', '*/*/process.json') for path in campaign.glob(pattern)}
    exits = {path.parent for path in campaign.glob('*/*/process-exit.json')}
    require(launches and launches == exits and launches <= set(allowed),
            'every launched process, and only a declared lane, needs an exit receipt')
    require([path for path in lane_paths if path in launches] == lane_paths[:len(launches)],
            'launched lanes are not the predeclared execution prefix')
    process_ids, exit_refs = set(), []
    for directory in sorted(launches):
        setting, name = allowed[directory]
        launch = evidence.json(directory / 'launch.json')
        lane = plan['lanes'][name]
        expected_argv = v2.controller.lane_argv(lane, setting, Path(launch['argv'][0]).parent)
        require(launch['lane'] == name and launch['setting'] == setting and launch['fresh_process'] is True
                and launch['argv'] == expected_argv
                and launch['environment'] == v2.controller.lane_environment(plan, lane, prepared['runtime'])
                and launch['cwd'] == lane['cwd'] and launch['condition'] == v2.controller.CONDITION,
                'launched process identity differs from declared lane')
        process = evidence.json(directory / 'process.json')
        pid = process['pid']
        require(type(pid) is int and pid > 0 and pid == process['session_id'] and pid not in process_ids,
                'fresh process/session identity repeats or differs')
        process_ids.add(pid)
        record = evidence.json(directory / 'process-exit.json')
        cleanup = record['cleanup']
        require(cleanup and cleanup['cleanup_verified'] is True and cleanup['remaining_owned_pids'] == [],
                'owned process cleanup is incomplete')
        evidence.ref(cleanup['log'])
        exit_refs.append({'path': str(directory / 'process-exit.json'),
                          'sha256': v2.controller.shared.digest(directory / 'process-exit.json')})

    # Frozen controller V4 stops at its first unsuccessful pair. Completed pairs
    # must be the exact ordered prefix, not a selected subset of later repeats.
    entries = final['completed_pairs']
    require(len(entries) <= len(pair_paths), 'finalization has excess completed pairs')
    for index, entry in enumerate(entries):
        require(evidence.ref(entry['pair']) == pair_paths[index].resolve(),
                'finalization is not the predeclared completed-pair prefix')
    expected = []
    seen_incomplete = False
    for group in analysis['groups']:
        for pair in group['pairs']:
            if not pair['validated']:
                seen_incomplete = True
                continue
            require(not seen_incomplete, 'validated pairs occur after an incomplete predeclared pair')
            index = len(expected)
            require(index < len(entries) and entries[index]['setting'] == group['setting']['id']
                    and entries[index]['ratios'] == pair['ratios'], 'finalization completed pair differs from V2 raw validation')
            expected.append(entries[index])
    require(entries == expected, 'finalization claims a pair that V2 could not validate')
    return plan, {'validated': True, 'finalization': {'path': str(campaign / 'finalization.json'),
                    'sha256': v2.controller.shared.digest(campaign / 'finalization.json')},
                 'restoration': restoration_ref, 'owned_processes_with_verified_cleanup': len(process_ids),
                 'process_exit_receipts': exit_refs, 'completed_pair_prefix': entries,
                 'scope': 'saved finalization, all launched process exits, and verified restoration; no current live-process assertion'}


def completed_cell(group, workload, aggregate):
    require(workload['purpose'] == 'initial-measurement' and workload['pairs'] == 5
            and workload['retained_requests_per_process'] >= 1000,
            'completed-cell scope requires five predeclared initial-measurement pairs and >=1000 responses per process')
    require(group['planned_pairs'] == group['validated_pairs'] == 5
            and len(group['pairs']) == 5 and [pair['index'] for pair in group['pairs']] == [1,2,3,4,5],
            'cell does not contain all five prescribed validated pairs')
    comparison, retained = group['comparison'], workload['retained_requests_per_process']
    pids = set()
    for pair in group['pairs']:
        order = [comparison['left'], comparison['right']]
        if pair['index'] % 2 == 0:
            order.reverse()
        require(pair['validated'] is True and pair['errors'] == [] and pair['order'] == order
                and [lane['lane'] for lane in pair['lanes']] == order, 'cell pair order or validation differs')
        for lane in pair['lanes']:
            require(lane['status'] == 'validated' and lane['errors'] == []
                    and lane['retained_response_samples'] == retained
                    and all(phase['validated'] is True for phase in lane['phases']),
                    'cell contains incomplete raw, warmup, or retained validation')
            require(lane['process_id'] not in pids, 'cell reused a server process')
            pids.add(lane['process_id'])
    return {'setting': group['setting'], 'comparison': comparison, 'planned_pairs': 5, 'validated_pairs': 5,
            'descriptive_cell_complete': True, 'pair_indices': [1,2,3,4,5],
            'left_lane': comparison['left'], 'right_lane': comparison['right'], 'ratio_direction': 'right / left',
            'retained_requests_per_process': retained, 'retained_requests_per_lane': 5*retained,
            'fresh_server_processes': len(pids), 'paired_ratios': aggregate(group['pairs']),
            'aggregation_scope': 'distribution of five per-pair right/left ratios; no pooled-request percentile or missing-repeat imputation',
            'selection_scope': 'all repeats of this exact predeclared setting/comparison; no selection by observed outcome',
            'numerical_acceptance_changed': False, 'performance_acceptance': False, 'performance_claim': False,
            'p99_stability_qualified': False, 'high_concurrency_qualified': False}


def analyze(campaign, mappings=()):
    require(v2.controller.shared.digest(v2.__file__) == V2_SHA
            and v2.controller.shared.digest(v2.controller.__file__) == CONTROLLER_SHA,
            'frozen V2 analyzer/controller changed')
    campaign = Path(campaign).resolve(strict=True)
    analysis = v2.analyze(campaign, mappings)  # Unchanged raw/provenance/qualifier validation.
    core, _ = v2.private_core()  # Existing read-only Evidence and aggregate functions; no replacement hooks.
    evidence = core.Evidence(mappings)
    for item in analysis['inputs']:
        evidence.capture(item['local_path'], item['original_path'], item['sha256'])
    for module in (sys.modules[__name__], v2):
        evidence.capture(module.__file__)
    errors, cells, excluded = [], [], []
    gate, plan = {'validated': False}, None
    try:
        plan, gate = cleanup_envelope(campaign, analysis, evidence)
        require(analysis['measurement_client_qualification']
                and analysis['measurement_client_qualification']['validated'] is True,
                'measurement client qualification was not validated')
    except Exception as error:
        gate['validated'] = False
        errors.append({'scope': 'completed-cell-gate', 'type': type(error).__name__, 'message': str(error)})
    for group in analysis['groups']:
        try:
            require(gate['validated'], 'full cleanup/restoration/qualification gate unavailable')
            cells.append(completed_cell(group, plan['workload'], core.aggregate_pairs))
        except Exception as error:
            excluded.append({'setting': group['setting'], 'comparison': group['comparison'],
                             'planned_pairs': group['planned_pairs'], 'validated_pairs': group['validated_pairs'],
                             'reason': str(error)})
    return {'schema_version': SCHEMA, 'analysis_kind': 'completed-cell-descriptive-only',
            'status': 'descriptive-cells-available' if cells else 'no-completed-cells',
            'whole_campaign_completed': analysis['validated_complete'],
            'overall_acceptance': False, 'performance_acceptance': False, 'performance_claim': False, 'winner': None,
            'completed_cells': cells, 'excluded_cells': excluded, 'cleanup_gate': gate, 'errors': errors,
            'campaign_analysis_v2': analysis, 'inputs': evidence.finish(),
            'whole_campaign_eligibility_unchanged': True,
            'scope': 'complete predeclared cells only, after all owned process cleanup and verified restoration; failed/missing cells remain excluded',
            'metrics_scope': analysis['metrics_scope'], 'zero_itl_scope': analysis['zero_itl_scope'],
            'no_missing_comparison_imputation': True, 'p99_stability_qualified': False, 'high_concurrency_qualified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    mappings = [item.split('=',1) for item in args.path_map]
    require(all(len(item) == 2 for item in mappings), 'path maps use ORIGINAL=LOCAL')
    result = analyze(args.campaign,mappings)
    if args.output:
        v2.controller.write(args.output,result)
    else:
        print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['cleanup_gate']['validated'] else 2


if __name__ == '__main__':
    sys.exit(main())
