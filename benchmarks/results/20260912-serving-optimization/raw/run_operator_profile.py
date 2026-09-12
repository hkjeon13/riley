"""Screen operation costs in separate fresh captures, then restore Blender.

Diagnostic events are excluded from serving results. Keep all raw warmups;
derive separate logs containing only the last three P128/O32 requests.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import run_serving_optimization as shared

ROOT = Path('/tmp/riley-opt-260912')
TOOL = ROOT / 'operator-tools/profile_decode_operators.py'
SESSION = ROOT / 'remote_session_round4.py'
REQUEST_PATH = Path('/tmp/riley-g04-vllm-profile-260911/request.json')


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify(build):
    assert shared.digest(build['binary']) == build['binary_sha256']
    assert shared.digest(TOOL) == build['tool_sha256']
    assert shared.digest(TOOL.with_name('profile_owned_graph.py')) == build['historical_tool_sha256']
    assert shared.digest(Path(build['source_root']) / 'kernels/src/graph_resources.cu') == build['instrumented_source_sha256']


def case(directory, build, plan, binding, request, family, projection='off', layers='0,15,29'):
    directory.mkdir()
    verify(build)
    shared.validate_artifacts(plan, REQUEST_PATH, ROOT / 'batch4-binding.json', request, binding)
    shared.preflight(plan, binding, directory)
    lane = plan['http_lanes']['riley']
    shared.check_port(lane['port'])
    argv = list(lane['argv'])
    argv[0] = build['binary']
    env = dict(os.environ, **lane['env'])
    env.update(RILEY_OWNED_GRAPH_PROFILE='1', RILEY_DECODE_OPERATOR=family,
               RILEY_DECODE_OPERATOR_LAYERS=layers, RILEY_OWNED_GRAPH_PROJECTION=projection)
    log_path = directory / 'raw.log'
    with (directory / 'telemetry.csv').open('x') as telemetry:
        probe = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used', '--format=csv', '--loop-ms=200'], stdout=telemetry, stderr=subprocess.STDOUT)
        try:
            with log_path.open('x') as log:
                process = subprocess.Popen(argv, cwd=build['source_root'], env=env, stdout=log, stderr=log)
                try:
                    import http.client
                    deadline = time.monotonic() + 180
                    while True:
                        assert process.poll() is None, 'diagnostic server exited'
                        connection = http.client.HTTPConnection('127.0.0.1', lane['port'], timeout=2)
                        try:
                            connection.request('GET', '/v1/models')
                            response = connection.getresponse()
                            response.read()
                            if response.status == 200:
                                break
                        except OSError:
                            pass
                        finally:
                            connection.close()
                        if time.monotonic() > deadline:
                            raise TimeoutError('diagnostic server startup')
                        time.sleep(.1)
                    requests = []
                    for index in range(6):
                        row = shared.http_request(lane['port'], 'g04-smol', request, lane['expected_output_text'], index != 0)
                        requests.append({'index': index, 'warmup': index < 3, 'validated_by_shared_http_runner': True})
                    process.terminate()
                    process.wait(timeout=30)
                    assert process.returncode == 0
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
        finally:
            probe.terminate()
            probe.wait(timeout=10)
    # Replay IDs 1..96 belong to the three warmup requests. Keep inventories
    # and capture receipts needed to bind the remaining operation records.
    retained = directory / 'retained.log'
    with retained.open('x') as output:
        for line in log_path.read_text().splitlines(keepends=True):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                output.write(line)
                continue
            if (isinstance(record, dict) and str(record.get('schema', '')).startswith('riley.')
                    and isinstance(record.get('replay_id'), int) and record['replay_id'] <= 96):
                continue
            output.write(line)
    verify(build)
    receipt = {'family': family, 'projection': projection, 'layers': layers, 'requests': requests,
               'warmup_requests': 3, 'retained_requests': 3, 'retained_replay_id_min': 97,
               'binary_sha256': build['binary_sha256'], 'tool_sha256': build['tool_sha256'],
               'base_source_commit': build['base_source_commit'], 'argv': argv,
               'raw_log_sha256': shared.digest(log_path), 'retained_log_sha256': shared.digest(retained),
               'performance_claim_eligible': False, 'server_exit_code': process.returncode}
    write(directory / 'validation.json', receipt)
    return retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'operator-screen')
    args = parser.parse_args()
    build = json.loads((ROOT / 'operator-profile-build.json').read_text())
    plan = json.loads((ROOT / 'batch4-http-plan.json').read_text())
    binding = json.loads((ROOT / 'batch4-binding.json').read_text())
    # Keep the original immutable request path; no relabelled request artifact.
    request = json.loads(REQUEST_PATH.read_text())
    verify(build)
    shared.validate_artifacts(plan, REQUEST_PATH, ROOT / 'batch4-binding.json', request, binding)
    shared.check_port(plan['http_lanes']['riley']['port'])
    args.output.mkdir()
    modes = [('off-before', 'off', 'off')]
    modes.extend((family, family, 'off') for family in ['qkv', 'rope_kv', 'attention', 'o', 'post_norm', 'gate_up', 'swiglu', 'down', 'next_norm', 'head'])
    modes.extend(('prefill-' + projection, 'off', projection) for projection in ['q', 'gate', 'down'])
    modes.append(('off-after', 'off', 'off'))
    try:
        run(['python3', str(SESSION), 'check'])
        run(['python3', str(SESSION), 'stop'])
        stopped = json.loads((ROOT / 'blender-round4/stopped.json').read_text())['pids']
        deadline = time.monotonic() + 30
        while True:
            sample = shared.gpu_snapshot(binding)
            assert set(sample['compute_pids']) <= {str(pid) for pid in stopped}
            if not sample['compute_pids'] and sample['memory_used_mib'] <= 512:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('terminated GPU contexts did not drain')
            time.sleep(.2)
        baseline = None
        for label, family, projection in modes:
            run(['python3', str(SESSION), 'extend'])
            print(json.dumps({'starting': label, 'diagnostic': True}), flush=True)
            retained = case(args.output / label, build, plan, binding, request, family, projection)
            if baseline is None:
                baseline = retained
            command = ['python3', str(TOOL), 'summarize', '--log', str(retained)]
            if retained != baseline:
                command += ['--baseline-log', str(baseline)]
            with (retained.parent / 'summary.json').open('x') as stream:
                run(command, stdout=stream)
            report = json.loads((retained.parent / 'summary.json').read_text())
            assert report['whole_graph']['failed_replays'] == 0
            assert report['whole_graph']['groups']['prefill']['replays'] == 3
            assert report['whole_graph']['groups']['decode']['replays'] == 93
            for group in report['whole_graph']['groups'].values():
                assert group['metrics']['cuda_graph_span_ns']['unmeasured_count'] == 0
            assert [g['operator'] for g in report['operator_groups']] == ([] if family == 'off' else [family])
            for group in report['operator_groups']:
                assert group['sum_across_selected_layers_ns']['unmeasured_count'] == 0
                assert group['replays'] == 93
                assert [row['layer'] for row in group['layers']] == ([None] if family == 'head' else [0, 15, 29])
            projections = report['whole_graph']['projection_groups']
            if projection == 'off':
                assert not projections
            else:
                prefill = [g for g in projections if g['phase'] == 'prefill']
                assert len(prefill) == 1 and prefill[0]['projection'] == projection
                assert prefill[0]['operator'] == 'override'
                assert prefill[0]['records'] == prefill[0]['measured_count'] == 3
                assert prefill[0]['unmeasured_or_failed_count'] == 0
                decode = [g for g in projections if g['phase'] == 'decode']
                if projection == 'down':
                    assert len(decode) == 1 and decode[0]['projection'] == 'down'
                    assert decode[0]['operator'] == 'canonical'
                    assert decode[0]['records'] == decode[0]['measured_count'] == 93
                    assert decode[0]['unmeasured_or_failed_count'] == 0
                else:
                    assert not decode
            print(json.dumps({'completed': label}), flush=True)
        write(args.output / 'completion.json', {'completed': True, 'cases': len(modes), 'performance_claim_eligible': False})
    finally:
        if (ROOT / 'blender-round4/session.json').exists():
            run(['python3', str(SESSION), 'restore'])


if __name__ == '__main__':
    main()
