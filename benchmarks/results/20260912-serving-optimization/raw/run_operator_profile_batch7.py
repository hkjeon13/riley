"""Screen operation costs in separate fresh captures, then restore Blender.

Diagnostic events are excluded from serving results. Keep all raw warmups;
derive separate logs containing only the last three P128/O32 requests.
"""
import json
import os
from pathlib import Path
import subprocess
import time

import run_serving_optimization as shared
import build_decode_profile_batch7 as builder

ROOT = Path('/tmp/riley-opt-260912')
TOOL = ROOT / 'decode7-operator-tools/profile_decode_operators_batch7.py'
SESSION = ROOT / 'remote_session_round11.py'
REQUEST_PATH = Path('/tmp/riley-g04-vllm-profile-260911/request.json')
FAMILIES = ('qkv', 'rope_kv', 'attention', 'o', 'post_norm', 'gate_up', 'swiglu', 'down', 'next_norm', 'head')
WHOLE_SCHEMA = 'riley.owned-graph-diagnostic.v1'


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify(build):
    base = json.loads((ROOT / 'batch7-build.json').read_text())
    builder.verify_base(base)
    builder.verify_instrumented(base)
    assert build['base_source_commit'] == base['source_commit'] == builder.BASE_COMMIT
    assert build['base_build_sha256'] == shared.digest(ROOT / 'batch7-build.json') == builder.BASE_BUILD_SHA
    assert build['binary'] == str(ROOT / 'decode7-profile-target/release/riley')
    assert build['source_root'] == str(ROOT / 'decode7-profile-source')
    assert shared.digest(build['binary']) == build['binary_sha256']
    assert shared.digest(TOOL) == build['tool_sha256'] == builder.TOOL_SHA
    assert shared.digest(TOOL.with_name('profile_owned_graph.py')) == build['historical_tool_sha256'] == builder.HISTORICAL_SHA
    assert shared.digest(Path(build['source_root']) / builder.GRAPH) == build['instrumented_source_sha256'] == builder.INSTRUMENTED_SHA
    assert shared.digest(ROOT / 'decode7-profile-build.log') == build['build_log_sha256']
    assert shared.digest(builder.__file__) == build['builder_sha256']
    instrument = build['instrumentation']
    assert instrument == json.loads((ROOT / 'decode7-profile-instrumentation.json').read_text())
    assert instrument['original_sha256'] == builder.GRAPH_SHA
    assert instrument['instrumented_sha256'] == builder.INSTRUMENTED_SHA
    assert instrument['tool_sha256'] == builder.TOOL_SHA and instrument['historical_tool_sha256'] == builder.HISTORICAL_SHA
    assert instrument['applied'] is True and instrument['projection_events'] is False
    assert instrument['synchronizations_added'] == 0 and instrument['performance_claim_eligible'] is False


def replay_rows(path, first, last):
    rows = []
    for line in path.read_text().splitlines():
        if WHOLE_SCHEMA not in line:
            continue
        row = json.loads(line)
        assert row['schema'] == WHOLE_SCHEMA
        assert row['kind'] != 'projection', 'historical projection hooks are forbidden'
        if row['kind'] == 'replay':
            rows.append(row)
    assert [row['replay_id'] for row in rows] == list(range(first, last + 1)), 'missing, duplicate, or reordered replay'
    for row in rows:
        assert type(row['replay_id']) is int and row['position'] == 127 + (row['replay_id'] - 1) % 32
        assert row['launch_status'] == row['completion_status'] == row['event_status'] == 0
    return rows


def case(directory, build, plan, binding, request, family, layers='all'):
    assert family in ('off', *FAMILIES) and layers == 'all'
    assert request['prompt_tokens'] == 128 and request['requested_output_tokens'] == 32
    directory.mkdir()
    verify(build)
    shared.validate_artifacts(plan, REQUEST_PATH, ROOT / 'batch7-binding.json', request, binding)
    shared.preflight(plan, binding, directory)
    lane = plan['http_lanes']['riley']
    shared.check_port(lane['port'])
    argv = list(lane['argv'])
    argv[0] = build['binary']
    env = dict(os.environ, **lane['env'])
    env.update(RILEY_OWNED_GRAPH_PROFILE='1', RILEY_DECODE_OPERATOR=family,
               RILEY_DECODE_OPERATOR_LAYERS=layers, RILEY_OWNED_GRAPH_PROJECTION='off')
    write(directory / 'launch.json', {
        'argv': argv, 'cwd': build['source_root'], 'fresh_process': True,
        'diagnostic_environment': {key: env[key] for key in ('RILEY_OWNED_GRAPH_PROFILE', 'RILEY_DECODE_OPERATOR',
                                  'RILEY_DECODE_OPERATOR_LAYERS', 'RILEY_OWNED_GRAPH_PROJECTION')},
        'binary_sha256': build['binary_sha256'], 'performance_claim_eligible': False,
    })
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
                    with (directory / 'http-requests.jsonl').open('x') as request_log:
                        for index in range(6):
                            model = lane.get('model_id', plan.get('http_model_id', 'g04-smol'))
                            row = shared.http_request(lane['port'], model, request, lane['expected_output_text'], index != 0)
                            row.update(index=index, warmup=index < 3, streaming=index != 0,
                                       validated_by_shared_http_runner=True)
                            requests.append(row)
                            request_log.write(json.dumps(row) + '\n')
                            request_log.flush()
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
            try:
                probe.wait(timeout=10)
            except subprocess.TimeoutExpired:
                probe.kill()
                probe.wait(timeout=10)
    # Replay IDs 1..96 belong to the three warmup requests. Keep inventories
    # and capture receipts needed to bind the remaining operation records.
    retained = directory / 'retained.log'
    replay_rows(log_path, 1, 192)
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
    shared.validate_artifacts(plan, REQUEST_PATH, ROOT / 'batch7-binding.json', request, binding)
    replay_rows(retained, 97, 192)
    receipt = {'family': family, 'projection': 'off', 'layers': layers, 'requests': requests,
               'warmup_requests': 3, 'retained_requests': 3, 'retained_replay_id_min': 97,
               'binary_sha256': build['binary_sha256'], 'tool_sha256': build['tool_sha256'],
               'base_source_commit': build['base_source_commit'], 'argv': argv,
               'raw_log_sha256': shared.digest(log_path), 'retained_log_sha256': shared.digest(retained),
               'http_requests_sha256': shared.digest(directory / 'http-requests.jsonl'),
               'case_runner_sha256': shared.digest(__file__), 'shared_runner_sha256': shared.digest(shared.__file__),
               'performance_claim_eligible': False, 'server_exit_code': process.returncode}
    write(directory / 'validation.json', receipt)
    return retained
