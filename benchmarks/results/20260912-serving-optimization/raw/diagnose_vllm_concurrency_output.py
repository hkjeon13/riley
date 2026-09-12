"""Capture C2 output divergence with Blender restored; never produce timing claims."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import subprocess
import threading

import run_serving_concurrency as load

ROOT = Path('/tmp/riley-opt-260912')
PLAN = ROOT / 'concurrency-screen-plans/c2-vllm-budget128.json'
OUTPUT = ROOT / 'vllm-c2-output-diagnostic-runtime'
RUNTIME = ROOT / 'driver580173-runtime-20260901'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not OUTPUT.exists()
    matrix = load.read(ROOT / 'concurrency-screen-plans/matrix.json')
    request_path, binding_path = Path(matrix['request']['path']), Path(matrix['binding']['path'])
    request, binding, plan = load.read(request_path), load.read(binding_path), load.read(PLAN)
    parent = load.validate_manifest(plan, request_path, binding_path, request, binding)
    restored = load.read(ROOT / 'blender-round12/verified.json')
    assert restored['alive_and_listening'] and restored['commands_and_gui_environment_match']
    assert load.read(ROOT / 'concurrency-screen/c2-vllm-budget128/completion.json')['completed'] is False
    lane = plan['http_lanes']['vllm']
    runtime = load.read(RUNTIME / 'receipt.json')
    assert runtime['completed'] and runtime['archive_signature_verified'] and not runtime['host_packages_modified']
    assert '580.173.02' in Path('/proc/driver/nvidia/version').read_text()
    for relative, digest in runtime['files'].items():
        assert load.shared.digest(RUNTIME / 'extracted' / relative) == digest
    env = {**plan['base_environment'], **lane['env']}
    env['LD_LIBRARY_PATH'] = ':'.join(filter(None, (runtime['library_path'], env.get('LD_LIBRARY_PATH'))))
    env['PATH'] = str(Path(runtime['nvidia_smi_path']).parent) + ':' + env['PATH']
    # Scope NVML queries and the diagnostic server to the extracted old libraries.
    os.environ['LD_LIBRARY_PATH'], os.environ['PATH'] = env['LD_LIBRARY_PATH'], env['PATH']
    load.shared.check_port(lane['port'])
    OUTPUT.mkdir()
    pins = {str(p): load.shared.digest(p) for p in (
        PLAN, RUNTIME / 'receipt.json', Path(__file__), Path(load.__file__), Path(load.shared.__file__))}
    write(OUTPUT / 'preparation.json', {'diagnostic_only': True, 'performance_claim': False,
          'reason': 'Capture rejected C2 outputs without changing the retained benchmark correctness gate.',
          'source_commit': parent['source_commit'], 'inputs': pins,
          'blender_restoration': load.evidence(ROOT / 'blender-round12/verified.json'),
          'argv': lane['argv'], 'environment': env,
          'runtime_override': load.evidence(RUNTIME / 'receipt.json'),
          'reference_text': lane['expected_output_text'],
          'reference_token_ids': binding['generated_token_ids'],
          'gpu_before': load.shared.gpu_snapshot(binding)})
    log_path = OUTPUT / 'server.log'
    results = []
    with log_path.open('x') as log:
        process = subprocess.Popen(lane['argv'], cwd=parent['source_root'],
            env=env, stdout=log, stderr=log, start_new_session=True)
        lock = threading.Lock()
        cleaned = False
        def cleanup():
            nonlocal cleaned
            with lock:
                if not cleaned:
                    load.shared.stop_owned_process(process)
                    cleaned = True
        try:
            load.shared.wait_ready(process, lane['port'], plan['startup_timeout_seconds'])
            startup = OUTPUT / 'vllm-startup.log'
            startup.write_text(log_path.read_text())
            write(OUTPUT / 'vllm-startup.json', load.validate_vllm_startup(startup, lane, plan['vllm_runtime']))
            mappings = {}
            for item in Path('/proc').iterdir():
                if item.name.isdigit():
                    try:
                        if os.getpgid(int(item.name)) == process.pid:
                            lines = (item / 'maps').read_text().splitlines()
                            mappings[item.name] = [line for line in lines if 'libcuda.so' in line or 'libnvidia-ml.so' in line]
                    except (ProcessLookupError, FileNotFoundError, PermissionError):
                        pass
            write(OUTPUT / 'owned-driver-mappings.json', mappings)
            # Same unmodified request first. Token-ID collection is separately labelled.
            for phase, concurrency, stream, ids in (
                    ('c2-plain-first', 2, False, False), ('c2-plain-repeat', 2, False, False),
                    ('c1-plain-capacity2', 1, False, False), ('c2-token-ids', 2, False, True),
                    ('c1-token-ids-capacity2', 1, False, True), ('c2-stream-token-ids', 2, True, True)):
                barrier = threading.Barrier(concurrency)
                def request_one(index):
                    body = {'model': lane.get('model_id', parent.get('http_model_id', 'g04-smol')),
                            'prompt': request['prompt'], 'max_tokens': request['requested_output_tokens'],
                            'temperature': 0, 'top_p': 1, 'stream': stream}
                    if ids:
                        body['return_token_ids'] = True
                    connection = http.client.HTTPConnection('127.0.0.1', lane['port'], timeout=120)
                    barrier.wait(timeout=30)
                    timer = threading.Timer(120, cleanup)
                    timer.daemon = True
                    timer.start()
                    try:
                        connection.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
                        response = connection.getresponse()
                        raw = response.read().decode('utf-8')
                        path = OUTPUT / f'{phase}-{index}.json'
                        record = {'phase': phase, 'offered_concurrency': concurrency,
                                  'request': body, 'status': response.status, 'raw_response': raw}
                        if not stream and response.status == 200:
                            payload = json.loads(raw)
                            choice = payload['choices'][0]
                            record.update(text_exact=choice['text'] == lane['expected_output_text'],
                                finish_reason=choice['finish_reason'], usage=payload.get('usage'),
                                token_ids_exact=(choice.get('token_ids') == binding['generated_token_ids']) if ids else None)
                        write(path, record)
                        return {'path': str(path), 'status': response.status,
                                'text_exact': record.get('text_exact'), 'token_ids_exact': record.get('token_ids_exact')}
                    finally:
                        timer.cancel()
                        connection.close()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results.extend(pool.map(request_one, range(concurrency)))
                assert process.poll() is None
            assert all(load.shared.digest(p) == digest for p, digest in pins.items())
        except BaseException as error:
            write(OUTPUT / 'failure.json', {'completed': False, 'error': str(error), 'type': type(error).__name__})
            raise
        finally:
            cleanup()
            write(OUTPUT / 'process-exit.json', {'pid': process.pid, 'returncode': process.returncode,
                  'owned_session_cleanup_finished': True, 'log': load.evidence(log_path)})
    write(OUTPUT / 'completion.json', {'completed': True, 'diagnostic_only': True, 'performance_claim': False,
          'requests': len(results), 'responses': results, 'gpu_after': load.shared.gpu_snapshot(binding)})
    print(json.dumps({'completed': True, 'diagnostic_only': True, 'requests': len(results)}))


if __name__ == '__main__':
    main()
