#!/usr/bin/env python3
"""V2 HTTP token correctness with the actual three-token stop prefix; no timing claim.

Only running main launches a private Riley server/GPU workload. It never stops
Blender or changes the host runtime. --validate-only rechecks saved evidence.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import time
import subprocess

import run_http_token_tests as shared
import check_gui_driver_runtime as runtime_check

SCHEMA = 'riley.http-token-observation-correctness.v1'
EXPECTED_COMMIT = 'a179617070526068b66ba5627ba82a7151da8c64'
EXPECTED_BINARY = '18cbd5f8a8ad8583ecfcd2d38815a484831c05eb9f081b1db67be6194ffebd8c'
CLIENT_SHA = 'a2a4a35569d6b892542097c119e2be8a6402beb60ab1565aa95658794c364766'
SAMPLERS = ('cpu', 'gpu-greedy')
CONCURRENCIES = (1, 2, 4, 8)
CANDIDATE = 'riley-0.0.0-rc13'
PROOFS = ('published_ids_match_reference', 'published_ids_match_committed_audit',
          'default_and_raw_text_usage_match', 'input_ids_exact', 'blank_commits_counted',
          'disconnect_reuse_exact', 'c02_audit_complete', 'owned_process_cleanup_verified')
MAX_BYTES = 8 * 1024 * 1024
STOP_TEXT = ", I'm"
STOP_TOKEN_IDS = (28, 339, 5248)
STOP_PREFIXES = (",", ", I", STOP_TEXT)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON: '+value))


def evidence(path):
    return {'path': str(Path(path).resolve(strict=True)), 'sha256': shared.sha(path)}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def write_bytes(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_client(path):
    spec = importlib.util.spec_from_file_location('optional_token_observation_client', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def process_start(pid):
    return int((Path('/proc')/str(pid)/'stat').read_text().rsplit(') ', 1)[1].split()[19])


def owned_members(process):
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry/'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) == process.pid and fields[0] != 'Z':
                result.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return sorted(result)


def stop_owned(process):
    if process is None:
        return
    # Only this fresh start_new_session child and members of its held session.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    require(not owned_members(process), 'owned Riley process group remains after cleanup')


def runtime_and_sessions(env, binding):
    compute = runtime_check.check_runtime(runtime_check.COMPUTE)
    runtime_check.check_runtime(runtime_check.GUI)
    require(Path('/proc/driver/nvidia/version').read_text() == compute['kernel_version'], 'kernel/runtime version differs')
    query = subprocess.check_output([compute['nvidia_smi_path'], '-i', '0', '--query-gpu=uuid,driver_version',
                                     '--format=csv,noheader'], env=env, text=True, timeout=15).strip()
    require(query == binding['environment']['gpu']['uuid']+', 580.173.02', 'GPU UUID/driver differs')
    return compute, runtime_check.check_sessions(), query


def driver_maps(process, compute):
    require(process.poll() is None, 'owned server exited before runtime mapping check')
    prefix = Path(compute['library_path'])
    selected = []
    for line in (Path('/proc')/str(process.pid)/'maps').read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not Path(parts[5]).name.startswith(('libcuda.so', 'libnvidia-')):
            continue
        path = Path(parts[5])
        require(not parts[5].endswith(' (deleted)') and path.is_relative_to(prefix), 'foreign/deleted NVIDIA mapping')
        expected = compute['files'].get(str(path.relative_to(runtime_check.COMPUTE/'extracted')))
        require(expected is not None and shared.sha(path) == expected, 'mapped NVIDIA file changed')
        info = path.stat()
        major, minor = map(lambda value: int(value, 16), parts[3].split(':'))
        require(info.st_ino == int(parts[4]) and (os.major(info.st_dev), os.minor(info.st_dev)) == (major, minor), 'mapped NVIDIA inode changed')
        selected.append(line)
    require(any('libcuda.so.580.173.02' in line for line in selected), 'private libcuda is not actually mapped')
    return {'pid': process.pid, 'start_ticks': process_start(process.pid), 'mappings': selected, 'all_mapped_vendor_files_pinned': True}


def wire_request(client_module, port, body, timeout, *, disconnect=None, method='POST', route='/v1/completions'):
    """Bounded default-wire/readiness/disconnect transport; exact body retained."""
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    lock, expired = threading.Lock(), threading.Event()
    held = {'socket': None}
    started = time.perf_counter_ns()
    chunks, frames, cleanup_errors = [], [], []
    response = None
    encoded = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    row = {'started_ns': started, 'http_status': None, 'headers': [], 'disconnect': disconnect,
           'disconnect_observed': False, 'complete': False, 'error': None}
    framer = client_module.SSEFramer(MAX_BYTES)
    streaming = bool(body and body.get('stream'))
    def abort():
        expired.set()
        with lock:
            sock = held['socket']
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    timer = threading.Timer(timeout, abort)
    timer.daemon = True
    timer.start()
    try:
        connection.connect()
        with lock:
            held['socket'] = connection.sock
        require(not expired.is_set(), 'whole request deadline expired while connecting')
        connection.request(method, route, encoded,
                           {'Content-Type': 'application/json', 'Connection': 'close'})
        response = connection.getresponse()
        row.update(http_status=response.status, headers=response.getheaders())
        require(not expired.is_set(), 'whole request deadline expired in headers')
        if disconnect == 'after_headers':
            require(response.status == 200, 'disconnect request was not admitted')
            row['disconnect_observed'] = True
        else:
            size = 0
            while True:
                chunk = response.read1(65536)
                arrived = time.perf_counter_ns()
                require(not expired.is_set() and arrived-started <= timeout*1e9, 'whole response deadline expired')
                if not chunk:
                    if streaming:
                        framer.finish()
                    row['complete'] = True
                    break
                chunks.append(chunk)
                size += len(chunk)
                require(size <= MAX_BYTES, 'response exceeds byte bound')
                if streaming:
                    for payload, timestamp in framer.feed(chunk, arrived):
                        text = payload.decode('utf-8')
                        frames.append({'data': text, 'arrived_ns': timestamp})
                        if disconnect == 'after_first_committed' and text != '[DONE]':
                            parsed = client_module.parse_json(text)
                            if any(choice.get('token_ids') for choice in parsed.get('choices', [])):
                                row['disconnect_observed'] = True
                    if row['disconnect_observed']:
                        break
            if disconnect:
                require(row['disconnect_observed'], 'disconnect boundary was not reached')
    except Exception as error:
        row['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        timer.cancel()
        timer.join()
        if disconnect or expired.is_set():
            with lock:
                sock = held['socket']
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        for resource in (response, connection):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    cleanup_errors.append({'type': type(error).__name__, 'message': str(error)})
        if cleanup_errors and row['error'] is None:
            row['error'] = {'type': 'CleanupError', 'message': 'owned HTTP resource cleanup failed'}
    row.update(finished_ns=time.perf_counter_ns(), deadline_fired=expired.is_set(),
               total_deadline_seconds=timeout, owned_connection_closed=not cleanup_errors, cleanup_errors=cleanup_errors,
               request=body, request_body_sha256=hashlib.sha256(encoded or b'').hexdigest(), frames=frames,
               raw_response_base64=base64.b64encode(b''.join(chunks)).decode())
    return row


def default_observation(module, row, reference, streaming):
    require(row['error'] is None and not row['deadline_fired'] and row['complete'] and row['http_status'] == 200, 'default response transport failed')
    raw = base64.b64decode(row['raw_response_base64'])
    values, done = [], 0
    if streaming:
        for frame in row['frames']:
            require(done == 0, 'default SSE follows DONE')
            if frame['data'] == '[DONE]': done += 1
            else: values.append(module.parse_json(frame['data']))
        require(done == 1, 'default SSE DONE count differs')
    else:
        values = [module.parse_json(raw)]
    require(values, 'default response contains no JSON')
    text, terminal, identity, usage = '', None, None, None
    for value in values:
        require('error' not in value and value.get('model') == reference.model and value.get('object') == 'text_completion', 'default response identity/error differs')
        current = {key: value[key] for key in ('id', 'model', 'object', 'created')}
        require(identity is None or identity == current, 'default SSE metadata changes')
        identity = current
        require(terminal is None and len(value.get('choices', [])) == 1, 'default response choice/terminal order differs')
        choice = value['choices'][0]
        require(choice.get('index') == 0 and 'token_ids' not in choice and 'prompt_token_ids' not in choice, 'default wire exposes opt-in token fields')
        text += choice['text']
        terminal = choice.get('finish_reason')
        if streaming:
            require('usage' not in value, 'default SSE exposes opt-in usage')
        else:
            usage = value['usage']
    require(text == reference.text and terminal == reference.finish_reason, 'default text/finish differs from reference')
    expected_usage = {'prompt_tokens': len(reference.prompt_token_ids), 'completion_tokens': len(reference.output_token_ids),
                      'total_tokens': len(reference.prompt_token_ids)+len(reference.output_token_ids)}
    require(streaming or usage == expected_usage, 'default nonstream usage differs')
    return {'response_identity': identity, 'text': text, 'finish_reason': terminal, 'usage': usage,
            'token_fields_absent': True, 'text_and_available_usage_exact': True}


def validate_audit(audit_dir, request_id, reference, streaming, sampler, identity, runtime_identity):
    require(re.fullmatch(r'cmpl-[A-Za-z0-9_-]{1,123}', request_id), 'unsafe source request ID')
    path = audit_dir/(request_id+'.json')
    marker = audit_dir/(path.name+'.complete')
    raw, complete = read(path), read(marker)
    require(complete == {'schema_version': 'riley.c02-generation-audit-completion.v2',
                         'artifact_filename': path.name, 'artifact_sha256': shared.sha(path)}, 'C02 audit completion marker differs')
    require(raw['schema_version'] == 'riley.c02-generation-audit.v2' and raw['candidate_id'] == CANDIDATE
            and raw['server_request_id'] == request_id and raw['runtime_identity'] == runtime_identity
            and raw['process_identity'] == identity and raw['delivery_mode'] == ('stream' if streaming else 'non-stream'), 'C02 audit identity differs')
    tokens = raw['committed_output_tokens']
    require(raw['prompt_token_ids'] == list(reference.prompt_token_ids)
            and [token['token_id'] for token in tokens] == list(reference.output_token_ids)
            and ''.join(token['emitted_text_delta'] for token in tokens) == reference.text
            and raw['finish_reason'] == reference.finish_reason, 'source committed audit differs from reference')
    require(raw['usage'] == {'prompt_tokens': len(reference.prompt_token_ids), 'completion_tokens': len(tokens),
                             'total_tokens': len(reference.prompt_token_ids)+len(tokens)}, 'C02 committed counts differ')
    selections = raw['sampling_selections']
    selected = 'cpu-normative' if sampler == 'cpu' else 'gpu-greedy'
    require(len(selections) == len(tokens) and all(item['committed'] is True and type(item['iteration_id']) is int
            and item['iteration_id'] > 0 and item['configured_backend'] == selected and item['selected_backend'] == selected
            for item in selections), 'audit sampler selection differs')
    require(all(item['ineligibility_reason'] is None for item in selections) if sampler == 'gpu-greedy'
            else all(item['ineligibility_reason'] == 'gpu-greedy-not-configured' for item in selections), 'sampling ineligibility attribution differs')
    return raw, {'artifact': evidence(path), 'completion_marker': evidence(marker)}


def validate_published_audit(module, client_row, audit, streaming):
    require(client_row['status'] == 'success' and client_row['protocol_valid'] and client_row['reference_match'], 'raw token client failed exact reference')
    expected = audit['committed_output_tokens']
    require(client_row['token_ids'] == [token['token_id'] for token in expected]
            and client_row['prompt_token_ids'] == audit['prompt_token_ids'] and client_row['usage'] == audit['usage'], 'published IDs/counts differ from committed audit')
    if streaming:
        generated = []
        for frame in client_row['frames']:
            if frame['data'] != '[DONE]':
                value = module.parse_json(frame['data'])
                for choice in value.get('choices', []):
                    if choice.get('token_ids'):
                        require(len(choice['token_ids']) == 1, 'ambiguous generated token frame')
                        generated.append({'token_id': choice['token_ids'][0], 'emitted_text_delta': choice['text']})
        require(generated == expected, 'published per-commit text/IDs differ from source audit')
    return sum(token['emitted_text_delta'] == '' for token in expected)


def wait_ready(module, process, port):
    deadline = time.monotonic()+180
    while process.poll() is None and time.monotonic() < deadline:
        row = wire_request(module, port, None, 2, method='GET', route='/v1/models')
        if row['http_status'] == 200 and row['error'] is None:
            value = module.parse_json(base64.b64decode(row['raw_response_base64']))
            if any(item.get('id') == 'g04-smol' for item in value.get('data', [])):
                return
        time.sleep(.1)
    raise RuntimeError('owned HTTP token server did not become ready')


def stop_reference(tokenizer, generated):
    """Validate the real frozen prefix before any output directory or server launch."""
    require(tuple(generated[:3]) == STOP_TOKEN_IDS, 'fixed stop prefix token IDs differ')
    prefixes = tuple(tokenizer.decode(generated[:count], skip_special_tokens=True) for count in range(1, 4))
    require(prefixes == STOP_PREFIXES, 'actual tokenizer stop prefixes differ')
    require(all(STOP_TEXT not in value for value in prefixes[:-1]) and prefixes[-1] == STOP_TEXT,
            'stop must first complete on the third committed token with no preceding text')
    return 3, ''


def check_specs():
    return [{'shape': shape, 'streaming': streaming, 'raw': raw}
            for shape in ('o32', 'o1', 'stop') for streaming, raw in
            ((False, False), (False, True), (True, False), (True, True))]


def payload(prompt, spec):
    result = {'model': 'g04-smol', 'prompt': prompt, 'temperature': 0, 'top_p': 1,
              'max_tokens': 1 if spec['shape'] == 'o1' else 32, 'stream': spec['streaming']}
    if spec['shape'] == 'stop': result['stop'] = STOP_TEXT
    if spec['raw']:
        result['return_token_ids'] = True
        if spec['streaming']: result['stream_options'] = {'include_usage': True}
    return result


def execute_check(module, client, port, directory, check_id, spec, prompt, references,
                  audit_dir, sampler, identity, runtime_identity, timeout):
    reference = references[spec['shape']]
    body = payload(prompt, spec)
    request_path, result_path = directory/(check_id+'-request.json'), directory/(check_id+'-response.json')
    write(request_path, {'payload': body, 'expected_reference': {'model': reference.model,
          'prompt_token_ids': list(reference.prompt_token_ids), 'output_token_ids': list(reference.output_token_ids),
          'text': reference.text, 'finish_reason': reference.finish_reason}, 'reference_sha256': reference.sha256})
    try:
        if spec['raw']:
            row = client.request(port, body, reference, streaming=spec['streaming'], mode='strict', timeout_seconds=timeout)
            write(result_path, row)
            require(row['request'] == body and row['request_body_sha256'] ==
                    hashlib.sha256(json.dumps(body, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                    'token client transmitted a different request')
            require(row['status'] == 'success' and row['protocol_valid'] and row['reference_match'], 'raw token response failed')
            observed = row
        else:
            row = wire_request(module, port, body, timeout)
            write(result_path, row)
            observed = default_observation(module, row, reference, spec['streaming'])
        request_id = observed['response_identity']['id']
        audit, audit_evidence = validate_audit(audit_dir, request_id, reference, spec['streaming'], sampler, identity, runtime_identity)
        blanks = validate_published_audit(module, row, audit, spec['streaming']) if spec['raw'] else 0
        require(observed['text'] == ''.join(token['emitted_text_delta'] for token in audit['committed_output_tokens']), 'default/raw text differs from audit')
        if spec['raw'] and spec['shape'] == 'o1':
            require(row['metrics']['token_tpot_ns'] is None, 'O1 token TPOT must be undefined')
        if spec['raw'] and spec['streaming'] and spec['shape'] == 'stop':
            require(blanks > 0, 'stop case did not exercise blank committed events')
        result = {'check_id': check_id, 'kind': 'completion', 'completed': True, 'sampler': sampler,
                  'spec': spec, 'server_request_id': request_id, 'request': evidence(request_path),
                  'response': evidence(result_path), 'audit': audit_evidence, 'blank_committed_tokens': blanks,
                  'started_ns': row['started_ns'], 'finished_ns': row.get('call_finished_ns', row['finished_ns']),
                  'published_ids_match_committed_audit': spec['raw'], 'performance_claim': False}
        write(directory/(check_id+'-check.json'), result)
        return result
    except BaseException as error:
        write(directory/(check_id+'-failure.json'), {'completed': False, 'type': type(error).__name__, 'error': str(error),
                                                    'request': evidence(request_path), 'performance_claim': False})
        raise


def mixed_wave(module, run_one, directory, concurrency):
    barrier = threading.Barrier(concurrency+1)
    specs, results, failures = check_specs(), [], []
    lock = threading.Lock()
    def worker(index):
        barrier.wait(timeout=30)
        for step in range(2):
            # Independent requests share the same canonical prompt. Output
            # limits, stream mode, opt-in flags and stop policy vary together.
            spec = specs[(index*3+step*5) % len(specs)]
            result = run_one(f'mixed-c{concurrency}-w{index}-r{step}', spec)
            with lock: results.append(result)
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='token-correctness-wave')
    futures = []
    try:
        futures = [pool.submit(worker, index) for index in range(concurrency)]
        barrier.wait(timeout=30)
        for future in futures:
            try: future.result()
            except BaseException as error: failures.append({'type': type(error).__name__, 'error': str(error)})
    finally:
        barrier.abort()
        pool.shutdown(wait=True, cancel_futures=True)
    peak = module.overlap_peak(results) if results else 0
    result = {'concurrency': concurrency, 'active_capacity': 1, 'requested': concurrency*2,
              'completed': len(results) == concurrency*2 and peak == concurrency and not failures,
              'observed_max_request_in_flight': peak, 'failures': failures,
              'checks': sorted(item['check_id'] for item in results), 'performance_claim': False}
    write(directory/f'mixed-c{concurrency}-accounting.json', result)
    require(result['completed'], 'mixed concurrent correctness wave incomplete or did not overlap')
    return sorted(results, key=lambda item: item['check_id']), result


def shutdown_proof(audit_dir, identity):
    path = audit_dir/'shutdown.json'
    marker = audit_dir/'shutdown.json.complete'
    record, complete = read(path), read(marker)
    require(complete == {'schema_version': 'riley.c02-shutdown-quiescence-complete.v2',
                         'artifact_filename': path.name, 'artifact_sha256': shared.sha(path)}, 'shutdown completion marker differs')
    require(record['schema_version'] == 'riley.c02-shutdown-quiescence.v2' and record['capture_status'] == 'captured'
            and record['qualification_status'] == 'not-run' and record['worker_ready'] is False
            and record['server_pid'] == identity['pid'] and record['server_start_ticks'] == identity['start_ticks'], 'shutdown process/capture identity differs')
    metrics = record['final_metrics']
    require(metrics['schema_version'] == 'riley.c02-capture-metrics.v2'
            and metrics['request_states']['active'] == metrics['request_states']['pending_requests'] == 0
            and metrics['kv_blocks']['reserved'] == metrics['kv_blocks']['active'] == 0
            and set(metrics['allocation']) == {'device_live_count', 'device_live_bytes', 'pinned_live_count', 'pinned_live_bytes'}
            and all(type(value) is int and value == 0 for value in metrics['allocation'].values())
            and metrics['quiescence'] == {'completion_outbox': 0, 'outstanding_iterations': 0,
                 'riley_owned_live_allocations': 0, 'worker_accepting': False, 'scheduler_accepting': False}, 'shutdown native resources are not quiescent')
    return {'artifact': evidence(path), 'completion_marker': evidence(marker), 'native_resources_quiescent': True}


def run_sampler(module, root, output, sampler, env, compute, prompt, references, verify, timeout):
    verify()
    directory = output/sampler
    directory.mkdir(mode=0o700)
    audits = directory/'audit'
    audits.mkdir(mode=0o700)
    profile = 'stable-default' if sampler == 'cpu' else 'max-performance-exact'
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        port = reserved.getsockname()[1]
    binary = root/'http-token-target/release/riley'
    argv = [str(binary), 'serve', '--model', str(shared.MODEL), '--model-id', 'g04-smol', '--bind', f'127.0.0.1:{port}',
            '--max-active-sequences', '1', '--max-waiting-requests', '16', '--batch-token-budget', '128',
            '--prefill-chunk-tokens', '128', '--max-sequence-tokens', '160', '--max-output-tokens', '32', '--kv-blocks', '10',
            '--residual-rmsnorm', 'separate', '--execution-completion', 'iteration-batch', '--metadata-transport', 'packed-async',
            '--execution-graph-policy', 'require', '--graph-numerics', 'vllm-smol-p128-v1', '--sampling-backend', sampler,
            '--c02-candidate-id', CANDIDATE, '--c02-configuration-profile', profile,
            '--c02-startup-artifact', str(directory/'startup.json'), '--c02-audit-dir', str(audits),
            '--c02-shutdown-artifact', str(audits/'shutdown.json')]
    runtime_identity = {'configuration_profile': profile, 'configuration_sha256': canonical_sha({'argv': argv[1:], 'environment': env})}
    write(directory/'preparation.json', {'argv': argv, 'cwd': str(root/'http-token-source'),
          'environment_sha256': canonical_sha(env), 'selected_environment': {key: env[key] for key in
          ('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CARGO_TARGET_DIR')},
          'runtime_identity': runtime_identity, 'performance_claim': False})
    process = None
    checks, waves, disconnects = [], [], []
    log_path = directory/'server.log'
    try:
        with log_path.open('x') as log:
            process = subprocess.Popen(argv, cwd=root/'http-token-source', env=env, stdout=log, stderr=log, start_new_session=True)
            identity = {'pid': process.pid, 'start_ticks': process_start(process.pid)}
            write(directory/'launch.json', identity)
            wait_ready(module, process, port)
            startup = read(directory/'startup.json')
            require(startup['schema_version'] == 'riley.effective-runtime-config-startup-artifact.v1'
                    and startup['candidate_id'] == CANDIDATE and startup['runtime_identity'] == runtime_identity
                    and startup['endpoint_path'] == '/v1/config'
                    and startup['endpoint_payload_sha256'] == canonical_sha(startup['endpoint_payload']), 'source startup contract differs')
            endpoint_row = wire_request(module, port, None, min(timeout, 5), method='GET', route='/v1/config')
            write(directory/'runtime-endpoint-response.json', endpoint_row)
            require(endpoint_row['complete'] and endpoint_row['http_status'] == 200 and endpoint_row['error'] is None
                    and module.parse_json(base64.b64decode(endpoint_row['raw_response_base64'])) == startup['endpoint_payload'], 'live runtime configuration differs from startup')
            write(directory/'driver-maps-before.json', driver_maps(process, compute))
            with module.TokenHttpClient() as client:
                def run_one(check_id, spec):
                    require(process.poll() is None, 'owned server exited before request')
                    return execute_check(module, client, port, directory, check_id, spec, prompt, references,
                                         audits, sampler, identity, runtime_identity, timeout)
                for index, spec in enumerate(check_specs()):
                    checks.append(run_one(f'fixed-{index:02d}', spec))
                for concurrency in CONCURRENCIES:
                    wave_checks, accounting = mixed_wave(module, run_one, directory, concurrency)
                    checks.extend(wave_checks)
                    waves.append(accounting)
                for index, point in enumerate(('after_headers', 'after_first_committed')):
                    body = payload(prompt, {'shape': 'o32', 'streaming': True, 'raw': True})
                    request_path = directory/f'disconnect-{index}-request.json'
                    response_path = directory/f'disconnect-{index}-response.json'
                    write(request_path, {'payload': body, 'disconnect_boundary': point})
                    row = wire_request(module, port, body, timeout, disconnect=point)
                    write(response_path, row)
                    require(row['http_status'] == 200 and row['error'] is None and not row['deadline_fired']
                            and row['disconnect_observed'] and row['owned_connection_closed'], 'disconnect probe failed')
                    item = {'check_id': f'disconnect-{index}', 'kind': 'intentional_disconnect', 'completed': True,
                            'sampler': sampler, 'request': evidence(request_path), 'response': evidence(response_path),
                            'boundary': point, 'claim': 'connection closed and following request reuses engine; cancellation race outcome is not inferred', 'performance_claim': False}
                    write(directory/f'disconnect-{index}-check.json', item)
                    disconnects.append(item)
                    checks.append(run_one(f'post-disconnect-{index}', {'shape': 'o32', 'streaming': True, 'raw': True}))
            require(len(checks) == 44 and len(disconnects) == 2 and len({item['server_request_id'] for item in checks}) == 44, 'sampler request coverage/IDs differ')
            known = {item['server_request_id'] for item in checks}
            extra = []
            for path in sorted(audits.glob('cmpl-*.json')):
                if path.stem not in known:
                    _, item = validate_audit(audits, path.stem, references['o32'], True, sampler, identity, runtime_identity)
                    extra.append(item)
            require(len(extra) <= 2, 'more additional terminal audits than intentional disconnects')
            write(directory/'driver-maps-after.json', driver_maps(process, compute))
            process.terminate()
            process.wait(timeout=30)
            require(process.returncode == 0, 'source HTTP server did not exit gracefully')
            shutdown = shutdown_proof(audits, identity)
    finally:
        try:
            stop_owned(process)
        finally:
            remaining = owned_members(process) if process is not None else []
            write(directory/'process-exit.json', {'pid': process.pid if process else None,
                  'returncode': process.returncode if process else None, 'remaining_owned_pids': remaining,
                  'cleanup_verified': process is not None and not remaining and process.poll() is not None})
    verify()
    result = {'sampling_backend': sampler, 'completed': True, 'completion_checks': 44, 'disconnect_checks': 2,
              'checks': checks+disconnects, 'mixed_waves': waves, 'additional_audits_from_disconnect_races': extra,
              'startup': evidence(directory/'startup.json'), 'shutdown': shutdown,
              'preparation': evidence(directory/'preparation.json'), 'launch': evidence(directory/'launch.json'),
              'runtime_endpoint_response': evidence(directory/'runtime-endpoint-response.json'),
              'wave_artifacts': [evidence(directory/f'mixed-c{c}-accounting.json') for c in CONCURRENCIES],
              'process_exit': evidence(directory/'process-exit.json'), 'driver_maps_before': evidence(directory/'driver-maps-before.json'),
              'driver_maps_after': evidence(directory/'driver-maps-after.json'), 'log': evidence(log_path), 'performance_claim': False}
    write(directory/'completion.json', result)
    return result


def check_artifacts(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            require(evidence(value['path']) == value, 'artifact changed: '+value['path'])
        else:
            for item in value.values(): check_artifacts(item)
    elif isinstance(value, list):
        for item in value: check_artifacts(item)


def check_saved_request(row, body):
    require(row['request'] == body and row['request_body_sha256'] ==
            hashlib.sha256(json.dumps(body, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
            'saved transmitted request differs')
    require(row['owned_connection_closed'] is True and not row['cleanup_errors']
            and row['error'] is None and row['http_status'] == 200, 'saved transport/cleanup failed')


def reconstruct_raw(module, row, reference, streaming):
    require(row['status'] == 'success' and row['transport_complete'] is True,
            'saved raw response transport incomplete')
    parser = module.TokenResponseParser(reference, streaming=streaming, started_ns=row['started_ns'])
    for frame in row['frames']:
        if streaming:
            parser.feed_sse(frame['data'].encode(), frame['arrived_ns'])
        else:
            parser.feed_nonstream(frame['data'].encode(), frame['arrived_ns'])
    reconstructed = parser.finish()
    require(all(row[key] == value for key, value in reconstructed.items()),
            'saved token observation differs from raw response reconstruction')


def check_saved_wire_frames(module, row, streaming):
    raw = base64.b64decode(row['raw_response_base64'], validate=True)
    require(len(raw) <= MAX_BYTES and not row['deadline_fired'], 'saved wire response bound failed')
    if streaming:
        framer = module.SSEFramer(MAX_BYTES)
        data = [payload.decode() for payload, _ in framer.feed(raw, 0)]
        require(data == [frame['data'] for frame in row['frames']], 'saved SSE frames differ from raw body')
        if row['complete']:
            framer.finish()
    else:
        require(not row['frames'], 'nonstream wire response has SSE frames')


def validate_result(result):
    require(result['schema_version'] == SCHEMA and result['completed'] is True and result['gpu_tests_executed'] is True
            and result['performance_claim'] is False and result['source_commit'] == EXPECTED_COMMIT
            and result['binary']['sha256'] == EXPECTED_BINARY, 'token observation qualification identity/scope differs')
    require(result['sampling_backends'] == list(SAMPLERS) and result['offered_concurrencies'] == list(CONCURRENCIES)
            and result['active_capacity'] == 1 and result['proofs'] == dict.fromkeys(PROOFS, True), 'qualification proof coverage differs')
    require(len(result['per_sampler']) == 2 and len(result['checks']) == 92, 'qualification check count differs')
    check_artifacts(result)
    for name in ('helpers', 'model_files'):
        require(result[name] and all(shared.sha(path) == digest for path, digest in result[name].items()), name+' changed')
    require(result['client_module']['sha256'] == CLIENT_SHA
            and result['helpers'].get(result['client_module']['path']) == CLIENT_SHA, 'saved token client pin differs')
    module = load_client(result['client_module']['path'])
    preparation, canonical_request = read(result['preparation']['path']), read(result['request']['path'])
    binding = read(result['reference_binding']['path'])
    require(set(preparation['references']) == {'o32', 'o1', 'stop'}, 'saved reference shapes differ')
    references = {key: module.TokenReference(**value) for key, value in preparation['references'].items()}
    for key, reference in references.items():
        count = 1 if key == 'o1' else preparation['stop_reference_count'] if key == 'stop' else 32
        require(list(reference.prompt_token_ids) == binding['input_token_ids'] == canonical_request['prompt_token_ids']
                and len(reference.prompt_token_ids) == 128
                and list(reference.output_token_ids) == binding['generated_token_ids'][:count]
                and reference.finish_reason == ('stop' if key == 'stop' else 'length'), 'saved reference IDs/finish differ')
    require(references['stop'].text == '' and preparation['stop_reference_count'] == 3
            and tuple(references['stop'].output_token_ids) == STOP_TOKEN_IDS
            and preparation['stop_fixture'] == {'text': STOP_TEXT, 'token_ids': list(STOP_TOKEN_IDS),
                                               'decoded_prefixes': list(STOP_PREFIXES)}, 'saved blank-stop reference differs')
    for sampler, row in zip(SAMPLERS, result['per_sampler']):
        require(row['sampling_backend'] == sampler and row['completed'] is True and row['completion_checks'] == 44
                and row['disconnect_checks'] == 2 and len(row['checks']) == 46
                and all(item['completed'] is True and item['sampler'] == sampler for item in row['checks'])
                and [wave['concurrency'] for wave in row['mixed_waves']] == list(CONCURRENCIES)
                and all(wave['completed'] is True for wave in row['mixed_waves']), 'per-sampler proof incomplete')
        directory = Path(row['startup']['path']).parent
        identity, launch = read(row['launch']['path']), read(row['preparation']['path'])
        startup = read(row['startup']['path'])
        require(startup['runtime_identity'] == launch['runtime_identity'] and startup['candidate_id'] == CANDIDATE
                and startup['endpoint_payload_sha256'] == canonical_sha(startup['endpoint_payload']), 'saved startup identity differs')
        endpoint = read(row['runtime_endpoint_response']['path'])
        require(endpoint['error'] is None and endpoint['complete'] and not endpoint['deadline_fired']
                and endpoint['http_status'] == 200 and module.parse_json(base64.b64decode(endpoint['raw_response_base64']))
                == startup['endpoint_payload'], 'saved live runtime configuration differs')
        exit_record = read(row['process_exit']['path'])
        require(exit_record['pid'] == identity['pid'] and exit_record['returncode'] == 0
                and exit_record['cleanup_verified'] is True and exit_record['remaining_owned_pids'] == [], 'saved owned cleanup failed')
        require(shutdown_proof(directory/'audit', identity) == row['shutdown'], 'saved shutdown proof differs')
        checks_by_id, ids, raw_stop_blank = {}, set(), False
        for item in row['checks']:
            require(item['check_id'] not in checks_by_id, 'duplicate saved check ID')
            checks_by_id[item['check_id']] = item
            saved_request, response = read(item['request']['path']), read(item['response']['path'])
            body = saved_request['payload']
            check_saved_request(response, body)
            if item['kind'] == 'intentional_disconnect':
                require(item['boundary'] in ('after_headers', 'after_first_committed')
                        and body == payload(canonical_request['prompt'], {'shape': 'o32', 'streaming': True, 'raw': True})
                        and saved_request['disconnect_boundary'] == response['disconnect'] == item['boundary']
                        and response['disconnect_observed'] is True, 'saved deliberate disconnect differs')
                check_saved_wire_frames(module, response, True)
                if item['boundary'] == 'after_first_committed':
                    require(any(frame['data'] != '[DONE]' and any(choice.get('token_ids') for choice in
                                module.parse_json(frame['data']).get('choices', [])) for frame in response['frames']),
                            'saved disconnect lacked committed-token boundary')
                continue
            require(item['kind'] == 'completion' and item['spec'] in check_specs(), 'unexpected saved check kind/spec')
            spec, reference = item['spec'], references[item['spec']['shape']]
            require(body == payload(canonical_request['prompt'], spec)
                    and saved_request['expected_reference'] == preparation['references'][spec['shape']]
                    and saved_request['reference_sha256'] == reference.sha256, 'saved request/reference differs')
            if spec['raw']:
                reconstruct_raw(module, response, reference, spec['streaming'])
                observed = response
            else:
                check_saved_wire_frames(module, response, spec['streaming'])
                observed = default_observation(module, response, reference, spec['streaming'])
            request_id = observed['response_identity']['id']
            require(request_id == item['server_request_id'] and request_id not in ids, 'saved source request IDs repeat/differ')
            ids.add(request_id)
            audit, audit_evidence = validate_audit(directory/'audit', request_id, reference, spec['streaming'], sampler,
                                                 identity, startup['runtime_identity'])
            require(audit_evidence == item['audit'], 'saved check points to a different audit')
            blanks = validate_published_audit(module, response, audit, spec['streaming']) if spec['raw'] else 0
            require(blanks == item['blank_committed_tokens'] and observed['text'] ==
                    ''.join(token['emitted_text_delta'] for token in audit['committed_output_tokens']), 'saved audit text/count differs')
            raw_stop_blank |= bool(spec['raw'] and spec['streaming'] and spec['shape'] == 'stop' and blanks > 0)
            require(item['started_ns'] == response['started_ns'] and item['finished_ns'] ==
                    response.get('call_finished_ns', response['finished_ns']), 'saved check clock differs')
        require(len(ids) == 44 and raw_stop_blank, 'saved completed request/blank token proof incomplete')
        expected_ids = {f'fixed-{i:02d}' for i in range(12)} | {f'post-disconnect-{i}' for i in range(2)} | {f'disconnect-{i}' for i in range(2)}
        for c, wave, artifact in zip(CONCURRENCIES, row['mixed_waves'], row['wave_artifacts']):
            expected_wave = sorted(f'mixed-c{c}-w{worker}-r{step}' for worker in range(c) for step in range(2))
            expected_ids.update(expected_wave)
            require(read(artifact['path']) == wave and wave['checks'] == expected_wave and wave['requested'] == c*2
                    and not wave['failures'] and wave['observed_max_request_in_flight'] == c
                    and module.overlap_peak([checks_by_id[key] for key in expected_wave]) == c, 'saved mixed wave overlap/accounting differs')
        require(set(checks_by_id) == expected_ids and len(row['wave_artifacts']) == 4, 'saved exact workload inventory differs')
        for i, spec in enumerate(check_specs()):
            require(checks_by_id[f'fixed-{i:02d}']['spec'] == spec, 'saved fixed shape coverage differs')
        for i in range(2):
            require(checks_by_id[f'post-disconnect-{i}']['spec'] == {'shape': 'o32', 'streaming': True, 'raw': True}, 'saved engine reuse differs')
        for item in row['additional_audits_from_disconnect_races']:
            path = Path(item['artifact']['path'])
            _, validated = validate_audit(directory/'audit', path.stem, references['o32'], True, sampler,
                                          identity, startup['runtime_identity'])
            require(validated == item and path.stem not in ids, 'saved disconnect-race audit differs')
        require(len(row['additional_audits_from_disconnect_races']) <= 2, 'too many disconnect-race audits')
    require(result['checks'] == [item for row in result['per_sampler'] for item in row['checks']], 'top-level check inventory differs')
    return result


def validate_completion(path):
    return validate_result(read(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=shared.DEFAULT_ROOT)
    parser.add_argument('--base', type=Path, default=shared.DEFAULT_BASE)
    parser.add_argument('--client-module', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=30)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root, base = args.root.resolve(strict=True), args.base.resolve(strict=True)
    output = root/'http-token-observation-correctness'
    if args.validate_only:
        validate_completion(output/'completion.json')
        print(json.dumps({'completed': True, 'saved_proof_valid': True, 'gpu_tests_executed_now': False}))
        return
    require(1 <= args.timeout_seconds <= 120, 'timeout outside client bounds')
    client_path = (args.client_module or root/'serving_token_client.py').resolve(strict=True)
    require(shared.sha(client_path) == CLIENT_SHA, 'token client differs from frozen validated helper')
    module = load_client(client_path)
    from tokenizers import Tokenizer
    build_path = root/'http-token-build.json'
    build = read(build_path)
    build_sha = shared.sha(build_path)
    initial = shared.verify_snapshot(root, build, build_sha)
    require(initial['source_commit'] == EXPECTED_COMMIT and initial['binaries'][str(root/'http-token-target/release/riley')] == EXPECTED_BINARY, 'requires frozen optional-token source/build')
    gpu_path, default_path = root/'http-token-gpu-tests.json', root/'http-token-http-correctness/http-validation.json'
    gpu, default = read(gpu_path), read(default_path)
    require(gpu['passed'] and gpu['gpu_tests_executed'] and gpu['full_logits_and_kv_exact']
            and gpu['source_commit'] == default['source_commit'] == EXPECTED_COMMIT
            and gpu['build_sha256'] == default['build_sha256'] == build_sha
            and [row['sampling'] for row in default['results']] == list(SAMPLERS), 'prerequisite model/default-wire gates differ')
    binding_path, request_path = base/'native-binding.json', base/'request.json'
    binding, request = read(binding_path), read(request_path)
    model_files = {str(shared.MODEL/name): binding['workload'][key] for name, key in
                   (('model.safetensors', 'weights_sha256'), ('tokenizer.json', 'tokenizer_sha256'))}
    require(all(shared.sha(path) == digest for path, digest in model_files.items()), 'model files differ from reference')
    tokenizer = Tokenizer.from_file(str(shared.MODEL/'tokenizer.json'))
    prompt, prompt_ids, generated = request['prompt'], binding['input_token_ids'], binding['generated_token_ids']
    require(tokenizer.encode(prompt, add_special_tokens=True).ids == prompt_ids == request['prompt_token_ids']
            and len(prompt_ids) == 128 and len(generated) == request['requested_output_tokens'] == 32, 'fixed P128/O32 reference differs')
    stop_count, stop_text = stop_reference(tokenizer, generated)
    references = {name: module.TokenReference('g04-smol', tuple(prompt_ids), tuple(generated[:count]),
                  tokenizer.decode(generated[:count], skip_special_tokens=True) if name != 'stop' else stop_text,
                  'stop' if name == 'stop' else 'length') for name, count in (('o32', 32), ('o1', 1), ('stop', stop_count))}
    env = {key: value for key, value in shared.environment(root).items() if re.fullmatch(r'[A-Z_][A-Z0-9_]*', key)
           and key not in ('RILEY_FREEZE_SHA', 'RILEY_GATE_E_REPORT_SHA', 'RILEY_CONFIGURATION_SHA', 'RILEY_BASE_RELEASE_CANDIDATE_REPORT_SHA')}
    compute, sessions, gpu_query = runtime_and_sessions(env, binding)
    helper_paths = [Path(__file__), Path(shared.__file__), Path(runtime_check.__file__),
                    Path(runtime_check.session.__file__), root/'remote_session.py', client_path]
    helpers = {str(path.resolve()): shared.sha(path) for path in helper_paths}
    pinned_paths = [build_path, gpu_path, default_path, binding_path, request_path,
                    runtime_check.COMPUTE/'receipt.json', runtime_check.GUI/'receipt.json']
    pins = {str(path): shared.sha(path) for path in pinned_paths}
    def verify():
        require(all(shared.sha(path) == digest for path, digest in {**pins, **helpers, **model_files}.items()), 'qualification input/helper/model changed')
        require(shared.verify_snapshot(root, build, build_sha) == initial, 'source/build changed')
        current_compute, current_sessions, current_query = runtime_and_sessions(env, binding)
        require(current_compute == compute and current_sessions == sessions and current_query == gpu_query, 'retained sessions/GPU runtime changed')
    output.mkdir(mode=0o700, exist_ok=False)
    write(output/'preparation.json', {'schema_version': SCHEMA, 'source_build': evidence(build_path), 'inputs': pins,
          'helpers': helpers, 'model_files': model_files, 'retained_sessions': sessions, 'gpu_query': gpu_query,
          'stop_reference_count': stop_count,
          'stop_fixture': {'text': STOP_TEXT, 'token_ids': list(STOP_TOKEN_IDS), 'decoded_prefixes': list(STOP_PREFIXES)},
          'references': {key: {'model': ref.model, 'prompt_token_ids': list(ref.prompt_token_ids),
                         'output_token_ids': list(ref.output_token_ids), 'text': ref.text,
                         'finish_reason': ref.finish_reason} for key, ref in references.items()},
          'offered_concurrencies': list(CONCURRENCIES), 'active_capacity': 1,
          'performance_claim': False})
    try:
        results = [run_sampler(module, root, output, sampler, env, compute, prompt, references, verify, args.timeout_seconds) for sampler in SAMPLERS]
        verify()
        require(all(any(item.get('blank_committed_tokens', 0) > 0 and item.get('spec', {}).get('shape') == 'stop'
                        and item['spec']['streaming'] and item['spec']['raw'] for item in row['checks']) for row in results), 'both samplers must prove blank commits')
        result = {'schema_version': SCHEMA, 'completed': True, 'gpu_tests_executed': True, 'performance_claim': False,
                  'source_build': evidence(build_path), 'source_commit': EXPECTED_COMMIT,
                  'binary': evidence(root/'http-token-target/release/riley'), 'request': evidence(request_path),
                  'reference_binding': evidence(binding_path), 'prerequisites': [evidence(gpu_path), evidence(default_path)],
                  'model_files': model_files, 'helpers': helpers, 'client_module': evidence(client_path), 'sampling_backends': list(SAMPLERS),
                  'offered_concurrencies': list(CONCURRENCIES), 'active_capacity': 1, 'per_sampler': results,
                  'checks': [item for row in results for item in row['checks']], 'proofs': dict.fromkeys(PROOFS, True),
                  'retained_sessions_unchanged': True, 'preparation': evidence(output/'preparation.json')}
        validate_result(result)
        write(output/'completion.json', result)
        print(json.dumps({'completed': True, 'receipt': evidence(output/'completion.json'), 'checks': 92, 'performance_claim': False}))
    except BaseException as error:
        write(output/'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error), 'performance_claim': False})
        raise


if __name__ == '__main__':
    main()
