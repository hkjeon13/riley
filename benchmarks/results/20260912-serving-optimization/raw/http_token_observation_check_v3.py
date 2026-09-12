#!/usr/bin/env python3
"""V3 HTTP token correctness with source-path and full-GPU proof; no C02 audit claim.

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

SCHEMA = 'riley.http-token-observation-correctness.v3'
EXPECTED_COMMIT = 'a179617070526068b66ba5627ba82a7151da8c64'
EXPECTED_BINARY = '18cbd5f8a8ad8583ecfcd2d38815a484831c05eb9f081b1db67be6194ffebd8c'
CLIENT_SHA = 'a2a4a35569d6b892542097c119e2be8a6402beb60ab1565aa95658794c364766'
SAMPLERS = ('cpu', 'gpu-greedy')
CONCURRENCIES = (1, 2, 4, 8)
CANDIDATE = 'riley-0.0.0-rc13'
PROOFS = ('published_ids_match_reference', 'default_and_raw_text_usage_match', 'input_ids_exact',
          'blank_generated_token_events_counted', 'disconnect_reuse_exact',
          'source_commit_path_reviewed_and_unit_tested', 'existing_full_gpu_logits_kv_exact',
          'owned_process_cleanup_verified')
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



def published_events(module, row):
    events = []
    for frame in row['frames']:
        if frame['data'] != '[DONE]':
            for choice in module.parse_json(frame['data']).get('choices', []):
                if choice.get('token_ids'):
                    require(len(choice['token_ids']) == 1, 'multiple IDs in one generated-token event')
                    events.append({'token_id': choice['token_ids'][0], 'text': choice['text']})
    require([item['token_id'] for item in events] == row['token_ids']
            and len(events) == row['usage']['completion_tokens']
            and ''.join(item['text'] for item in events) == row['text'], 'raw event IDs/text/count disagree')
    return events


def execute_check(module, client, port, directory, check_id, spec, prompt, references, sampler, timeout):
    reference, body = references[spec['shape']], payload(prompt, spec)
    request_path, result_path = directory/(check_id+'-request.json'), directory/(check_id+'-response.json')
    write(request_path, {'payload': body, 'expected_reference': {'model': reference.model,
          'prompt_token_ids': list(reference.prompt_token_ids), 'output_token_ids': list(reference.output_token_ids),
          'text': reference.text, 'finish_reason': reference.finish_reason}, 'reference_sha256': reference.sha256})
    try:
        if spec['raw']:
            row = client.request(port, body, reference, streaming=spec['streaming'], mode='strict', timeout_seconds=timeout)
            write(result_path, row)
            check_saved_request(row, body)
            require(row['status'] == 'success' and row['protocol_valid'] and row['reference_match'], 'raw token response failed')
            observed = row
        else:
            row = wire_request(module, port, body, timeout)
            write(result_path, row)
            check_saved_request(row, body)
            observed = default_observation(module, row, reference, spec['streaming'])
        events = published_events(module, row) if spec['raw'] and spec['streaming'] else []
        blanks = sum(item['text'] == '' for item in events)
        if spec['raw'] and spec['shape'] == 'o1':
            require(row['metrics']['token_tpot_ns'] is None, 'O1 token TPOT must be undefined')
        if spec['raw'] and spec['streaming'] and spec['shape'] == 'stop':
            require(blanks == 3 and len(events) == 3 and row['text'] == '', 'stop must retain all three blank token events')
        result = {'check_id': check_id, 'kind': 'completion', 'completed': True, 'sampler': sampler,
                  'spec': spec, 'server_request_id': observed['response_identity']['id'],
                  'request': evidence(request_path), 'response': evidence(result_path), 'blank_generated_token_events': blanks,
                  'started_ns': row['started_ns'], 'finished_ns': row.get('call_finished_ns', row['finished_ns']),
                  'published_ids_match_reference': spec['raw'], 'performance_claim': False}
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



SOURCE_TESTS = ('committed_empty_token_is_delivered_and_channel_overflow_cancels',
    'token_collector_rejects_missing_repeated_or_out_of_order_metadata_and_bounds',
    'opted_in_sse_keeps_empty_tokens_and_requires_checked_usage_before_one_done',
    'committed_tokens_and_usage_survive_concurrent_http_delivery',
    'opt_in_nonstream_response_counts_serialized_token_bytes')
HTTP_FILES = tuple('crates/riley-server/src/'+name+'.rs' for name in ('domain', 'engine', 'openai', 'service'))


def validate_full_gpu_receipt(build, build_path, gpu_path, binding_path):
    gpu = read(gpu_path)
    require(gpu['schema_version'] == 'riley.http-token-gpu-correctness.v1' and gpu['passed'] is True
            and gpu['gpu_tests_executed'] is True and gpu['full_logits_and_kv_exact'] is True
            and gpu['vllm_reference_tokens_exact'] is True and gpu['source_clean'] is True
            and gpu['source_commit'] == build['source_commit'] == EXPECTED_COMMIT
            and gpu['binaries'] == build['binaries'] and gpu['build_sha256'] == shared.sha(build_path)
            and gpu['reference_binding_sha256'] == shared.sha(binding_path), 'full GPU prerequisite identity/scope differs')
    require(len(gpu['checks']) == len(shared.GPU_TESTS) == 4, 'full GPU prerequisite coverage differs')
    for row, name in zip(gpu['checks'], shared.GPU_TESTS):
        expected_argv = ['cargo', 'test', '--release', '-p', 'riley-runtime', '--features', 'cuda', '--lib', name,
                         '--', '--ignored', '--nocapture', '--test-threads=1']
        require(row['name'] == name and row['argv'] == expected_argv and row['passed'] is True
                and shared.sha(row['path']) == row['sha256'], 'full GPU prerequisite log/invocation changed')
        counts = shared.validate_test_log(Path(row['path']), name)
        require(all(row[key] == value for key, value in counts.items()), 'full GPU prerequisite counts differ')
    return gpu


def source_commit_contract(source, build, server_tests, gpu_receipt):
    files = {name: evidence(source/name) for name in HTTP_FILES}
    require(all(files[name]['sha256'] == build['source_files'][name] for name in HTTP_FILES), 'reviewed HTTP source differs from build')
    engine = (source/'crates/riley-server/src/engine.rs').read_text()
    first = engine.index('self.scheduler_mut()?.complete_iteration(&result, now_ns)')
    publication = engine.index('for token in updates.token_events() {', first)
    event = engine.index('GenerationEvent::CommittedToken {', publication)
    tail = engine.index('events.extend(self.process_completions(updates.completions())?);', event)
    block = engine[publication:tail]
    require(first < publication < event < tail and all(text in block for text in
            ('pending.token_id != token.token_id()', 'token_id: token.token_id()',
             'generated_index: token.generated_index()', 'prompt_token_ids: request.prompt_ids_for_delivery.take()',
             'events.push(BackendEvent::new(request.engine_id, event));')), 'source committed publication contract changed')
    openai = (source/'crates/riley-server/src/openai.rs').read_text()
    require(all(text in openai for text in ('self.token_ids.push(*token_id);',
            'chunk.choices[0].token_ids = Some(vec![*token_id]);',
            'self.token_ids.len() as u64 != usage.completion_tokens()',
            'response.choices[0].token_ids = Some(self.token_ids);')), 'source direct ID/count forwarding changed')
    log = Path(server_tests['path'])
    require(server_tests['passed'] is True and shared.sha(log) == server_tests['sha256'], 'server unit-test evidence changed')
    counts = shared.validate_test_log(log)
    require(all(server_tests[key] == value for key, value in counts.items()), 'server unit-test counts differ')
    content = log.read_text()
    require(all(re.search(r'^test (?:[A-Za-z0-9_]+::)*'+re.escape(name)+r' \.\.\. ok$', content, re.MULTILINE)
                for name in SOURCE_TESTS), 'required committed-token unit tests did not pass')
    return {'files': files, 'server_unit_tests': {'log': evidence(log), 'tests': list(SOURCE_TESTS), **counts},
            'full_gpu_receipt': evidence(gpu_receipt), 'review': {
                'scheduler_commit_precedes_external_publication': True,
                'token_ids_are_forwarded_without_retokenization': True,
                'generated_order_prompt_once_and_terminal_counts_checked': True,
                'empty_text_committed_events_remain_serialized_when_opted_in': True},
            'independent_per_request_committed_audit': False, 'c02_enabled': False}


def launch_identity(process, argv, cwd):
    require(process.poll() is None, 'owned server exited before launch identity check')
    directory = Path('/proc')/str(process.pid)
    observed_argv = (directory/'cmdline').read_bytes().split(b'\0')
    require(observed_argv[-1:] == [b''], 'incomplete process argv')
    observed_argv = [part.decode() for part in observed_argv[:-1]]
    require(observed_argv == argv and (directory/'cwd').resolve() == cwd.resolve(), 'actual server argv/cwd differs')
    return {'pid': process.pid, 'start_ticks': process_start(process.pid), 'argv': observed_argv,
            'cwd': str(cwd.resolve()), 'actual_process_identity_verified': True}


def check_config_unavailable(module, row):
    require(row['error'] is None and row['complete'] and not row['deadline_fired'] and row['http_status'] == 503,
            'profile without C02 must keep runtime configuration unavailable')
    value = module.parse_json(base64.b64decode(row['raw_response_base64']))
    require(value['error']['code'] == 'config_unavailable', 'unexpected runtime configuration error')
    return {'available': False, 'expected_status': 503, 'reason': 'profile excludes C02 runtime config artifacts'}


def run_sampler(module, root, output, sampler, env, compute, prompt, references, verify, timeout):
    verify()
    directory = output/sampler
    directory.mkdir(mode=0o700)
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0)); port = reserved.getsockname()[1]
    binary, cwd = root/'http-token-target/release/riley', root/'http-token-source'
    argv = [str(binary), 'serve', '--model', str(shared.MODEL), '--model-id', 'g04-smol', '--bind', f'127.0.0.1:{port}',
            '--max-active-sequences', '1', '--max-waiting-requests', '16', '--batch-token-budget', '128',
            '--prefill-chunk-tokens', '128', '--max-sequence-tokens', '160', '--max-output-tokens', '32', '--kv-blocks', '10',
            '--residual-rmsnorm', 'separate', '--execution-completion', 'iteration-batch', '--metadata-transport', 'packed-async',
            '--execution-graph-policy', 'require', '--graph-numerics', 'vllm-smol-p128-v1', '--sampling-backend', sampler]
    write(directory/'preparation.json', {'argv': argv, 'cwd': str(cwd), 'environment_sha256': canonical_sha(env),
          'selected_environment': {key: env[key] for key in ('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CARGO_TARGET_DIR')},
          'c02_enabled': False, 'performance_claim': False})
    process, checks, waves, disconnects = None, [], [], []
    log_path = directory/'server.log'
    try:
        with log_path.open('x') as log:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=log, start_new_session=True)
            wait_ready(module, process, port)
            identity = launch_identity(process, argv, cwd)
            write(directory/'launch.json', identity)
            endpoint_row = wire_request(module, port, None, min(timeout, 5), method='GET', route='/v1/config')
            write(directory/'runtime-endpoint-response.json', endpoint_row)
            config_status = check_config_unavailable(module, endpoint_row)
            write(directory/'driver-maps-before.json', driver_maps(process, compute))
            with module.TokenHttpClient() as client:
                def run_one(check_id, spec):
                    require(process.poll() is None, 'owned server exited before request')
                    return execute_check(module, client, port, directory, check_id, spec, prompt, references, sampler, timeout)
                for index, spec in enumerate(check_specs()): checks.append(run_one(f'fixed-{index:02d}', spec))
                for concurrency in CONCURRENCIES:
                    wave_checks, accounting = mixed_wave(module, run_one, directory, concurrency)
                    checks.extend(wave_checks); waves.append(accounting)
                for index, point in enumerate(('after_headers', 'after_first_committed')):
                    body = payload(prompt, {'shape': 'o32', 'streaming': True, 'raw': True})
                    request_path, response_path = directory/f'disconnect-{index}-request.json', directory/f'disconnect-{index}-response.json'
                    write(request_path, {'payload': body, 'disconnect_boundary': point})
                    row = wire_request(module, port, body, timeout, disconnect=point)
                    write(response_path, row)
                    require(row['http_status'] == 200 and row['error'] is None and not row['deadline_fired']
                            and row['disconnect_observed'] and row['owned_connection_closed'], 'disconnect probe failed')
                    item = {'check_id': f'disconnect-{index}', 'kind': 'intentional_disconnect', 'completed': True,
                            'sampler': sampler, 'request': evidence(request_path), 'response': evidence(response_path),
                            'boundary': point, 'claim': 'connection closed and following request reuses engine; cancellation race outcome is not inferred', 'performance_claim': False}
                    write(directory/f'disconnect-{index}-check.json', item); disconnects.append(item)
                    checks.append(run_one(f'post-disconnect-{index}', {'shape': 'o32', 'streaming': True, 'raw': True}))
            require(len(checks) == 44 and len(disconnects) == 2 and len({item['server_request_id'] for item in checks}) == 44, 'sampler request coverage/IDs differ')
            require(launch_identity(process, argv, cwd) == identity, 'owned server identity changed')
            write(directory/'driver-maps-after.json', driver_maps(process, compute))
            process.terminate(); process.wait(timeout=30)
            require(process.returncode == 0, 'source HTTP server did not exit gracefully')
    finally:
        try: stop_owned(process)
        finally:
            remaining = owned_members(process) if process is not None else []
            write(directory/'process-exit.json', {'pid': process.pid if process else None,
                  'returncode': process.returncode if process else None, 'remaining_owned_pids': remaining,
                  'cleanup_verified': process is not None and not remaining and process.poll() is not None})
    verify()
    result = {'sampling_backend': sampler, 'completed': True, 'completion_checks': 44, 'disconnect_checks': 2,
              'checks': checks+disconnects, 'mixed_waves': waves, 'runtime_config': config_status,
              'preparation': evidence(directory/'preparation.json'), 'launch': evidence(directory/'launch.json'),
              'runtime_endpoint_response': evidence(directory/'runtime-endpoint-response.json'),
              'wave_artifacts': [evidence(directory/f'mixed-c{c}-accounting.json') for c in CONCURRENCIES],
              'process_exit': evidence(directory/'process-exit.json'), 'driver_maps_before': evidence(directory/'driver-maps-before.json'),
              'driver_maps_after': evidence(directory/'driver-maps-after.json'), 'log': evidence(log_path),
              'c02_enabled': False, 'native_shutdown_allocation_receipt': False, 'performance_claim': False}
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
    require(result['independent_per_request_committed_audit'] is False and result['c02_enabled'] is False, 'unsupported audit claim')
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
        directory = Path(row['preparation']['path']).parent
        identity, launch = read(row['launch']['path']), read(row['preparation']['path'])
        require(identity['argv'] == launch['argv'] and identity['cwd'] == launch['cwd']
                and identity['actual_process_identity_verified'] is True and launch['c02_enabled'] is False
                and not any(arg.startswith('--c02-') for arg in launch['argv']), 'saved launch identity differs or enabled C02')
        require(row['runtime_config'] == check_config_unavailable(module, read(row['runtime_endpoint_response']['path']))
                and row['c02_enabled'] is False and row['native_shutdown_allocation_receipt'] is False,
                'saved profile/config scope differs')
        exit_record = read(row['process_exit']['path'])
        require(exit_record['pid'] == identity['pid'] and exit_record['returncode'] == 0
                and exit_record['cleanup_verified'] is True and exit_record['remaining_owned_pids'] == [], 'saved owned cleanup failed')
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
            events = published_events(module, response) if spec['raw'] and spec['streaming'] else []
            blanks = sum(event['text'] == '' for event in events)
            require(blanks == item['blank_generated_token_events'], 'saved empty token event count differs')
            if spec['raw'] and spec['streaming'] and spec['shape'] == 'stop':
                require(blanks == len(events) == 3 and response['text'] == '', 'saved stop lost blank generated-token events')
                raw_stop_blank = True
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
    require(result['checks'] == [item for row in result['per_sampler'] for item in row['checks']], 'top-level check inventory differs')
    build = read(result['source_build']['path'])
    gpu = validate_full_gpu_receipt(build, Path(result['source_build']['path']),
                                    Path(result['prerequisites'][0]['path']), Path(result['reference_binding']['path']))
    require(result['source_commit_path'] == source_commit_contract(Path(build['source_root']), build, gpu['server_unit_tests'],
            Path(result['prerequisites'][0]['path'])), 'saved source path/unit-test evidence changed')
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
    output = root/'http-token-observation-correctness-v3'
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
    validate_full_gpu_receipt(build, build_path, gpu_path, base/'native-binding.json')
    source_proof = source_commit_contract(root/'http-token-source', build, gpu['server_unit_tests'], gpu_path)
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
        require(source_commit_contract(root/'http-token-source', build, gpu['server_unit_tests'], gpu_path) == source_proof, 'source/unit proof changed')
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
        require(all(any(item.get('blank_generated_token_events', 0) == 3 and item.get('spec', {}).get('shape') == 'stop'
                        and item['spec']['streaming'] and item['spec']['raw'] for item in row['checks']) for row in results), 'both samplers must prove blank commits')
        result = {'schema_version': SCHEMA, 'completed': True, 'gpu_tests_executed': True, 'performance_claim': False,
                  'source_build': evidence(build_path), 'source_commit': EXPECTED_COMMIT,
                  'source_commit_path': source_proof, 'independent_per_request_committed_audit': False, 'c02_enabled': False,
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
