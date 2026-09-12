"""Capture seven fresh vLLM output settings with Blender retained; no performance claim.

Run on the prepared host: python3 diagnose_vllm_output_matrix.py [--logprobs]
The optional final wave requests logprobs=1; it is separately labelled because
logprob collection can change execution. All writes are to a new output tree.
Reference differences never abort response collection or qualify performance.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time

import run_serving_concurrency as load
import remote_session_round12 as session

ROOT = Path('/tmp/riley-opt-260912')
OUTPUT = ROOT / 'vllm-output-matrix-diagnostic'
RUNTIME = ROOT / 'driver580173-runtime-20260901'
RUNTIME_SHA = '36dba27991800124660e8f07274ffde21b79ebef14bb365d37a610f9a0802d70'
SETTINGS = [(1, 128), *((c, budget) for c in (2, 4, 8) for budget in (128, 128*c))]
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def require(value, label):
    if not value:
        raise RuntimeError(label)
    return value


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_bytes(path, value):
    with path.open('xb') as stream:
        stream.write(value)


def check_runtime():
    receipt_path = RUNTIME / 'receipt.json'
    require(load.shared.digest(receipt_path) == RUNTIME_SHA, 'accepted compute runtime receipt changed')
    receipt = load.read(receipt_path)
    require(receipt['completed'] and receipt['archive_signature_verified']
            and not receipt['host_packages_modified'] and not receipt['host_restart_performed'],
            'compute runtime was not privately verified')
    require(re.search(r'\b580\.173\.02\b', Path('/proc/driver/nvidia/version').read_text()),
            'loaded kernel no longer matches the old private runtime')
    extracted = RUNTIME / 'extracted'
    require(RUNTIME.is_dir() and not RUNTIME.is_symlink() and extracted.is_dir() and not extracted.is_symlink(),
            'private runtime root must be a real directory')
    observed_files, observed_links = set(), set()
    for path in extracted.rglob('*'):
        name = str(path.relative_to(extracted))
        if path.is_symlink():
            observed_links.add(name)
            require(receipt['symlinks'].get(name) == str(path.readlink()), 'runtime symlink changed: ' + name)
            resolved = path.resolve(strict=True)
            require(resolved.is_relative_to(extracted.resolve()) and resolved.is_file(),
                    'runtime symlink escaped its private directory: ' + name)
        elif path.is_file():
            observed_files.add(name)
            require(receipt['files'].get(name) == load.shared.digest(path), 'runtime file changed: ' + name)
        else:
            require(path.is_dir(), 'unexpected runtime filesystem entry: ' + name)
    require(observed_files == set(receipt['files']) and observed_links == set(receipt['symlinks']),
            'runtime file/symlink inventory changed')
    require(Path(receipt['library_path']) == extracted / 'usr/lib/x86_64-linux-gnu'
            and Path(receipt['nvidia_smi_path']) == extracted / 'usr/bin/nvidia-smi', 'runtime paths changed')
    return receipt


def check_sessions():
    originals = load.read(ROOT / 'blender-round12/session.json')
    restored = load.read(ROOT / 'blender-round12/verified.json')
    require(restored['alive_and_listening'] and restored['commands_and_gui_environment_match']
            and len(originals) == len(restored['processes']) == 3, 'three restored Blender sessions required')
    current = []
    for original, row in zip(originals, restored['processes']):
        actual = session.identity(row['new_pid'])
        require(actual['start'] == row['start'] and session.live(row['new_pid'])
                and session.same_command(actual, original)
                and session.process_tag(row['new_pid']) == row['tag'] and session.listening(row['port']),
                'canonical retained Blender identity changed')
        current.append({'pid': row['new_pid'], 'start': actual['start'], 'port': row['port'],
                        'argv': actual['argv'], 'cwd': actual['cwd'], 'gui_env': actual['env']})
    return current


def gpu_snapshot(binding, env, runtime):
    command = [runtime['nvidia_smi_path'], '-i', str(binding['environment']['gpu']['device_index'])]
    raw = subprocess.check_output(command + ['--query-gpu=uuid,driver_version,temperature.gpu,memory.used',
                                  '--format=csv,noheader,nounits'], env=env, text=True, timeout=15).strip()
    fields = [part.strip() for part in raw.split(',')]
    require(len(fields) == 4 and fields[0] == binding['environment']['gpu']['uuid']
            and fields[1] == '580.173.02', 'GPU identity or runtime version differs')
    processes = subprocess.check_output(command + ['--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                        env=env, text=True, timeout=15).strip()
    return {'gpu_uuid': fields[0], 'driver_version': fields[1], 'temperature_c': int(fields[2]),
            'memory_used_mib': int(fields[3]), 'compute_pids': processes.splitlines() if processes else [],
            'diagnostic_only': True, 'blender_retained': True}


def owned_members(process):
    members = []
    for item in Path('/proc').iterdir():
        if item.name.isdigit():
            try:
                if os.getpgid(int(item.name)) == process.pid and (item / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z':
                    members.append(int(item.name))
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                pass
    return sorted(members)


def driver_maps(process, runtime):
    records = {}
    checked = set()
    prefix = Path(runtime['library_path'])
    for pid in owned_members(process):
        try:
            lines = (Path('/proc') / str(pid) / 'maps').read_text().splitlines()
        except (ProcessLookupError, FileNotFoundError):
            continue
        selected = [line for line in lines if 'libcuda.so' in line or 'libnvidia' in line]
        for line in selected:
            path = Path(line.split(maxsplit=5)[5])
            if path not in checked:
                require(path.is_relative_to(prefix) and path.exists()
                        and runtime['files'].get(str(path.relative_to(RUNTIME / 'extracted'))) == load.shared.digest(path),
                        'foreign NVIDIA driver mapping: ' + line)
                checked.add(path)
        records[str(pid)] = selected
    all_lines = [line for lines in records.values() for line in lines]
    for stem in ('libcuda.so.580.173.02', 'libnvidia-ml.so.580.173.02'):
        require(any(str(prefix / stem) in line for line in all_lines), 'owned driver mapping missing: ' + stem)
    return records


def first_difference(actual, expected):
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            return index
    return min(len(actual), len(expected)) if len(actual) != len(expected) else None


def id_list(value):
    return isinstance(value, list) and all(type(token) is int and 0 <= token <= 0xffffffff for token in value)


def summarize_response(raw, stream, expected_text, binding, include_logprobs=False):
    """Validate structure/counts independently of exact reference continuation."""
    chunks = []
    done_count = 0
    if stream:
        normalized = raw.replace('\r\n', '\n')
        require(normalized.endswith('\n\n'), 'incomplete SSE frame')
        for event in normalized.split('\n\n'):
            if not event.strip():
                continue
            lines = event.splitlines()
            require(all(line.startswith('data:') or line.startswith(':') for line in lines), 'unexpected SSE field')
            data = '\n'.join(line[5:].lstrip(' ') for line in lines if line.startswith('data:'))
            if not data:
                continue
            require(done_count == 0, 'SSE data follows DONE')
            if data == '[DONE]':
                done_count += 1
            else:
                chunks.append(json.loads(data))
        require(done_count == 1, 'SSE DONE missing')
    else:
        chunks = [json.loads(raw)]
    text, ids, prompts, usages, finishes, token_counts, logprobs = '', [], [], [], [], [], []
    usage_only = 0
    identity = None
    for chunk in chunks:
        require(isinstance(chunk, dict) and isinstance(chunk.get('choices'), list), 'invalid completion object')
        current_identity = (chunk.get('id'), chunk.get('model'))
        require(all(isinstance(item, str) and item for item in current_identity), 'completion identity missing')
        if identity is None:
            identity = current_identity
        require(current_identity == identity, 'completion identity changed within stream')
        if chunk.get('usage') is not None:
            usages.append(chunk['usage'])
        if not chunk['choices']:
            require(stream and chunk.get('usage') is not None and len(finishes) == 1, 'misordered usage-only frame')
            usage_only += 1
            continue
        require(usage_only == 0 and len(chunk['choices']) == 1, 'choice count/order differs')
        choice = chunk['choices'][0]
        require(choice.get('index') == 0 and isinstance(choice.get('text'), str), 'invalid completion choice')
        require(not finishes, 'choice follows finish')
        text += choice['text']
        tokens = choice.get('token_ids')
        require(id_list(tokens), 'generated token IDs missing or invalid')
        ids.extend(tokens)
        token_counts.append(len(tokens))
        prompt = choice.get('prompt_token_ids')
        if prompt is not None:
            require(not prompts and len(token_counts) == 1 and id_list(prompt), 'prompt IDs repeated, invalid, or late')
            prompts.append(prompt)
        if choice.get('finish_reason') is not None:
            finishes.append(choice['finish_reason'])
        if choice.get('logprobs') is not None:
            logprobs.append(choice['logprobs'])
    require(len(prompts) == len(finishes) == len(usages) == 1, 'prompt/finish/usage cardinality differs')
    prompt, usage = prompts[0], usages[0]
    require(isinstance(usage, dict) and all(type(usage.get(key)) is int for key in
            ('prompt_tokens', 'completion_tokens', 'total_tokens')), 'usage integer fields missing')
    expected_ids, expected_prompt = binding['generated_token_ids'], binding['input_token_ids']
    valid = (len(ids) == len(expected_ids) and prompt == expected_prompt and finishes == ['length']
             and usage['prompt_tokens'] == len(prompt) and usage['completion_tokens'] == len(ids)
             and usage['total_tokens'] == len(prompt) + len(ids)
             and (not stream or usage_only == 1))
    result = {'structure_parsed': True, 'counts_prompt_usage_and_finish_valid': valid,
              'response_id': identity[0], 'response_model': identity[1],
              'text': text, 'token_ids': ids, 'prompt_token_ids': prompt,
              'generated_token_count': len(ids), 'prompt_token_count': len(prompt), 'usage': usage,
              'finish_reason': finishes[0], 'sse_done_count': done_count,
              'usage_only_frame_count': usage_only, 'choice_frame_token_counts': token_counts,
              'text_exact': text == expected_text, 'token_ids_exact': ids == expected_ids,
              'prompt_token_ids_exact': prompt == expected_prompt,
              'first_different_generated_index': first_difference(ids, expected_ids)}
    if include_logprobs:
        require(len(logprobs) == 1, 'one nonstream logprob object required')
        detail = logprobs[0]
        chosen, top = detail.get('token_logprobs'), detail.get('top_logprobs')
        require(isinstance(chosen, list) and isinstance(top, list) and len(chosen) == len(top) == len(ids),
                'logprob counts differ from generated IDs')
        checks = []
        for index, (probability, alternatives) in enumerate(zip(chosen, top)):
            require(type(probability) in (int, float) and math.isfinite(probability)
                    and isinstance(alternatives, dict) and alternatives
                    and all(type(value) in (int, float) and math.isfinite(value) for value in alternatives.values()),
                    'invalid observed logprobs')
            maximum = max(alternatives.values())
            checks.append({'generated_index': index, 'token_id': ids[index], 'selected_logprob': probability,
                           'maximum_returned_logprob': maximum, 'selected_at_returned_maximum': probability >= maximum})
        result['greedy_logprob_observation'] = {'requested_logprobs': 1, 'tokens': checks,
            'all_selected_at_returned_maximum': all(row['selected_at_returned_maximum'] for row in checks),
            'scope': 'Reported selected probability versus returned top-1; not an independent full-logit oracle.'}
    return result


def run_wave(directory, phase, concurrency, body, port, binding, expected_text, deadline, cleanup):
    """One simultaneous wave, preserving mismatch and partial-failure responses."""
    barrier = threading.Barrier(concurrency + 1)
    sockets, lock, timed_out = {}, threading.Lock(), set()
    rows, worker_errors = [], []
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='vllm-output-diagnostic')

    def abort():
        # Interrupt reads immediately, then terminate only the newly owned session.
        with lock:
            active = list(sockets.values())
        for connection in active:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        cleanup()

    def request_one(index):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=deadline)
        timer = response = None
        chunks = []
        record = {'phase': phase, 'index': index, 'offered_concurrency': concurrency, 'request': body,
                  'status': None, 'response_complete': False, 'diagnostic_only': True, 'performance_claim': False}
        def timeout():
            with lock:
                timed_out.add(index)
            try:
                abort()
            except Exception as error:
                with lock:
                    worker_errors.append({'operation': 'deadline_cleanup', 'error': str(error)})
        try:
            barrier.wait(timeout=30)
            record['started_ns'] = time.perf_counter_ns()
            timer = threading.Timer(deadline, timeout)
            timer.daemon = True
            timer.start()
            connection.connect()
            with lock:
                sockets[index] = connection.sock
            connection.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            record.update(status=response.status, headers=response.getheaders())
            size = 0
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                require(size <= MAX_RESPONSE_BYTES, 'response exceeded diagnostic byte bound')
            record['response_complete'] = True
            record['finished_ns'] = time.perf_counter_ns()
            require(record['finished_ns'] - record['started_ns'] <= deadline * 1e9, 'total request deadline exceeded')
            raw = b''.join(chunks).decode('utf-8')
            record['raw_response'] = raw
            if response.status == 200:
                try:
                    record['observation'] = summarize_response(raw, body['stream'], expected_text, binding,
                                                               body.get('logprobs') == 1)
                    require(record['observation']['response_model'] == body['model'], 'response model differs from request')
                except Exception as error:
                    record['parse_error'] = {'type': type(error).__name__, 'error': str(error)}
        except Exception as error:
            record['error'] = {'type': type(error).__name__, 'error': str(error)}
        finally:
            if timer:
                timer.cancel()
                timer.join()
            for close in ([response.close] if response else []) + [connection.close]:
                try:
                    close()
                except Exception as error:
                    record.setdefault('close_errors', []).append(str(error))
            with lock:
                sockets.pop(index, None)
                record['deadline_fired'] = index in timed_out
            record.setdefault('finished_ns', time.perf_counter_ns())
            raw_path = directory / f'{phase}-{index}.response'
            write_bytes(raw_path, b''.join(chunks))
            record['raw_bytes'] = load.evidence(raw_path)
            record['raw_byte_count'] = sum(map(len, chunks))
            path = directory / f'{phase}-{index}.json'
            write(path, record)
        return record, load.evidence(path)

    futures = []
    try:
        futures = [pool.submit(request_one, index) for index in range(concurrency)]
        barrier.wait(timeout=30)
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                worker_errors.append({'operation': 'worker', 'type': type(error).__name__, 'error': str(error)})
                abort()
    except BaseException:
        barrier.abort()
        abort()
        raise
    finally:
        barrier.abort()
        pool.shutdown(wait=True, cancel_futures=True)
    rows.sort(key=lambda pair: pair[0]['index'])
    intervals = [row for row, _ in rows if 'started_ns' in row]
    result = {'phase': phase, 'requested': concurrency, 'recorded': len(rows),
              'response_records': [evidence for _, evidence in rows],
              'responses_complete': sum(row['response_complete'] for row, _ in rows),
              'http_200': sum(row['status'] == 200 for row, _ in rows),
              'observed_max_request_in_flight': load.overlap_peak(intervals) if intervals else 0,
              'reference_text_equal': sum(row.get('observation', {}).get('text_exact') is True for row, _ in rows),
              'reference_token_ids_equal': sum(row.get('observation', {}).get('token_ids_exact') is True for row, _ in rows),
              'counts_prompt_usage_and_finish_valid': sum(row.get('observation', {}).get('counts_prompt_usage_and_finish_valid') is True for row, _ in rows),
              'parse_errors': sum('parse_error' in row for row, _ in rows),
              'request_errors': sum('error' in row for row, _ in rows),
              'watchdog_timed_out_indices': sorted(timed_out), 'worker_errors': worker_errors,
              'total_request_deadline_seconds': deadline, 'performance_claim': False}
    write(directory / (phase + '-accounting.json'), result)
    return result


def run_setting(label, plan, parent, request, binding, runtime, output, include_logprobs, verify):
    lane, concurrency = plan['http_lanes']['vllm'], plan['workload']['offered_concurrency']
    env = {**plan['base_environment'], **lane['env']}
    require(not env.get('LD_PRELOAD'), 'preload forbidden')
    env['LD_LIBRARY_PATH'] = ':'.join(filter(None, (runtime['library_path'], env.get('LD_LIBRARY_PATH'))))
    env['PATH'] = str(Path(runtime['nvidia_smi_path']).parent) + ':' + env['PATH']
    before = verify()
    load.shared.check_port(lane['port'])
    output.mkdir()
    write(output / 'preparation.json', {'label': label, 'argv': lane['argv'], 'environment': env,
          'cwd': parent['source_root'], 'source_commit': parent['source_commit'], 'fresh_process': True,
          'active_capacity': lane['active_capacity'], 'token_budget': lane['token_budget'],
          'offered_concurrency': concurrency, 'gpu_before': gpu_snapshot(binding, env, runtime),
          'blender_before': before, 'diagnostic_only': True, 'performance_claim': False})
    log_path = output / 'server.log'
    process = None
    phases, errors = [], []
    cleanup_lock = threading.Lock()
    cleaned = False
    def cleanup():
        nonlocal cleaned
        with cleanup_lock:
            if process is not None and not cleaned:
                load.shared.stop_owned_process(process)
                cleaned = True
    with log_path.open('x') as log:
        try:
            process = subprocess.Popen(lane['argv'], cwd=parent['source_root'], env=env,
                                       stdout=log, stderr=log, start_new_session=True)
            write(output / 'launch.json', {'pid': process.pid, 'process_group': process.pid,
                                         'fresh_process': True, 'argv': lane['argv']})
            load.shared.wait_ready(process, lane['port'], plan['startup_timeout_seconds'])
            startup = output / 'vllm-startup.log'
            write_bytes(startup, log_path.read_bytes())
            write(output / 'vllm-startup.json', load.validate_vllm_startup(startup, lane, plan['vllm_runtime']))
            write(output / 'owned-driver-mappings-before.json', driver_maps(process, runtime))
            waves = [('nonstream-first', concurrency, False, False), ('nonstream-repeat', concurrency, False, False),
                     ('offered-c1-same-capacity', 1, False, False), ('stream-with-usage', concurrency, True, False)]
            if include_logprobs:
                waves.append(('nonstream-logprobs-diagnostic', concurrency, False, True))
            for phase, offered, streaming, logs in waves:
                require(process.poll() is None, 'owned server exited before ' + phase)
                body = {'model': lane.get('model_id', parent.get('http_model_id', 'g04-smol')),
                        'prompt': request['prompt'], 'max_tokens': request['requested_output_tokens'],
                        'temperature': 0, 'top_p': 1, 'stream': streaming, 'return_token_ids': True}
                if streaming:
                    body['stream_options'] = {'include_usage': True}
                if logs:
                    body['logprobs'] = 1
                phases.append(run_wave(output, phase, offered, body, lane['port'], binding,
                                       lane['expected_output_text'], plan['request_timeout_seconds'], cleanup))
            write(output / 'owned-driver-mappings-after.json', driver_maps(process, runtime))
            require(process.poll() is None, 'owned server exited during collection')
        except Exception as error:
            errors.append({'type': type(error).__name__, 'error': str(error)})
        finally:
            try:
                cleanup()
            finally:
                remaining = owned_members(process) if process is not None else []
                write(output / 'process-exit.json', {'pid': process.pid if process else None,
                      'returncode': process.returncode if process else None, 'remaining_owned_pids': remaining,
                      'owned_session_cleanup_finished': cleaned and not remaining, 'log': load.evidence(log_path)})
            require(process is None or (cleaned and not remaining), 'owned process cleanup incomplete; stop matrix')
    after = verify()
    require(before == after, 'Blender sessions changed during diagnostic')
    expected = (4 if include_logprobs else 3) * concurrency + 1
    recorded = sum(phase['recorded'] for phase in phases)
    collection_completed = not errors and recorded == expected and all(
        phase['responses_complete'] == phase['requested'] and phase['http_200'] == phase['requested']
        and phase['counts_prompt_usage_and_finish_valid'] == phase['requested']
        and phase['observed_max_request_in_flight'] == phase['requested']
        and not phase['parse_errors'] and not phase['request_errors']
        and not phase['watchdog_timed_out_indices'] and not phase['worker_errors'] for phase in phases)
    result = {'label': label, 'completed': collection_completed,
              'diagnostic_only': True, 'performance_claim': False, 'concurrent_correctness_qualified': False,
              'reference_equality_is_observation_only': True, 'requests_expected': expected,
              'requests_recorded': recorded, 'phases': phases, 'errors': errors, 'blender_after': after,
              'gpu_after': gpu_snapshot(binding, env, runtime),
              'artifacts': [load.evidence(path) for path in sorted(output.iterdir()) if path.is_file()]}
    write(output / 'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logprobs', action='store_true', help='add a separately labelled C-wave with logprobs=1')
    args = parser.parse_args()
    require(not OUTPUT.exists(), 'output directory already exists')
    matrix_path = ROOT / 'concurrency-screen-plans/matrix.json'
    matrix = load.read(matrix_path)
    require(matrix['schema_version'] == 'riley.concurrency-screen-matrix.v1', 'unknown prepared matrix')
    request_path, binding_path = Path(matrix['request']['path']), Path(matrix['binding']['path'])
    require(load.evidence(request_path) == matrix['request'] and load.evidence(binding_path) == matrix['binding'],
            'prepared reference input pins changed')
    request, binding = load.read(request_path), load.read(binding_path)
    runtime = check_runtime()
    originals = check_sessions()
    helpers = [Path(__file__), Path(load.__file__), Path(load.shared.__file__), Path(session.__file__),
               ROOT / 'remote_session.py', ROOT / 'diagnose_vllm_concurrency_output.py']
    inputs = [matrix_path, request_path, binding_path, RUNTIME / 'receipt.json',
              ROOT / 'blender-round12/session.json', ROOT / 'blender-round12/verified.json', *helpers]
    prepared = []
    require(len(matrix['plans']) == len(SETTINGS), 'expected exactly seven plans')
    for entry, (concurrency, budget) in zip(matrix['plans'], SETTINGS):
        label = f'c{concurrency}-vllm-budget{budget}'
        require(entry['label'] == label and load.evidence(entry['plan']['path']) == entry['plan'], 'matrix plan differs')
        plan = load.read(entry['plan']['path'])
        parent = load.validate_manifest(plan, request_path, binding_path, request, binding)
        require(parent['source_commit'] == matrix['source_commit']
                and plan['workload']['offered_concurrency'] == plan['http_lanes']['vllm']['active_capacity'] == concurrency
                and plan['http_lanes']['vllm']['token_budget'] == budget, 'matrix setting/source differs')
        prepared.append((label, plan, parent))
        inputs.append(Path(entry['plan']['path']))
    require(all(plan['parent_c1_plan'] == prepared[0][1]['parent_c1_plan'] for _, plan, _ in prepared),
            'all settings must share the identical qualified reference parent')
    pins = {str(path): load.shared.digest(path) for path in inputs}
    def verify():
        require(all(load.shared.digest(path) == digest for path, digest in pins.items()), 'diagnostic input changed')
        check_runtime()
        current = check_sessions()
        require(current == originals, 'retained Blender sessions changed')
        # All seven plans were checked above and remain byte-pinned. Recheck
        # their shared recursive source/model/binary parent once per boundary.
        load.validate_manifest(prepared[0][1], request_path, binding_path, request, binding)
        return current
    OUTPUT.mkdir()
    write(OUTPUT / 'preparation.json', {'schema_version': 'riley.vllm-output-matrix-diagnostic.v1',
          'diagnostic_only': True, 'performance_claim': False, 'concurrent_correctness_qualified': False,
          'source_commit': matrix['source_commit'], 'inputs': pins, 'blender_before': originals,
          'runtime_override': load.evidence(RUNTIME / 'receipt.json'), 'logprobs_wave_enabled': args.logprobs,
          'reference_text': prepared[0][1]['http_lanes']['vllm']['expected_output_text'],
          'reference_token_ids': binding['generated_token_ids'], 'reference_prompt_token_ids': binding['input_token_ids'],
          'expected_requests': sum((4 if args.logprobs else 3) * c + 1 for c, _ in SETTINGS),
          'scope': 'Fresh comparator outputs with Blender retained. C1 qualification is reference-only; no timing metric or new numerical gate.'})
    results = []
    try:
        for label, plan, parent in prepared:
            result = run_setting(label, plan, parent, request, binding, runtime, OUTPUT / label, args.logprobs, verify)
            results.append({'label': label, 'completed': result['completed'],
                            'requests_recorded': result['requests_recorded'],
                            'completion': load.evidence(OUTPUT / label / 'completion.json')})
        verify()
        completed = len(results) == 7 and all(row['completed'] for row in results)
        write(OUTPUT / 'completion.json', {'completed': completed, 'diagnostic_only': True, 'performance_claim': False,
              'concurrent_correctness_qualified': False, 'reference_equality_is_observation_only': True,
              'fresh_process_settings': len(results), 'requests_recorded': sum(row['requests_recorded'] for row in results),
              'settings': results, 'blender_sessions_unchanged': True,
              'preparation': load.evidence(OUTPUT / 'preparation.json')})
        print(json.dumps({'completed': completed, 'diagnostic_only': True, 'output': str(OUTPUT)}))
        return 0 if completed else 1
    except BaseException as error:
        write(OUTPUT / 'failure.json', {'completed': False, 'error': str(error), 'type': type(error).__name__,
                                     'finished_settings': results, 'performance_claim': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
