#!/usr/bin/env python3
"""Local CPU-only contracts; no model, CUDA, NVIDIA APIs, or remote process actions."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import http_token_observation_check_v3 as subject
CLIENT = HERE.parents[1]/'scripts/serving_token_client.py'
client = subject.load_client(CLIENT)
IDENTITY = {'pid': 42, 'start_ticks': 99}
RUNTIME = {'configuration_profile': 'stable-default', 'configuration_sha256': 'a'*64}


def references():
    prompt = tuple([19556]*128)
    generated = tuple(json.loads((HERE.parent/'20260911-g04-vllm-profile/native-binding.json').read_text())['generated_token_ids'])
    return {key: client.TokenReference('g04-smol', prompt, generated[:count], text, finish)
            for key, count, text, finish in (('o32', 32, 'T'*32, 'length'), ('o1', 1, 'T', 'length'),
                                            ('stop', 3, '', 'stop'))}


def audit(directory, request_id, ref, streaming, sampler='cpu', runtime=RUNTIME):
    backend = 'cpu-normative' if sampler == 'cpu' else sampler
    record = {'schema_version': 'riley.c02-generation-audit.v2', 'candidate_id': subject.CANDIDATE,
              'server_request_id': request_id, 'runtime_identity': runtime, 'process_identity': IDENTITY,
              'delivery_mode': 'stream' if streaming else 'non-stream', 'prompt_token_ids': list(ref.prompt_token_ids),
              'committed_output_tokens': [{'token_id': value, 'emitted_text_delta': '' if ref.finish_reason == 'stop' else 'T'}
                                          for value in ref.output_token_ids], 'finish_reason': ref.finish_reason,
              'usage': {'prompt_tokens': 128, 'completion_tokens': len(ref.output_token_ids), 'total_tokens': 128+len(ref.output_token_ids)},
              'sampling_selections': [{'iteration_id': index+1, 'configured_backend': backend, 'selected_backend': backend,
                                      'ineligibility_reason': 'gpu-greedy-not-configured' if sampler == 'cpu' else None,
                                      'committed': True} for index in range(len(ref.output_token_ids))]}
    return record


@contextmanager
def serving(directory, sampler='cpu', behavior='normal', version='HTTP/1.1', port=0, endpoint=None, runtime=RUNTIME):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = version
        def log_message(self, *args): pass
        def do_GET(self):
            encoded = json.dumps(endpoint).encode()
            self.send_response(503 if endpoint and endpoint.get('error', {}).get('code') == 'config_unavailable' else 200); self.send_header('Content-Length', str(len(encoded)))
            self.send_header('Connection', 'close'); self.end_headers(); self.wfile.write(encoded)
            self.close_connection = True
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with self.server.counter_lock:
                index = self.server.counter
                self.server.counter += 1
            key = 'stop' if body.get('stop') else 'o1' if body['max_tokens'] == 1 else 'o32'
            ref = references()[key]
            request_id = f'cmpl-local{index}'
            record = audit(directory, request_id, ref, body['stream'], sampler, runtime)
            metadata = {'id': request_id, 'model': 'g04-smol', 'object': 'text_completion', 'created': 1}
            values = []
            if body['stream']:
                for token_index, token in enumerate(record['committed_output_tokens']):
                    choice = {'index': 0, 'text': token['emitted_text_delta'], 'finish_reason': None}
                    if body.get('return_token_ids'):
                        choice['token_ids'] = [token['token_id']]
                        if token_index == 0: choice['prompt_token_ids'] = list(ref.prompt_token_ids)
                    values.append({**metadata, 'choices': [choice]})
                values.append({**metadata, 'choices': [{'index': 0, 'text': '', 'finish_reason': ref.finish_reason}]})
                if body.get('stream_options', {}).get('include_usage'):
                    values.append({**metadata, 'choices': [], 'usage': record['usage']})
                encoded = b''.join(b'data: '+json.dumps(value).encode()+b'\n\n' for value in values)+b'data: [DONE]\n\n'
            else:
                choice = {'index': 0, 'text': ref.text, 'finish_reason': ref.finish_reason}
                if body.get('return_token_ids'):
                    choice.update(token_ids=list(ref.output_token_ids), prompt_token_ids=list(ref.prompt_token_ids))
                encoded = json.dumps({**metadata, 'choices': [choice], 'usage': record['usage']}).encode()
            try:
                if behavior == 'headers_timeout':
                    time.sleep(.35)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream' if body['stream'] else 'application/json')
                self.send_header('Connection', 'close')
                if version == 'HTTP/1.1': self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                if behavior == 'trickle':
                    for offset in range(0, len(encoded), 10):
                        self.wfile.write(encoded[offset:offset+10]); self.wfile.flush(); time.sleep(.025)
                else:
                    time.sleep(.025)  # Demonstrate concurrent offered requests even on a fast CPU mock.
                    self.wfile.write(encoded); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
            self.close_connection = True
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.counter, server.counter_lock = 0, threading.Lock()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try: yield server.server_port
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=2)
        assert not worker.is_alive()


class ProtocolContracts(unittest.TestCase):
    def test_frozen_client_and_source_delivery_name(self):
        self.assertEqual(subject.shared.sha(CLIENT), subject.CLIENT_SHA)
        self.assertIn('Self::NonStream => "non-stream"', (HERE.parents[2]/'crates/riley-server/src/engine.rs').read_text())

    def test_real_frozen_stop_ids_and_decoded_prefixes(self):
        binding = subject.read(HERE.parent/'20260911-g04-vllm-profile/native-binding.json')
        # These actual prefixes were independently decoded by the root's real
        # tokenizer preflight. The production helper repeats that decode itself.
        observed = {tuple(binding['generated_token_ids'][:count]): text
                    for count, text in enumerate((",", ", I", ", I'm"), 1)}
        class RecordedTokenizer:
            def decode(self, values, skip_special_tokens):
                self_skip = skip_special_tokens
                assert self_skip is True
                return observed[tuple(values)]
        self.assertEqual(subject.stop_reference(RecordedTokenizer(), binding['generated_token_ids']), (3, ''))
        self.assertEqual(subject.payload('prompt', {'shape': 'stop', 'streaming': True, 'raw': True})['stop'], ", I'm")
        with patch.object(subject, 'STOP_TEXT', "I'm"):
            with self.assertRaisesRegex(ValueError, 'third committed token'):
                subject.stop_reference(RecordedTokenizer(), binding['generated_token_ids'])
        corrupted = list(binding['generated_token_ids']); corrupted[0] += 1
        with self.assertRaisesRegex(ValueError, 'token IDs differ'):
            subject.stop_reference(RecordedTokenizer(), corrupted)

    def test_all_twelve_formats_for_both_samplers_reconstruct_exactly(self):
        for sampler in subject.SAMPLERS:
            with self.subTest(sampler=sampler), tempfile.TemporaryDirectory() as name:
                root = Path(name); audits = root/'audit'; audits.mkdir()
                with serving(audits, sampler) as port, client.TokenHttpClient() as transport:
                    for index, spec in enumerate(subject.check_specs()):
                        result = subject.execute_check(client, transport, port, root, f'case-{index}', spec, 'prompt',
                                                       references(), sampler, 2)
                        response = subject.read(result['response']['path'])
                        subject.check_saved_request(response, subject.payload('prompt', spec))
                        if spec['raw']:
                            subject.reconstruct_raw(client, response, references()[spec['shape']], spec['streaming'])
                        else:
                            subject.check_saved_wire_frames(client, response, spec['streaming'])
                        self.assertTrue(result['completed'])
                        if spec == {'shape': 'stop', 'streaming': True, 'raw': True}:
                            self.assertEqual(result['blank_generated_token_events'], 3)

    def test_mixed_c8_overlap_and_unique_request_ids(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); audits = root/'audit'; audits.mkdir()
            with serving(audits) as port, client.TokenHttpClient() as transport:
                def run_one(check_id, spec):
                    return subject.execute_check(client, transport, port, root, check_id, spec, 'prompt', references(), 'cpu', 3)
                results, summary = subject.mixed_wave(client, run_one, root, 8)
                self.assertEqual(summary['observed_max_request_in_flight'], 8)
                self.assertEqual(len({item['server_request_id'] for item in results}), 16)

    def test_mixed_failure_joins_workers_and_records_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root, returned = Path(name), []
            def run_one(check_id, spec):
                if 'w0' in check_id: raise ValueError('injected worker failure')
                start = time.perf_counter_ns(); time.sleep(.02); returned.append(check_id)
                return {'check_id': check_id, 'started_ns': start, 'finished_ns': time.perf_counter_ns()}
            with self.assertRaisesRegex(ValueError, 'wave incomplete'):
                subject.mixed_wave(client, run_one, root, 2)
            record = subject.read(root/'mixed-c2-accounting.json')
            self.assertFalse(record['completed']); self.assertTrue(record['failures']); self.assertEqual(len(returned), 2)
            self.assertFalse(any(thread.name.startswith('token-correctness-wave') for thread in threading.enumerate()))

    def test_timeout_bounds_http10_and_http11_trickle_with_partial_bytes(self):
        for version in ('HTTP/1.0', 'HTTP/1.1'):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as name:
                with serving(Path(name), behavior='trickle', version=version) as port:
                    started = time.monotonic()
                    row = subject.wire_request(client, port, subject.payload('prompt', {'shape': 'o32', 'streaming': True, 'raw': True}), .12)
                    self.assertLess(time.monotonic()-started, 1)
                    self.assertIsNotNone(row['error']); self.assertTrue(row['deadline_fired'] or 'deadline' in row['error']['message'] or row['error']['type'] == 'TimeoutError')
                    self.assertTrue(row['owned_connection_closed']); self.assertTrue(base64.b64decode(row['raw_response_base64']))

    def test_header_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as name, serving(Path(name), behavior='headers_timeout') as port:
            start = time.monotonic()
            row = subject.wire_request(client, port, subject.payload('prompt', {'shape': 'o1', 'streaming': False, 'raw': False}), .1)
            self.assertLess(time.monotonic()-start, .8); self.assertIsNotNone(row['error']); self.assertTrue(row['owned_connection_closed'])

    def test_both_disconnect_boundaries_followed_by_exact_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); audits = root/'audit'; audits.mkdir()
            with serving(audits) as port, client.TokenHttpClient() as transport:
                for index, boundary in enumerate(('after_headers', 'after_first_committed')):
                    spec = {'shape': 'o32', 'streaming': True, 'raw': True}
                    row = subject.wire_request(client, port, subject.payload('prompt', spec), 2, disconnect=boundary)
                    self.assertTrue(row['disconnect_observed']); self.assertIsNone(row['error']); self.assertTrue(row['owned_connection_closed'])
                    result = subject.execute_check(client, transport, port, root, f'reuse-{index}', spec, 'prompt', references(), 'cpu', 2)
                    self.assertTrue(result['completed'])

    def test_wire_cleanup_error_is_preserved(self):
        class Connection:
            sock = None
            def __init__(self, *args, **kwargs): pass
            def connect(self): raise OSError('connect fails')
            def close(self): raise OSError('close fails')
        with patch.object(subject.http.client, 'HTTPConnection', Connection):
            row = subject.wire_request(client, 1, {}, .1)
        self.assertFalse(row['owned_connection_closed']); self.assertEqual(row['error']['message'], 'connect fails')
        self.assertEqual(row['cleanup_errors'][0]['message'], 'close fails')

    def test_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'bad.json'
            for body in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(body)
                with self.assertRaises(ValueError): subject.read(path)


class LifecycleContracts(unittest.TestCase):
    def test_owned_child_termination_and_forced_fallback(self):
        class Process:
            pid = 1234
            returncode = None
            def poll(self): return self.returncode
            def wait(self, timeout):
                if timeout == 30: raise subject.subprocess.TimeoutExpired('mock', timeout)
                self.returncode = -9
        with patch.object(subject.os, 'killpg') as kill, patch.object(subject, 'owned_members', return_value=[]):
            subject.stop_owned(Process())
            self.assertEqual([call.args for call in kill.call_args_list], [(1234, subject.signal.SIGTERM), (1234, subject.signal.SIGKILL)])

    def test_remaining_owned_member_fails_closed(self):
        class Process:
            pid = 1234
            def poll(self): return 0
        with patch.object(subject, 'owned_members', return_value=[1235]):
            with self.assertRaisesRegex(ValueError, 'remains'): subject.stop_owned(Process())



class SamplerLifecycle(unittest.TestCase):
    def test_complete_sampler_and_injected_request_failure_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as name:
                root = Path(name); output = root/'output'; output.mkdir()
                env = dict.fromkeys(('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CARGO_TARGET_DIR'), 'test')
                cleanup, verify_calls, servers = [], [], []
                class Process:
                    pid, returncode = 42, None
                    def __init__(self, *args, **kwargs): pass
                    def poll(self): return self.returncode
                    def terminate(self): pass
                    def wait(self, timeout): self.returncode = 0
                def ready(module, process, port):
                    directory = output/'cpu'
                    runtime = RUNTIME
                    endpoint = {'error': {'code': 'config_unavailable', 'message': 'runtime configuration is unavailable'}}
                    subject.write(directory/'startup.json', {'schema_version': 'riley.effective-runtime-config-startup-artifact.v1',
                        'candidate_id': subject.CANDIDATE, 'runtime_identity': runtime, 'endpoint_path': '/v1/config',
                        'endpoint_payload': endpoint, 'endpoint_payload_sha256': subject.canonical_sha(endpoint)})
                    server = serving(directory/'audit', port=port, endpoint=endpoint, runtime=runtime)
                    server.__enter__(); servers.append(server)
                def close(process):
                    cleanup.append(process.pid); process.returncode = 0
                original_execute = subject.execute_check
                def execute(*args, **kwargs):
                    if fail and args[4] == 'fixed-04': raise ValueError('injected request failure')
                    return original_execute(*args, **kwargs)
                try:
                    with patch.object(subject.subprocess, 'Popen', Process), patch.object(subject, 'process_start', return_value=99), \
                         patch.object(subject, 'wait_ready', side_effect=ready), patch.object(subject, 'driver_maps', return_value={'mock': True}), \
                         patch.object(subject, 'launch_identity', side_effect=lambda process, argv, cwd: {**IDENTITY, 'argv': argv, 'cwd': str(cwd), 'actual_process_identity_verified': True}), \
                         patch.object(subject, 'stop_owned', side_effect=close), patch.object(subject, 'owned_members', return_value=[]), \
                         patch.object(subject, 'execute_check', side_effect=execute):
                        if fail:
                            with self.assertRaisesRegex(ValueError, 'injected request failure'):
                                subject.run_sampler(client, root, output, 'cpu', env, {}, 'prompt', references(), lambda: verify_calls.append(1), 3)
                            self.assertFalse((output/'cpu/completion.json').exists())
                            self.assertEqual(len(verify_calls), 1)
                        else:
                            result = subject.run_sampler(client, root, output, 'cpu', env, {}, 'prompt', references(), lambda: verify_calls.append(1), 3)
                            self.assertEqual(len(result['checks']), 46)
                            self.assertEqual([wave['concurrency'] for wave in result['mixed_waves']], [1, 2, 4, 8])
                            self.assertEqual(len(verify_calls), 2)
                        self.assertEqual(cleanup, [42])
                        self.assertTrue(subject.read(output/'cpu/process-exit.json')['cleanup_verified'])
                finally:
                    for server in servers: server.__exit__(None, None, None)


class ReceiptValidation(unittest.TestCase):
    def test_complete_92_check_receipt_reconstructs_then_rejects_changed_token(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); output = root/'output'; output.mkdir()
            env = dict.fromkeys(('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CARGO_TARGET_DIR'), 'test')
            servers = []
            class Process:
                pid, returncode = 42, None
                def __init__(self, *args, **kwargs): pass
                def poll(self): return self.returncode
                def terminate(self): pass
                def wait(self, timeout): self.returncode = 0
            def ready(module, process, port):
                directory = next(item for item in output.iterdir() if item.is_dir() and not (item/'startup.json').exists())
                runtime = RUNTIME
                endpoint = {'error': {'code': 'config_unavailable', 'message': 'runtime configuration is unavailable'}}
                subject.write(directory/'startup.json', {'schema_version': 'riley.effective-runtime-config-startup-artifact.v1',
                    'candidate_id': subject.CANDIDATE, 'runtime_identity': runtime, 'endpoint_path': '/v1/config',
                    'endpoint_payload': endpoint, 'endpoint_payload_sha256': subject.canonical_sha(endpoint)})
                server = serving(directory/'audit', directory.name, port=port, endpoint=endpoint, runtime=runtime)
                server.__enter__(); servers.append(server)
            try:
                with patch.object(subject.subprocess, 'Popen', Process), patch.object(subject, 'process_start', return_value=99), \
                     patch.object(subject, 'wait_ready', side_effect=ready), patch.object(subject, 'driver_maps', return_value={'mock': True}), \
                     patch.object(subject, 'owned_members', return_value=[]), \
                     patch.object(subject, 'stop_owned', side_effect=lambda process: setattr(process, 'returncode', 0)), \
                     patch.object(subject, 'launch_identity', side_effect=lambda process, argv, cwd: {**IDENTITY, 'argv': argv, 'cwd': str(cwd), 'actual_process_identity_verified': True}):
                    results = [subject.run_sampler(client, root, output, sampler, env, {}, 'prompt', references(), lambda: None, 3)
                               for sampler in subject.SAMPLERS]
                refs = {key: {'model': ref.model, 'prompt_token_ids': list(ref.prompt_token_ids), 'output_token_ids': list(ref.output_token_ids),
                              'text': ref.text, 'finish_reason': ref.finish_reason} for key, ref in references().items()}
                subject.write(root/'preparation.json', {'references': refs, 'stop_reference_count': 3,
                    'stop_fixture': {'text': subject.STOP_TEXT, 'token_ids': list(subject.STOP_TOKEN_IDS),
                                     'decoded_prefixes': list(subject.STOP_PREFIXES)}})
                subject.write(root/'request.json', {'prompt': 'prompt', 'prompt_token_ids': list(references()['o32'].prompt_token_ids)})
                subject.write(root/'binding.json', {'input_token_ids': list(references()['o32'].prompt_token_ids),
                                                   'generated_token_ids': list(references()['o32'].output_token_ids)})
                (root/'binary').write_bytes(b'CPU mock, not a GPU binary')
                helpers = {str(CLIENT.resolve()): subject.CLIENT_SHA, str(Path(subject.__file__).resolve()): subject.shared.sha(subject.__file__)}
                result = {'schema_version': subject.SCHEMA, 'completed': True, 'gpu_tests_executed': True,
                    'performance_claim': False, 'source_commit': subject.EXPECTED_COMMIT, 'binary': subject.evidence(root/'binary'),
                    'sampling_backends': list(subject.SAMPLERS), 'offered_concurrencies': list(subject.CONCURRENCIES), 'active_capacity': 1,
                    'proofs': dict.fromkeys(subject.PROOFS, True), 'per_sampler': results,
                    'checks': [check for sampler in results for check in sampler['checks']], 'helpers': helpers,
                    'model_files': {str(root/'binary'): subject.shared.sha(root/'binary')}, 'client_module': subject.evidence(CLIENT),
                    'preparation': subject.evidence(root/'preparation.json'), 'request': subject.evidence(root/'request.json'),
                    'reference_binding': subject.evidence(root/'binding.json')}
                subject.write(root/'build.json', {'source_root': str(root)})
                subject.write(root/'gpu.json', {'server_unit_tests': {}})
                result.update(source_build=subject.evidence(root/'build.json'), prerequisites=[subject.evidence(root/'gpu.json')],
                              source_commit_path={'mock': True}, independent_per_request_committed_audit=False, c02_enabled=False)
                with patch.object(subject, 'EXPECTED_BINARY', subject.shared.sha(root/'binary')), \
                     patch.object(subject, 'source_commit_contract', return_value={'mock': True}), \
                     patch.object(subject, 'validate_full_gpu_receipt', return_value={'server_unit_tests': {}}):
                    self.assertIs(subject.validate_result(result), result)
                    subject.write(root/'completion.json', result)
                    self.assertTrue(subject.validate_completion(root/'completion.json')['completed'])
                    item = result['per_sampler'][0]['checks'][1]
                    path = Path(item['response']['path']); row = subject.read(path)
                    row['token_ids'][0] += 1; path.write_text(json.dumps(row))
                    item['response'] = subject.evidence(path)
                    with self.assertRaisesRegex(ValueError, 'raw response reconstruction'):
                        subject.validate_result(result)
            finally:
                for server in servers: server.__exit__(None, None, None)


class SourceScopeContracts(unittest.TestCase):
    def test_source_path_and_required_test_names_fail_closed(self):
        source = HERE.parents[2]
        build = {'source_files': {name: subject.shared.sha(source/name) for name in subject.HTTP_FILES}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); log = root/'server.log'; gpu = root/'gpu.json'; gpu.write_text('{}')
            log.write_text(''.join('test engine::tests::'+name+' ... ok\n' for name in subject.SOURCE_TESTS)
                           +'test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n')
            counts = {'passed_tests': 5, 'failed_tests': 0, 'ignored_tests': 0}
            receipt = {'path': str(log), 'sha256': subject.shared.sha(log), 'passed': True, **counts}
            proof = subject.source_commit_contract(source, build, receipt, gpu)
            self.assertFalse(proof['independent_per_request_committed_audit']); self.assertFalse(proof['c02_enabled'])
            log.write_text(log.read_text().replace(subject.SOURCE_TESTS[0], 'unrelated_test'))
            receipt['sha256'] = subject.shared.sha(log)
            with self.assertRaisesRegex(ValueError, 'required committed-token unit tests'):
                subject.source_commit_contract(source, build, receipt, gpu)

    def test_full_gpu_receipt_rejects_wrong_source_before_log_use(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for leaf in ('gpu.json', 'build.json', 'binding.json'): (root/leaf).write_text('{}')
            receipt = {'schema_version': 'riley.http-token-gpu-correctness.v1', 'passed': True, 'gpu_tests_executed': True,
                       'full_logits_and_kv_exact': True, 'vllm_reference_tokens_exact': True, 'source_clean': True,
                       'source_commit': 'wrong'}
            (root/'gpu.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'identity/scope differs'):
                subject.validate_full_gpu_receipt({'source_commit': subject.EXPECTED_COMMIT}, root/'build.json', root/'gpu.json', root/'binding.json')

    def test_v3_does_not_enable_or_claim_c02(self):
        import inspect
        self.assertNotIn('--c02-', inspect.getsource(subject.run_sampler))
        self.assertNotIn('published_ids_match_committed_audit', subject.PROOFS)
        self.assertNotIn('c02_audit_complete', subject.PROOFS)
        self.assertIn('source_commit_path_reviewed_and_unit_tested', subject.PROOFS)
        self.assertIn('existing_full_gpu_logits_kv_exact', subject.PROOFS)


if __name__ == '__main__': unittest.main()
