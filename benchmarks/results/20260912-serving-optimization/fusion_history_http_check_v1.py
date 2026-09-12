#!/usr/bin/env python3
"""Fresh 92-case HTTP token correctness for fusion-history-v1.

Imports the pinned V3 wire/source-path/receipt contract. Only the sampler's source and
binary paths are adapted; all cases, deadlines, proof boundaries and cleanup are retained.
No Blender controls or GPU activity occurs at import or --validate-only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

PARENT_SHA = '5ce121cf81e9251592f972d8105e0dd636bf9af4c237ca2e1d4555d4477804f0'
QUALIFIER_SHA = '8c22c9baf301d60de61eff20c8ffca9daeb5f18e48d608df45b4becba0213ade'
FUSION_HELPER_SHA = 'c025519755c4ce73bf61d2aa80a3fc36ec6ae745843fb0995058c312a8b74904'
SCHEMA = 'riley.fusion-history-http-correctness-v1.v1'
EXPECTED_COMMIT = '4c5bcff43d1942fdd3c396b2b9bcd9df3bf63593'
EXPECTED_BINARY = '89e57efff9f13af3d5a3b01cfec7359c9eb0788c094320a6c546418cdd10ae67'
HERE = Path(__file__).resolve().parent


def pinned_module(filename, module_name, expected):
    path = HERE/filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('frozen dependency differs: '+filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contract = pinned_module('http_token_observation_check_v3.py', '_batch8_http_v3_contract', PARENT_SHA)
# These identity values belong only to this isolated module instance. Frozen
# helper files and independently imported baseline validators remain untouched.
contract.SCHEMA, contract.EXPECTED_COMMIT, contract.EXPECTED_BINARY = SCHEMA, EXPECTED_COMMIT, EXPECTED_BINARY
qualifier = pinned_module('qualify_fusion_history_v1.py', 'qualify_fusion_history_v1', QUALIFIER_SHA)
shared, runtime_check = contract.shared, contract.runtime_check
require, read, write, evidence = contract.require, contract.read, contract.write, contract.evidence
canonical_sha, load_client = contract.canonical_sha, contract.load_client
SAMPLERS, CONCURRENCIES, CANDIDATE = contract.SAMPLERS, contract.CONCURRENCIES, contract.CANDIDATE
PROOFS, CLIENT_SHA = contract.PROOFS, contract.CLIENT_SHA
STOP_TEXT, STOP_TOKEN_IDS, STOP_PREFIXES = contract.STOP_TEXT, contract.STOP_TOKEN_IDS, contract.STOP_PREFIXES
check_specs, execute_check, mixed_wave = contract.check_specs, contract.execute_check, contract.mixed_wave
payload, wire_request = contract.payload, contract.wire_request
launch_identity, check_config_unavailable = contract.launch_identity, contract.check_config_unavailable
wait_ready, driver_maps, process_start = contract.wait_ready, contract.driver_maps, contract.process_start
stop_owned, owned_members = contract.stop_owned, contract.owned_members


def dependency_evidence():
    files = {'http_token_observation_check_v3.py': PARENT_SHA, 'qualify_fusion_history_v1.py': QUALIFIER_SHA,
             'batch8_fusion_probe.py': FUSION_HELPER_SHA}
    require(all(shared.sha(HERE/name) == digest for name, digest in files.items()), 'frozen child dependency changed')
    return {str(HERE/name): digest for name, digest in files.items()}


def validate_prerequisites(root, base):
    dependency_evidence()
    require(root == qualifier.ROOT and base == qualifier.frozen.BASE, 'candidate root differs')
    subprocess.run([sys.executable, str(HERE/'qualify_fusion_history_v1.py'), 'validate'], check=True, stdout=subprocess.DEVNULL)
    context, binding = qualifier.inputs()
    build = read(root/'fusion-history-build-v1.json')
    require(build['source_commit'] == EXPECTED_COMMIT and build['binaries'][str(root/'fusion-history-target-v1/release/riley')] == EXPECTED_BINARY, 'candidate identity differs')
    return build, {'qualification': evidence(root/'fusion-history-model-tests-v1.json'),
        'model_tests': evidence(root/'fusion-history-model-tests-v1.json'),
        'fusion_probe': evidence(root/'fusion-history-probe-v1/receipt.json'),
        'inherited_http_files': {name: build['source_files'][name] for name in qualifier.frozen.HTTP_FILES}}


def run_sampler(module, root, output, sampler, env, compute, prompt, references, verify, timeout):
    verify()
    directory = output/sampler
    directory.mkdir(mode=0o700)
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0)); port = reserved.getsockname()[1]
    binary, cwd = root/'fusion-history-target-v1/release/riley', root/'fusion-history-source-v1'
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



def validate_child_full_gpu_receipt(build, build_path, gpu_path, binding_path):
    root, base = Path(build_path).parent, Path(binding_path).parent
    require(Path(build_path) == root/'fusion-history-build-v1.json' and Path(gpu_path) == root/'fusion-history-model-tests-v1.json', 'candidate model paths differ')
    checked, _ = validate_prerequisites(root, base)
    require(checked == build, 'candidate build differs')
    return {'server_unit_tests': model_receipt(root)['server_lib_tests']}


contract.validate_full_gpu_receipt = validate_child_full_gpu_receipt



def validate_result(result):
    require(result['contract_helper'] == evidence(HERE/'http_token_observation_check_v3.py')
            and result['contract_helper']['sha256'] == PARENT_SHA, 'child contract dependency differs')
    root = Path(result['source_build']['path']).parent
    base = Path(result['reference_binding']['path']).parent
    build, prerequisites = validate_prerequisites(root, base)
    require(result['source_build'] == evidence(root/'fusion-history-build-v1.json') and result['candidate_qualification'] == prerequisites
            and result['prerequisites'] == [prerequisites['model_tests'], prerequisites['qualification'], prerequisites['fusion_probe']]
            and result['binary'] == evidence(root/'fusion-history-target-v1/release/riley')
            and result['source_commit'] == build['source_commit'], 'saved child build/qualification differs')
    contract.validate_result(result)
    return result


def validate_completion(path):
    return validate_result(read(path))


def model_receipt(root):
    model = read(root/'fusion-history-model-tests-v1.json')
    item = next(x for x in model['checks'] if x['name'] == 'server-lib-tests')
    return {**model, 'server_lib_tests': {**item['log'], **item['counts'], 'passed': True}}


def runtime_and_sessions(env, binding):
    # Explicit Round15 successor check; never substitute an old restoration proof.
    import remote_session_round15 as session15
    compute = runtime_check.check_runtime(runtime_check.COMPUTE)
    runtime_check.check_runtime(runtime_check.GUI)
    require(Path('/proc/driver/nvidia/version').read_text() == compute['kernel_version'], 'kernel changed')
    query = subprocess.check_output([compute['nvidia_smi_path'], '-i', '0', '--query-gpu=uuid,driver_version', '--format=csv,noheader'], env=env, text=True).strip()
    require(query == binding['environment']['gpu']['uuid']+', 580.173.02', 'GPU changed')
    root = qualifier.ROOT
    require(shared.sha(root/'blender-round15/verified.json') == '39487322b5ca6cfd41d8a87892a47259ff901455f12222d890a9bc6261274d92', 'Round15 proof changed')
    require(shared.sha(root/'remote_session_round15.py') == 'a7e8b194bd0d0bccd4ef0631dc8b295a718d8478266882cc0eeebb2716a222d2', 'Round15 helper changed')
    previous = read(root/'blender-round15/session.json')
    restored = read(root/'blender-round15/verified.json')
    require(len(previous) == len(restored['processes']) == 3, 'session count differs')
    sessions = []
    for original, row in zip(previous, restored['processes']):
        current = session15.identity(row['new_pid'])
        require(current['start'] == row['start'] and session15.live(row['new_pid']), 'successor changed')
        require(session15.same_command(current, original) and session15.process_tag(row['new_pid']) == row['tag'] and session15.listening(row['port']), 'successor command or port changed')
        sessions.append({'pid': row['new_pid'], 'start': row['start'], 'port': row['port'], 'command_and_gui_environment_verified': True})
    return compute, sessions, query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=shared.DEFAULT_ROOT)
    parser.add_argument('--base', type=Path, default=shared.DEFAULT_BASE)
    parser.add_argument('--client-module', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=30)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root, base = args.root.resolve(strict=True), args.base.resolve(strict=True)
    output = root/'fusion-history-http-correctness-v1'
    if args.validate_only:
        validate_completion(output/'completion.json')
        print(json.dumps({'completed': True, 'saved_proof_valid': True, 'gpu_tests_executed_now': False}))
        return
    require(1 <= args.timeout_seconds <= 120, 'timeout outside client bounds')
    client_path = (args.client_module or root/'serving_token_client.py').resolve(strict=True)
    require(shared.sha(client_path) == CLIENT_SHA, 'token client differs from frozen validated helper')
    module = load_client(client_path)
    from tokenizers import Tokenizer
    build, prerequisites = validate_prerequisites(root, base)
    build_path = root/'fusion-history-build-v1.json'
    model = model_receipt(root)
    source_proof = contract.source_commit_contract(root/'fusion-history-source-v1', build, model['server_lib_tests'], root/'fusion-history-model-tests-v1.json')
    binding_path, request_path = base/'native-binding.json', base/'request.json'
    binding, request = read(binding_path), read(request_path)
    model_files = {str(shared.MODEL/name): binding['workload'][key] for name, key in
                   (('model.safetensors', 'weights_sha256'), ('tokenizer.json', 'tokenizer_sha256'))}
    require(all(shared.sha(path) == digest for path, digest in model_files.items()), 'model files differ from reference')
    tokenizer = Tokenizer.from_file(str(shared.MODEL/'tokenizer.json'))
    prompt, prompt_ids, generated = request['prompt'], binding['input_token_ids'], binding['generated_token_ids']
    require(tokenizer.encode(prompt, add_special_tokens=True).ids == prompt_ids == request['prompt_token_ids']
            and len(prompt_ids) == 128 and len(generated) == request['requested_output_tokens'] == 32, 'fixed P128/O32 reference differs')
    stop_count, stop_text = contract.stop_reference(tokenizer, generated)
    references = {name: module.TokenReference('g04-smol', tuple(prompt_ids), tuple(generated[:count]),
                  tokenizer.decode(generated[:count], skip_special_tokens=True) if name != 'stop' else stop_text,
                  'stop' if name == 'stop' else 'length') for name, count in (('o32', 32), ('o1', 1), ('stop', stop_count))}
    env = {key: value for key, value in shared.environment(root).items() if re.fullmatch(r'[A-Z_][A-Z0-9_]*', key)
           and key not in ('RILEY_FREEZE_SHA', 'RILEY_GATE_E_REPORT_SHA', 'RILEY_CONFIGURATION_SHA', 'RILEY_BASE_RELEASE_CANDIDATE_REPORT_SHA')}
    env['CARGO_TARGET_DIR'] = str(root/'fusion-history-target-v1')
    compute, sessions, gpu_query = runtime_and_sessions(env, binding)
    helper_paths = [Path(__file__), Path(shared.__file__), Path(runtime_check.__file__), Path(runtime_check.session.__file__),
                    root/'remote_session.py', client_path, Path(qualifier.frozen.__file__), Path(qualifier.frozen.prior.__file__)]
    helpers = {**dependency_evidence(), **{str(path.resolve()): shared.sha(path) for path in helper_paths}}
    pinned_paths = [build_path, root/'fusion-history-model-tests-v1.json', root/'fusion-history-model-tests-v1.json',
                    root/'fusion-history-probe-v1/receipt.json', binding_path, request_path,
                    runtime_check.COMPUTE/'receipt.json', runtime_check.GUI/'receipt.json']
    pins = {str(path): shared.sha(path) for path in pinned_paths}
    def verify():
        require(all(shared.sha(path) == digest for path, digest in {**pins, **helpers, **model_files}.items()), 'qualification input/helper/model changed')
        require(validate_prerequisites(root, base) == (build, prerequisites), 'Batch8 source/qualification changed')
        require(contract.source_commit_contract(root/'fusion-history-source-v1', build, model['server_lib_tests'], root/'fusion-history-model-tests-v1.json')
                == source_proof, 'child source publication proof changed')
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
          'offered_concurrencies': list(CONCURRENCIES), 'active_capacity': 1, 'performance_claim': False})
    try:
        results = [run_sampler(module, root, output, sampler, env, compute, prompt, references, verify, args.timeout_seconds) for sampler in SAMPLERS]
        verify()
        require(all(any(item.get('blank_generated_token_events', 0) == 3 and item.get('spec', {}).get('shape') == 'stop'
                        and item['spec']['streaming'] and item['spec']['raw'] for item in row['checks']) for row in results), 'both samplers must prove blank commits')
        result = {'schema_version': SCHEMA, 'completed': True, 'gpu_tests_executed': True, 'performance_claim': False,
                  'source_build': evidence(build_path), 'source_commit': EXPECTED_COMMIT,
                  'binary': evidence(root/'fusion-history-target-v1/release/riley'), 'request': evidence(request_path),
                  'reference_binding': evidence(binding_path), 'candidate_qualification': prerequisites,
                  'prerequisites': [prerequisites['model_tests'], prerequisites['qualification'], prerequisites['fusion_probe']],
                  'source_commit_path': source_proof, 'independent_per_request_committed_audit': False, 'c02_enabled': False,
                  'contract_helper': evidence(HERE/'http_token_observation_check_v3.py'),
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
