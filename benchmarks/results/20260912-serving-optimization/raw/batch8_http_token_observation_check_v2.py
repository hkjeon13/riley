#!/usr/bin/env python3
"""Fresh 92-case HTTP token correctness for the qualified Batch8 build.

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
QUALIFIER_SHA = 'ec00a9beabffa94a9d80b69b9eeda8a28cd7ad889202018b385ea7b1761b0790'
FUSION_HELPER_SHA = 'c025519755c4ce73bf61d2aa80a3fc36ec6ae745843fb0995058c312a8b74904'
SCHEMA = 'riley.batch8-http-token-observation-correctness.v1'
EXPECTED_COMMIT = '8329c1aeec6e013f581128888c536e15f8bf7300'
EXPECTED_BINARY = '4c0876dbe080920bcc0d689df16d52b7714f31aa625b603fb9f7f1acc760c741'
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
qualifier = pinned_module('qualify_batch8.py', 'qualify_batch8', QUALIFIER_SHA)
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
    files = {'http_token_observation_check_v3.py': PARENT_SHA, 'qualify_batch8.py': QUALIFIER_SHA,
             'batch8_fusion_probe.py': FUSION_HELPER_SHA}
    require(all(shared.sha(HERE/name) == digest for name, digest in files.items()), 'frozen child dependency changed')
    return {str(HERE/name): digest for name, digest in files.items()}


def validate_prerequisites(root, base):
    """Read-only, fresh reconstruction of model/fusion/build and HTTP lineage."""
    dependency_evidence()
    build = qualifier.validate_build(root)
    require(build['source_commit'] == EXPECTED_COMMIT and build['source_root'] == str(root/'batch8-source')
            and build['binaries'][str(root/'batch8-target/release/riley')] == EXPECTED_BINARY,
            'Batch8 build identity differs')
    model = qualifier.validate_model(root, base, build)
    binding, refs = qualifier.reference(base)
    fusion = qualifier.validate_fusion(root, build, binding)
    path = root/'batch8-qualification.json'
    qualified = read(path)
    expected = {
        'schema_version': 'riley.batch8-fusion-model-qualification.v1', 'passed': True,
        'source_root': build['source_root'], 'source_commit': build['source_commit'], 'source_files': build['source_files'],
        'binaries': build['binaries'], 'source_clean': True, 'implementation_id': qualifier.IMPLEMENTATION,
        'numerical_profile': qualifier.PROFILE, 'correctness_gate_id': qualifier.GATE, 'graph_implementation_signature': '0xF108',
        'build': evidence(root/'batch8-build.json'), 'model_tests': evidence(root/'batch8-model-tests.json'),
        'fusion_probe': evidence(root/'batch8-fusion-probe/receipt.json'), 'references': refs,
        'gpu_tests_executed_on_current_snapshot': True, 'full_logits_and_kv_exact': True,
        'fusion_cases_exact': fusion['cases'], 'runtime': model['runtime'], 'gpu': model['gpu_after'],
        'source_lineage': {'parent_commit': qualifier.PARENT, 'parent_build': evidence(root/'http-token-build.json'),
            'overlay': evidence(root/'batch8-source-overlay.json'), 'changed_files': sorted(qualifier.CHANGED),
            'inherited_http_files': {name: build['source_files'][name] for name in sorted(qualifier.HTTP_FILES)},
            'other_parent_tracked_files_unchanged': True},
        'helpers': model['helpers'], 'fusion_validator': evidence(root/'batch8_fusion_probe.py'),
        'http_correctness_qualified': False, 'serving_performance_qualified': False,
        'performance_measured': False, 'performance_claim_eligible': False,
        'performance_metrics': {'throughput': None, 'ttft': None, 'tpot': None, 'p95': None, 'p99': None},
    }
    require(qualified == expected, 'fresh Batch8 qualification receipt differs from reconstructed evidence')
    parent = read(root/'http-token-build.json')
    require(set(qualified['source_lineage']['inherited_http_files']) == qualifier.HTTP_FILES
            and all(build['source_files'][name] == parent['source_files'][name]
                    == shared.sha(root/'batch8-source'/name) for name in qualifier.HTTP_FILES),
            'four inherited HTTP source files differ from qualified parent')
    # The pinned qualifier validates runtime symlinks and file hashes while
    # preserving receipt aliases (for example libcuda.so.1). V3 evidence()
    # canonicalizes paths, so its generic artifact comparison is incompatible
    # with this already reconstructed, independently validated receipt.
    return build, {'qualification': evidence(path), 'model_tests': evidence(root/'batch8-model-tests.json'),
                   'fusion_probe': evidence(root/'batch8-fusion-probe/receipt.json'),
                   'inherited_http_files': qualified['source_lineage']['inherited_http_files']}


def run_sampler(module, root, output, sampler, env, compute, prompt, references, verify, timeout):
    verify()
    directory = output/sampler
    directory.mkdir(mode=0o700)
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0)); port = reserved.getsockname()[1]
    binary, cwd = root/'batch8-target/release/riley', root/'batch8-source'
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
    """Use the existing Batch8 model validator behind V3's prerequisite interface."""
    root, base = Path(build_path).parent, Path(binding_path).parent
    require(Path(build_path) == root/'batch8-build.json' and Path(gpu_path) == root/'batch8-model-tests.json'
            and Path(binding_path) == base/'native-binding.json', 'child prerequisite path differs')
    require(build['source_commit'] == EXPECTED_COMMIT and build['binaries'][str(root/'batch8-target/release/riley')] == EXPECTED_BINARY,
            'child prerequisite build differs')
    model = qualifier.validate_model(root, base, build)
    return {'server_unit_tests': model['server_lib_tests']}


contract.validate_full_gpu_receipt = validate_child_full_gpu_receipt



def validate_result(result):
    require(result['contract_helper'] == evidence(HERE/'http_token_observation_check_v3.py')
            and result['contract_helper']['sha256'] == PARENT_SHA, 'child contract dependency differs')
    root = Path(result['source_build']['path']).parent
    base = Path(result['reference_binding']['path']).parent
    build, prerequisites = validate_prerequisites(root, base)
    require(result['source_build'] == evidence(root/'batch8-build.json') and result['batch8_qualification'] == prerequisites
            and result['prerequisites'] == [prerequisites['model_tests'], prerequisites['qualification'], prerequisites['fusion_probe']]
            and result['binary'] == evidence(root/'batch8-target/release/riley')
            and result['source_commit'] == build['source_commit'], 'saved child build/qualification differs')
    contract.validate_result(result)
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
    output = root/'batch8-http-token-observation-correctness'
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
    build_path = root/'batch8-build.json'
    model = read(root/'batch8-model-tests.json')
    source_proof = contract.source_commit_contract(root/'batch8-source', build, model['server_lib_tests'], root/'batch8-model-tests.json')
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
    env['CARGO_TARGET_DIR'] = str(root/'batch8-target')
    compute, sessions, gpu_query = contract.runtime_and_sessions(env, binding)
    helper_paths = [Path(__file__), Path(shared.__file__), Path(runtime_check.__file__), Path(runtime_check.session.__file__),
                    root/'remote_session.py', client_path, Path(qualifier.frozen.__file__), Path(qualifier.prior.__file__)]
    helpers = {**dependency_evidence(), **{str(path.resolve()): shared.sha(path) for path in helper_paths}}
    pinned_paths = [build_path, root/'batch8-qualification.json', root/'batch8-model-tests.json',
                    root/'batch8-fusion-probe/receipt.json', binding_path, request_path,
                    runtime_check.COMPUTE/'receipt.json', runtime_check.GUI/'receipt.json']
    pins = {str(path): shared.sha(path) for path in pinned_paths}
    def verify():
        require(all(shared.sha(path) == digest for path, digest in {**pins, **helpers, **model_files}.items()), 'qualification input/helper/model changed')
        require(validate_prerequisites(root, base) == (build, prerequisites), 'Batch8 source/qualification changed')
        require(contract.source_commit_contract(root/'batch8-source', build, model['server_lib_tests'], root/'batch8-model-tests.json')
                == source_proof, 'child source publication proof changed')
        current_compute, current_sessions, current_query = contract.runtime_and_sessions(env, binding)
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
                  'binary': evidence(root/'batch8-target/release/riley'), 'request': evidence(request_path),
                  'reference_binding': evidence(binding_path), 'batch8_qualification': prerequisites,
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
