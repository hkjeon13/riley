#!/usr/bin/env python3
"""Run fresh batch8 model checks, then qualify their exact source and fusion proof.

`run` executes four CUDA tests plus server-library/profile tests with the privately
verified driver. `validate` rechecks their evidence and the independent fusion
probe and writes an exclusive qualification. Neither command controls Blender,
serves HTTP, prepares serving plans, or makes performance claims.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import prepare_batch2_plan as prior
import run_batch7_tests as frozen

read, require, sha, evidence = prior.read, prior.require, prior.sha, prior.evidence
ROOT = Path('/tmp/riley-opt-260912')
BASE = Path('/tmp/riley-g04-vllm-profile-260911')
PARENT = 'a179617070526068b66ba5627ba82a7151da8c64'
GRANDPARENT = '1a2be0df01fe49daa4d4db155ad5c44f34ead6df'
PROFILE, GATE = 'vllm-smol-p128-v1', 'g04-vllm-smol-p128-v1'
IMPLEMENTATION = 'g12-fused-rope-attention-v1'
ENTRY = 'enqueue_compiled_packed_decode_rope_attention'
CHANGED = {'kernels/src/graph_numerics.cu', 'kernels/src/ffi_internal.hpp',
           'kernels/src/graph_resources.cu', 'crates/riley-runtime/src/llama/graph_decode_full.rs'}
HTTP_FILES = {f'crates/riley-server/src/{name}.rs' for name in ('domain', 'openai', 'engine', 'service')}
OVERLAY_SHA = 'e542bd579525a37576905d61e0361ce50fb2952bc688724a14df8872f1570e1f'
HTTP_OVERLAY_SHA = 'd8a03a06c3bf7f930d044cf082574fb9443ed8a3023b8df7dce495a525610f53'
RUNTIME_SHA = '36dba27991800124660e8f07274ffde21b79ebef14bb365d37a610f9a0802d70'
GPU_TESTS = frozen.GPU_TESTS
FLAGS = ('exact_outputs', 'finite_outputs', 'guards_intact', 'inputs_unchanged',
         'oracle_outputs_unchanged', 'inactive_kv_unchanged', 'current_values_raw_exact',
         'mapping_invariant', 'all_allocations_freed')


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(prior.encoded(value).decode())


def check_tree(source, commit, files):
    require(git(source, 'rev-parse', 'HEAD') == commit and
            not git(source, 'status', '--porcelain', '--untracked-files=all'), 'source is not the clean built commit')
    require(bool(files), 'empty source inventory')
    for name, digest in files.items():
        path = source / name
        require(not Path(name).is_absolute() and prior.below(path, source), 'source path escapes snapshot')
        require(sha(path) == digest, 'source changed: ' + name)


def exact_diff(source, parent, names):
    rows = git(source, 'diff', '--name-status', '--no-renames', parent, 'HEAD').splitlines()
    require(sorted(rows) == sorted('M\t' + name for name in names), 'tracked source diff is not the exact modified-file set')


def build_commands():
    return [['cargo', 'test', '--release', '-p', 'riley-runtime', '--features', 'cuda', '--lib', '--no-run'],
            ['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
             '--bin', 'riley', '--bin', 'riley-profile']]


def build_environment(root):
    return {'CUDA_HOME': '/data/riley-g04-cuda13', 'CUDAToolkit_ROOT': '/data/riley-g04-cuda13',
            'CMAKE': '/data/cmake-3.31.12/bin/cmake', 'CMAKE_BUILD_PARALLEL_LEVEL': '4',
            'CARGO_BUILD_JOBS': '4', 'CARGO_TARGET_DIR': str(root / 'batch8-target'),
            'LD_LIBRARY_PATH': '/data/riley-g04-cuda13/lib'}


def validate_build(root):
    build, parent = read(root / 'batch8-build.json'), read(root / 'http-token-build.json')
    previous = read(root / 'batch7-build.json')
    source, parent_source = root / 'batch8-source', root / 'http-token-source'
    require(sha(root / 'batch8-source-overlay.json') == OVERLAY_SHA, 'frozen fusion overlay changed')
    require(sha(root / 'http-token-source-after-v2.json') == HTTP_OVERLAY_SHA, 'frozen HTTP overlay changed')
    overlay, http_overlay = read(root / 'batch8-source-overlay.json'), read(root / 'http-token-source-after-v2.json')
    http_files = {row['path']: row['after_sha256'] for row in http_overlay['files']}
    require(set(overlay) == CHANGED and set(http_files) == HTTP_FILES and len(http_overlay['files']) == 4,
            'overlay scope differs')
    require(parent['source_root'] == str(parent_source) and parent['source_commit'] == PARENT
            and parent['parent_source_commit'] == GRANDPARENT and previous['source_commit'] == GRANDPARENT
            and parent['parent_build_sha256'] == sha(root / 'batch7-build.json')
            and parent['local_source_receipt_sha256'] == HTTP_OVERLAY_SHA, 'HTTP parent lineage differs')
    require(parent['source_files'] == {**previous['source_files'], **http_files}, 'HTTP parent source pins differ')
    check_tree(parent_source, PARENT, parent['source_files'])
    require(git(parent_source, 'rev-parse', 'HEAD^') == GRANDPARENT, 'HTTP parent is not direct child of accepted7')
    exact_diff(parent_source, GRANDPARENT, HTTP_FILES)
    require(parent['binaries'] == {str(root / 'http-token-target/release' / name): sha(root / 'http-token-target/release' / name)
                                   for name in ('riley', 'riley-profile')}, 'HTTP parent binaries changed')
    require(build['source_root'] == str(source) and build['parent_source_commit'] == PARENT
            and build['parent_build_sha256'] == sha(root / 'http-token-build.json'), 'batch8 parent differs')
    require(build['changed_files'] == sorted(CHANGED) and build['source_files'] == {**parent['source_files'], **overlay},
            'batch8 source inventory differs')
    check_tree(source, build['source_commit'], build['source_files'])
    require(git(source, 'rev-parse', 'HEAD^') == PARENT, 'batch8 is not a direct child of HTTP parent')
    exact_diff(source, PARENT, CHANGED)
    signature = (source / 'crates/riley-runtime/src/llama/graph_decode_full.rs').read_text()
    require(signature.count('let implementation = if packed.is_some() {\n            0xF108') == 1, 'F108 signature missing')
    native = 'kernels/src/graph_numerics.cu'
    require((source / native).read_bytes().startswith((parent_source / native).read_bytes()), 'accepted numerical prefix changed')
    require(build['build_argv'] == build_commands() and build['build_environment'] == build_environment(root),
            'production build command/environment differs')
    require(build['build_log_sha256'] == sha(root / 'batch8-build.log') and build['gpu_tests_executed'] is False
            and build['performance_measured'] is False, 'build log or compile-only scope differs')
    require(build['binaries'] == {str(root / 'batch8-target/release' / name): sha(root / 'batch8-target/release' / name)
                                  for name in ('riley', 'riley-profile')}, 'batch8 serving binaries changed')
    return build


def verify_runtime(root, library_dir):
    runtime_root = root / 'driver580173-runtime-20260901'
    path, extracted = runtime_root / 'receipt.json', runtime_root / 'extracted'
    require(sha(path) == RUNTIME_SHA, 'private driver receipt changed')
    receipt = read(path)
    require(receipt['completed'] is True and receipt['archive_signature_verified'] is True
            and receipt['host_packages_modified'] is False and receipt['host_restart_performed'] is False,
            'private driver provenance differs')
    require(runtime_root.is_dir() and not runtime_root.is_symlink() and extracted.is_dir() and not extracted.is_symlink(),
            'private runtime root is not a real directory')
    files, links = {}, {}
    for item in extracted.rglob('*'):
        name = str(item.relative_to(extracted))
        if item.is_symlink():
            links[name] = str(item.readlink())
            target = item.resolve(strict=True)
            require(target.is_file() and target.is_relative_to(extracted.resolve()), 'runtime symlink escaped private root')
        elif item.is_file():
            files[name] = sha(item)
        else:
            require(item.is_dir(), 'unexpected runtime filesystem entry')
    require(files == receipt['files'] and links == receipt['symlinks'], 'runtime file/symlink inventory changed')
    require(str(library_dir) == receipt['library_path'] == str(extracted / 'usr/lib/x86_64-linux-gnu'), 'driver library path differs')
    require(receipt['nvidia_smi_path'] == str(extracted / 'usr/bin/nvidia-smi'), 'private nvidia-smi path differs')
    require(re.search(r'\b580\.173\.02\b', Path('/proc/driver/nvidia/version').read_text()), 'loaded driver kernel differs')
    return {'receipt': evidence(path), 'library_directory': str(library_dir),
            'libcuda': evidence(library_dir / 'libcuda.so.1'), 'nvidia_smi': evidence(Path(receipt['nvidia_smi_path']))}


def reference(base):
    path = base / 'native-binding.json'
    binding = read(path)
    require(binding['source']['correctness_gate_id'] == GATE and len(binding['input_token_ids']) == 128
            and len(binding['generated_token_ids']) == 32, 'g04 fixed token binding differs')
    require(binding['workload']['prompt_tokens'] == 128 and binding['workload']['output_tokens'] == 32
            and binding['workload']['concurrency'] == 1, 'fixed reference workload differs')
    require(binding['environment']['gpu']['device_index'] == 0
            and binding['environment']['software']['cuda_runtime_version'] == '13.0'
            and binding['environment']['software']['nvidia_driver_version'] == '580.173.02', 'bound runtime differs')
    model = {str(frozen.MODEL / name): sha(frozen.MODEL / name)
             for name in ('model.safetensors', 'tokenizer.json')}
    require(model[str(frozen.MODEL / 'model.safetensors')] == binding['workload']['weights_sha256']
            and model[str(frozen.MODEL / 'tokenizer.json')] == binding['workload']['tokenizer_sha256'], 'model reference changed')
    return binding, {'binding': evidence(path), 'model_files': model}


def runtime_environment(root, library_dir):
    env = build_environment(root)
    env['LD_LIBRARY_PATH'] = str(library_dir) + ':/data/riley-g04-cuda13/lib'
    env.update(RILEY_REAL_CHECKPOINT=str(frozen.MODEL), CUDA_VISIBLE_DEVICES='0', CARGO_TERM_COLOR='never')
    return env


def gpu_identity(runtime, binding, env):
    argv = [runtime['nvidia_smi']['path'], '-i', '0', '--query-gpu=uuid,name,driver_version,pci.bus_id', '--format=csv,noheader']
    values = [value.strip() for value in subprocess.check_output(argv, text=True, env=env).strip().split(',')]
    require(len(values) == 4, 'GPU query shape differs')
    expected = binding['environment']['gpu']
    require(values == [expected['uuid'], expected['model'], '580.173.02', expected['pci_bus_id']], 'GPU binding identity differs')
    return {'uuid': values[0], 'name': values[1], 'driver_version': values[2], 'pci_bus_id': values[3], 'device_index': 0}


def commands():
    result = [(name, ['cargo', 'test', '--release', '-p', 'riley-runtime', '--features', 'cuda', '--lib', name,
                     '--', '--ignored', '--nocapture', '--test-threads=1'], True) for name in GPU_TESTS]
    result += [('server-lib-tests', ['cargo', 'test', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
                                     '--lib', '--', '--test-threads=1'], False),
               ('profile-unit-tests', ['cargo', 'test', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
                                       '--bin', 'riley-profile', '--', '--test-threads=1'], False)]
    return result


def helper_evidence():
    return {name: evidence(Path(module.__file__).resolve()) for name, module in
            (('runner', __import__(__name__)), ('frozen_test_contract', frozen), ('frozen_validation', prior))}


def run_checks(root, base, library_dir):
    root, base, library_dir = root.resolve(), base.resolve(), library_dir.absolute()
    output = root / 'batch8-model-tests.json'
    paths = [output] + [root / ('batch8-' + name + '.log') for name, _, _ in commands()]
    require(all(not path.exists() for path in paths), 'refusing to replace model evidence')
    require(not os.environ.get('LD_PRELOAD'), 'LD_PRELOAD would change the test runtime')
    build, pinned_build = validate_build(root), evidence(root / 'batch8-build.json')
    binding, refs = reference(base)
    runtime, helpers = verify_runtime(root, library_dir), helper_evidence()
    fixed = runtime_environment(root, library_dir)
    env = {**os.environ, **fixed}
    env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
    initial_gpu = gpu_identity(runtime, binding, env)
    def recheck():
        require(evidence(root / 'batch8-build.json') == pinned_build and validate_build(root) == build
                and reference(base) == (binding, refs) and verify_runtime(root, library_dir) == runtime
                and helper_evidence() == helpers, 'test inputs changed')
        require(gpu_identity(runtime, binding, env) == initial_gpu, 'GPU identity changed')
    results = []
    for name, argv, gpu in commands():
        recheck()
        path = root / ('batch8-' + name + '.log')
        with path.open('x', encoding='utf-8') as stream:
            subprocess.run(argv, cwd=root / 'batch8-source', env=env, stdout=stream, stderr=stream, check=True)
        recheck()
        counts = frozen.validate_test_log(path, name if gpu else None)
        results.append({'name': name, 'argv': argv, 'passed': True, **evidence(path), **counts})
    result = {'schema_version': 'riley.batch8-model-correctness.v1', 'passed': True,
              'source_root': build['source_root'], 'source_commit': build['source_commit'], 'source_clean': True,
              'source_files': build['source_files'], 'binaries': build['binaries'], 'build': pinned_build,
              'references': refs, 'helpers': helpers, 'runtime': runtime, 'runtime_environment': fixed,
              'gpu_before': initial_gpu, 'gpu_after': gpu_identity(runtime, binding, env),
              'checks': results[:4], 'server_lib_tests': results[4], 'profile_unit_tests': results[5],
              'gpu_tests_executed': True, 'vllm_reference_tokens_exact': True, 'full_logits_and_kv_exact': True,
              'desktop_session_policy': 'untouched by this helper', 'http_correctness_qualified': False,
              'performance_claim_eligible': False, 'performance_measured': False}
    write_new(output, result)
    return result


def validate_model(root, base, build):
    receipt = read(root / 'batch8-model-tests.json')
    require(receipt['schema_version'] == 'riley.batch8-model-correctness.v1' and receipt['passed'] is True
            and receipt['gpu_tests_executed'] is True and receipt['source_clean'] is True, 'model tests did not execute successfully')
    require(all(receipt[key] == build[key] for key in ('source_root', 'source_commit', 'source_files', 'binaries'))
            and receipt['build'] == evidence(root / 'batch8-build.json'), 'model build/source differs')
    binding, refs = reference(base)
    require(receipt['references'] == refs and receipt['helpers'] == helper_evidence(), 'reference/helper changed')
    library_dir = Path(receipt['runtime']['library_directory'])
    require(receipt['runtime'] == verify_runtime(root, library_dir)
            and receipt['runtime_environment'] == runtime_environment(root, library_dir), 'runtime evidence differs')
    gpu = binding['environment']['gpu']
    expected_gpu = {'uuid': gpu['uuid'], 'name': gpu['model'], 'driver_version': '580.173.02',
                    'pci_bus_id': gpu['pci_bus_id'], 'device_index': 0}
    require(receipt['gpu_before'] == receipt['gpu_after'] == expected_gpu, 'GPU evidence differs from reference')
    items = receipt['checks'] + [receipt['server_lib_tests'], receipt['profile_unit_tests']]
    require(len(receipt['checks']) == 4 and len(items) == len(commands()), 'model test count differs')
    for item, (name, argv, is_gpu) in zip(items, commands()):
        path = root / ('batch8-' + name + '.log')
        require(item['name'] == name and item['argv'] == argv and item['passed'] is True
                and {key: item[key] for key in ('path', 'sha256')} == evidence(path), 'test invocation/log changed')
        counts = frozen.validate_test_log(path, name if is_gpu else None)
        require(all(type(item[key]) is int and item[key] == value for key, value in counts.items()), 'test counts differ')
    require(receipt['vllm_reference_tokens_exact'] is True and receipt['full_logits_and_kv_exact'] is True
            and receipt['http_correctness_qualified'] is False and receipt['performance_claim_eligible'] is False
            and receipt['performance_measured'] is False, 'model receipt scope differs')
    return receipt


def validate_fusion(root, build, binding):
    import batch8_fusion_probe as probe
    path = root / 'batch8-fusion-probe/receipt.json'
    result = probe.validate_receipt(path, root / 'batch8-build.json')
    require(result['schema_version'] == 'riley.batch8-fusion-correctness.v1' and result['passed'] is True
            and result['gpu_tests_executed'] is True, 'fusion probe did not execute and pass')
    require(all(result[key] == build[key] for key in ('source_root', 'source_commit', 'source_files', 'binaries'))
            and result['source_build'] == evidence(root / 'batch8-build.json'), 'fusion source/build differs')
    require(result['candidate_entry'] == ENTRY and result['candidate_source_sha256'] == build['source_files']['kernels/src/graph_numerics.cu'],
            'fusion entry/source differs')
    require(result['runner'] == evidence(root / 'batch8_fusion_probe.py')
            and result['runner']['sha256'] == sha(Path(probe.__file__)), 'fusion validator differs')
    for key, value in {'cases': 8640, 'layers': 30, 'positions': 32, 'patterns': 3, 'mappings': 3}.items():
        require(type(result[key]) is int and result[key] == value, 'fusion coverage differs: ' + key)
    require(all(result[key] is True for key in FLAGS) and result['performance_measured'] is False
            and result['performance_claim_eligible'] is False, 'fusion correctness/scope differs')
    expected_uuid = binding['environment']['gpu']['uuid'].removeprefix('GPU-').replace('-', '').lower()
    require(result['device']['uuid_hex'].lower() == expected_uuid and result['device']['runtime_version'] == 13000,
            'fusion device/runtime differs from model binding')
    return result


def qualify(root, base):
    root, base = root.resolve(), base.resolve()
    path = root / 'batch8-qualification.json'
    require(not path.exists(), 'refusing to replace qualification')
    build = validate_build(root)
    model = validate_model(root, base, build)
    binding, refs = reference(base)
    fusion = validate_fusion(root, build, binding)
    # Revalidate after the independent hook; no qualification is emitted if
    # source, logs, helpers, references or the selected runtime changed.
    require(validate_build(root) == build and validate_model(root, base, build) == model, 'qualification evidence changed')
    result = {'schema_version': 'riley.batch8-fusion-model-qualification.v1', 'passed': True,
              'source_root': build['source_root'], 'source_commit': build['source_commit'], 'source_files': build['source_files'],
              'binaries': build['binaries'], 'source_clean': True, 'implementation_id': IMPLEMENTATION,
              'numerical_profile': PROFILE, 'correctness_gate_id': GATE, 'graph_implementation_signature': '0xF108',
              'build': evidence(root / 'batch8-build.json'), 'model_tests': evidence(root / 'batch8-model-tests.json'),
              'fusion_probe': evidence(root / 'batch8-fusion-probe/receipt.json'), 'references': refs,
              'gpu_tests_executed_on_current_snapshot': True, 'full_logits_and_kv_exact': True,
              'fusion_cases_exact': fusion['cases'], 'runtime': model['runtime'], 'gpu': model['gpu_after'],
              'source_lineage': {'parent_commit': PARENT, 'parent_build': evidence(root / 'http-token-build.json'),
                                 'overlay': evidence(root / 'batch8-source-overlay.json'), 'changed_files': sorted(CHANGED),
                                 'inherited_http_files': {name: build['source_files'][name] for name in sorted(HTTP_FILES)},
                                 'other_parent_tracked_files_unchanged': True},
              'helpers': model['helpers'], 'fusion_validator': evidence(root / 'batch8_fusion_probe.py'),
              'http_correctness_qualified': False, 'serving_performance_qualified': False,
              'performance_measured': False, 'performance_claim_eligible': False,
              'performance_metrics': {'throughput': None, 'ttft': None, 'tpot': None, 'p95': None, 'p99': None}}
    write_new(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'validate'):
        part = sub.add_parser(command)
        part.add_argument('--root', type=Path, default=ROOT)
        part.add_argument('--base', type=Path, default=BASE)
        if command == 'run':
            part.add_argument('--driver-library-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run_checks(args.root, args.base, args.driver_library_dir) if args.command == 'run' else qualify(args.root, args.base)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
