"""Independent, teacher-forced HF logits after the serving campaign is finalized.

This diagnostic does not change serving settings, replace a reference, assign a
numerical tolerance, or establish performance. Each dtype uses a fresh process.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

CASES_SHA = 'e8acc4dac75b97a3a3820cbf8dbddb3903c4e6af440ea3ed08b9c8052685e9f6'
SOURCE_SHA = '8a8b46e0eb6071356a2452c3e029e8276fbc0b2409357ec248645324374dfa03'
BUILDER_SHA = 'c21db7598264a10735b9f38fd3dba5c69233f63fa94ebf2a00c5bafe7380b67a'
SESSION_SHA = 'aad60fca63db57251ba7c75f7a286b4b066c24ed2a8d8dbdf5941a30c9de585e'
GPU_UUID = 'GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0'
VOCAB = 49152


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def load(path, expected, name):
    require(sha(path) == expected, 'helper source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mapped(path, rules):
    path = Path(path)
    for old, new in sorted(rules, key=lambda pair: len(str(pair[0])), reverse=True):
        if path.is_relative_to(old):
            return new / path.relative_to(old)
    return path


def case_inputs(document, rules):
    require(document['schema_version'] == 'riley.common-prefix-divergence-cases.v1', 'case schema')
    prompt, reference = document['reference_prompt_token_ids'], document['reference_output_token_ids']
    require(len(prompt) == 128 and len(reference) == 32, 'reference lengths')
    require(all(type(t) is int and 0 <= t < VOCAB for t in prompt + reference), 'reference token IDs')
    checked = {}
    def check(item):
        path = mapped(item['path'], rules)
        require(sha(path) == item['sha256'], 'case provenance changed: ' + str(path))
        checked[str(path)] = item['sha256']
        return path
    for item in document['inputs'].values():
        check(item)
    names = set()
    for case in document['cases']:
        name, index = case['case_id'], case['generated_index']
        require(name not in names and name and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name), 'case name')
        names.add(name)
        require(type(index) is int and 0 <= index < 32, 'generated index')
        require(case['input_token_ids'] == prompt + reference[:index], 'common prefix differs')
        require(case['input_length'] == 128 + index and case['logits_input_position'] == 127 + index, 'prefix position')
        require(case['reference_token_id'] == reference[index], 'reference choice')
        observed = set()
        require(case['source_response_count'] == len(case['source_responses']) > 0, 'source count')
        for source in case['source_responses']:
            record = read(check(source['record']))
            raw = check(source['raw_response']).read_bytes()
            require(record['raw_response'].encode() == raw and record['response_complete'], 'raw response binding')
            ids = record['observation']['token_ids']
            require(len(ids) == 32 and all(type(t) is int and 0 <= t < VOCAB for t in ids), 'observed output IDs')
            require(ids[:index] == reference[:index] and ids[index] != reference[index], 'first divergent prefix')
            require(source['first_divergent_generated_index'] == index and source['observed_token_id'] == ids[index], 'source choice')
            require(record['index'] == source['request_index'] and record['phase'] == source['phase'], 'source routing')
            observed.add(ids[index])
        require(case['observed_token_ids'] == sorted(observed), 'observed choices')
    require(len(names) == 2, 'expected two distinct common prefixes')
    return checked


def check_sources(root, package):
    path = root / 'common-prefix-reference-source.json'
    require(sha(path) == SOURCE_SHA, 'reference source inventory changed')
    manifest = read(path)
    actual = {str(p.relative_to(package)): sha(p) for p in package.rglob('*.py')}
    require(actual == manifest['files'], 'reference package source differs')
    return actual


def prerequisites(args):
    root = Path(args.root)
    builder = load(root / 'build_decode_profile_batch8.py', BUILDER_SHA, 'common_prefix_campaign_gate')
    require(builder.ROOT == root, 'campaign root differs')
    gate = builder.campaign_gate()
    session = load(root / 'remote_session_round14.py', SESSION_SHA, 'common_prefix_private_runtime')
    runtime = session.bound_runtime()
    cases_path = root / 'numerical-divergence-cases.json'
    require(sha(cases_path) == CASES_SHA, 'frozen prefix cases differ')
    rules = [(Path(a), Path(b)) for a, b in (value.split('=', 1) for value in args.path_map)]
    cases = read(cases_path)
    inputs = case_inputs(cases, rules)
    package = Path(args.reference_package).resolve()
    sources = check_sources(root, package)
    return builder, gate, session, runtime, cases, inputs, package, sources


def scores(values, case):
    require(len(values) == VOCAB and all(math.isfinite(x) for x in values), 'full finite vocabulary logits required')
    order = sorted(range(VOCAB), key=lambda i: (-values[i], i))
    maximum = values[order[0]]
    return {'argmax_lowest_id_on_tie': order[0], 'top1_top2_logit_gap': maximum - values[order[1]],
            'top10': [{'token_id': i, 'logit': values[i]} for i in order[:10]],
            'choices': [{'token_id': i, 'logit': values[i], 'gap_from_max': maximum-values[i],
                         'strictly_greater_count': sum(x > values[i] for x in values),
                         'equal_count': sum(x == values[i] for x in values)}
                        for i in [case['reference_token_id'], *case['observed_token_ids']]]}


def gpu_identity(root, runtime):
    plan = read(root / 'token-serving-round14-plan.json')
    env = {**plan['base_environment'], **runtime['child_only_overrides']}
    data = subprocess.check_output([plan['nvidia_smi']['path'], '-i', '0',
            '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'], env=env, text=True, timeout=20)
    fields = [part.strip() for part in data.strip().split(',')]
    require(fields == [GPU_UUID, '580.173.02'], 'diagnostic GPU/private driver differs')
    return {'uuid': fields[0], 'driver_version': fields[1]}


def tensor_file(path, tensor, expected_shape, torch):
    require(list(tensor.shape) == expected_shape and str(tensor.device) == 'cpu', 'captured tensor shape/device')
    require(tensor.dtype in (torch.float32, torch.bfloat16) and bool(torch.isfinite(tensor).all()), 'captured tensor dtype/nonfinite')
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    with path.open('xb') as handle:
        handle.write(raw)
    return {**ref(path), 'shape': expected_shape, 'dtype': str(tensor.dtype).removeprefix('torch.'),
            'byte_order': 'little', 'bytes': len(raw)}


def dependencies():
    """Newly record installed implementations, without claiming an old oracle hash."""
    result = {}
    for name in ('torch', 'transformers', 'safetensors', 'huggingface_hub', 'tokenizers', 'numpy'):
        distribution = importlib.metadata.distribution(name)
        files = {}
        for relative in distribution.files or ():
            if str(relative).endswith(('.py', '.so', '/METADATA', '/WHEEL', '/RECORD')):
                path = Path(distribution.locate_file(relative)).resolve(strict=True)
                files[str(path)] = sha(path)
        require(files, 'installed dependency inventory missing: ' + name)
        result[name] = {'version': distribution.version, 'files': files}
    return result


def cuda_libraries():
    files = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(('libcudart', 'libcublas', 'libnvJitLink', 'libcusparse', 'libcudnn')):
            require(not fields[5].endswith(' (deleted)'), 'deleted CUDA library mapping')
            path = Path(fields[5])
            info = path.stat()
            major, minor = (int(part, 16) for part in fields[3].split(':'))
            require(info.st_ino == int(fields[4]) and os.major(info.st_dev) == major and os.minor(info.st_dev) == minor,
                    'CUDA library mapping inode differs')
            files[str(path)] = files.get(str(path)) or sha(path)
    require(any(Path(p).name.startswith('libcudart') for p in files)
            and any(Path(p).name.startswith('libcublas') for p in files), 'CUDA runtime/BLAS mappings missing')
    return files


def loaded_implementations(backend, inventory):
    files = {path: digest for item in inventory.values() for path, digest in item['files'].items()}
    modules = {name: sys.modules[name] for name in ('torch', 'torch._C', 'transformers', 'safetensors', 'huggingface_hub', 'tokenizers', 'numpy')}
    paths = {name: Path(module.__file__).resolve(strict=True) for name, module in modules.items()}
    paths['concrete_model_class'] = Path(inspect.getfile(type(backend._model))).resolve(strict=True)
    result = {}
    for name, path in paths.items():
        require(str(path) in files and sha(path) == files[str(path)], 'loaded implementation is outside recorded distribution: ' + name)
        result[name] = ref(path)
    return result


def load_backend(package, dtype):
    require(not any(name == 'riley_reference' or name.startswith('riley_reference.') for name in sys.modules), 'reference was imported before source validation')
    sys.path.insert(0, str(package.parent))
    from riley_reference.hf_calibration import HuggingFaceCalibrationBackend
    from riley_reference.calibration import FP32_ORACLE_KIND, BF16_ORACLE_KIND
    import riley_reference.hf_calibration as hf
    require(Path(hf.__file__).resolve() == package / 'hf_calibration.py', 'reference import shadowed')
    return HuggingFaceCalibrationBackend.load(artifact_kind=FP32_ORACLE_KIND if dtype == 'fp32' else BF16_ORACLE_KIND)


def worker(args):
    require(sys.byteorder == 'little', 'little-endian host required')
    builder, gate, session, runtime, cases, inputs, package, sources = prerequisites(args)
    session.process_runtime_environment(os.getpid(), runtime)
    output = Path(args.output)
    backend, failure, captured = None, None, []
    context = {'campaign_gate': gate, 'case_inputs': inputs, 'reference_source_files': sources,
               'tool': ref(__file__), 'dtype': args.dtype, 'case_manifest_sha256': CASES_SHA,
               'performance_claim': False, 'acceptance_gate_changed': False}
    try:
        backend = load_backend(package, args.dtype)
        torch = backend._torch
        context['reference_metadata'] = dataclasses.asdict(backend.metadata)
        context['gpu_before'] = gpu_identity(Path(args.root), runtime)
        context['private_maps_before'] = session.verify_private_maps(os.getpid(), runtime)
        require(context['private_maps_before']['compute_loaded'], 'private CUDA was not loaded')
        context['installed_dependencies'] = dependencies()
        context['loaded_implementations'] = loaded_implementations(backend, context['installed_dependencies'])
        for case in cases['cases']:
            ids = torch.tensor([case['input_token_ids']], dtype=torch.long, device=backend._device)
            mask = torch.ones_like(ids)
            hidden, logits, log_probs = backend._capture_numeric(ids, mask)
            prefix = output / case['case_id']
            capture = {'case_id': case['case_id'], 'input_token_ids': case['input_token_ids'],
                       'generated_index': case['generated_index'], 'execution': 'HF eager cache-off teacher-forced B1',
                       'retokenized': False, 'free_generation': False,
                       'logits': tensor_file(prefix.with_suffix('.logits.bin'), logits, [VOCAB], torch),
                       'log_probs': tensor_file(prefix.with_suffix('.log_probs.bin'), log_probs, [VOCAB], torch),
                       'first_layer_hidden': tensor_file(prefix.with_suffix('.hidden.bin'), hidden, [case['input_length'], 576], torch),
                       'scores': scores(logits.float().tolist(), case)}
            captured.append(capture)
            write(prefix.with_suffix('.json'), capture)
            del ids, mask, hidden, logits, log_probs
        torch.cuda.synchronize()
        context['gpu_after'] = gpu_identity(Path(args.root), runtime)
        context['private_maps_after'] = session.verify_private_maps(os.getpid(), runtime)
        context['cuda_library_files'] = cuda_libraries()
        require(context['private_maps_after']['compute_loaded'], 'private CUDA disappeared')
        require(dependencies() == context['installed_dependencies'], 'installed dependency bytes changed')
        require(loaded_implementations(backend, context['installed_dependencies']) == context['loaded_implementations'], 'loaded implementation changed')
        require(check_sources(Path(args.root), package) == sources and builder.campaign_gate() == gate, 'capture dependencies changed')
        require(sha(Path(args.root) / 'numerical-divergence-cases.json') == CASES_SHA and ref(__file__) == context['tool'], 'case/tool bytes changed')
        require(case_inputs(cases, [(Path(a), Path(b)) for a, b in (v.split('=', 1) for v in args.path_map)]) == inputs, 'input provenance changed')
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        close_error = None
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                close_error = error
                failure = failure or {'type': type(error).__name__, 'message': str(error)}
        write(output / 'worker-receipt.json', {**context, 'cases': captured, 'failure': failure,
              'completed': failure is None and len(captured) == 2,
              'backend_closed': backend is not None and close_error is None,
              'cleanup_failure': {'type': type(close_error).__name__, 'message': str(close_error)} if close_error else None,
              'scope': 'independent HF logits only; not captured vLLM graph logits, kernel-cause isolation, or accuracy acceptance'})
        if close_error is not None:
            raise close_error


def stop_child(process):
    # This invocation created the session; an orphan may outlive its leader.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    remaining = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                if os.getsid(int(path.name)) == process.pid and (path / 'stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z':
                    remaining.append(int(path.name))
            except (ProcessLookupError, FileNotFoundError):
                pass
    require(not remaining, 'owned diagnostic session still has live members')
    return {'returncode': process.returncode, 'remaining_owned_pids': remaining, 'cleanup_verified': True}


def run(args):
    builder, gate, _session, runtime, _cases, _inputs, _package, _sources = prerequisites(args)
    root, output = Path(args.root), Path(args.output)
    output.mkdir(mode=0o700)
    plan = read(root / 'token-serving-round14-plan.json')
    env = {**plan['base_environment'], **runtime['child_only_overrides'],
           'HF_HOME': args.hf_home, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    argv = [sys.executable, str(Path(__file__).resolve()), '_worker', '--root', str(root),
            '--reference-package', args.reference_package, '--output', str(output), '--dtype', args.dtype,
            '--hf-home', args.hf_home]
    for value in args.path_map:
        argv += ['--path-map', value]
    write(output / 'launch.json', {'argv': argv, 'environment': env, 'tool': ref(__file__), 'campaign_gate': gate,
                                 'diagnostic_only': True, 'performance_claim': False})
    process, failure, cleanup, cleanup_failure = None, None, None, None
    try:
        with (output / 'worker.log').open('x') as log:
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=log, start_new_session=True)
            write(output / 'process.json', {'pid': process.pid, 'session_id': os.getsid(process.pid)})
            process.wait(timeout=args.timeout)
            require(process.returncode == 0, 'common-prefix worker failed')
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if process is not None:
            try:
                cleanup = stop_child(process)
            except BaseException as error:
                cleanup_failure = {'type': type(error).__name__, 'message': str(error)}
                failure = failure or cleanup_failure
        write(output / 'finalization.json', {'failure': failure, 'cleanup': cleanup, 'cleanup_failure': cleanup_failure})
    require(failure is None and cleanup and cleanup['returncode'] == 0, 'diagnostic incomplete: ' + str(failure))
    require(builder.campaign_gate() == gate, 'serving finalization changed')
    receipt = read(output / 'worker-receipt.json')
    require(receipt['completed'] and receipt['failure'] is None, 'worker receipt incomplete')
    write(output / 'completion.json', {'schema': 'riley.common-prefix-hf-logits.v1', 'worker': ref(output / 'worker-receipt.json'),
          'finalization': ref(output / 'finalization.json'), 'performance_claim': False, 'acceptance_gate_changed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', '_worker'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--reference-package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], required=True)
    parser.add_argument('--hf-home', required=True)
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    require(args.timeout > 0, 'timeout must be positive')
    def interrupted(signum, _frame):
        raise InterruptedError('signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    (run if args.command == 'run' else worker)(args)


if __name__ == '__main__':
    main()
