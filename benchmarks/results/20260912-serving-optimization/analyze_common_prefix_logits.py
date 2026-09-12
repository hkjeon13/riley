"""CPU-only descriptive analysis of two frozen HF teacher-forced captures.

No torch import, GPU execution, tolerance, accuracy acceptance or performance
inference. Installed dependency/map hashes are receipt evidence, not a local
rehash of unavailable remote installations or old canonical-oracle identity.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

CAPTURE_SHA = '413fa06c6e86bee1b2b525854b837c25e19f650ffc5dd38f74040d56a95b129a'
CASES_SHA = 'e8acc4dac75b97a3a3820cbf8dbddb3903c4e6af440ea3ed08b9c8052685e9f6'
SOURCE_SHA = '8a8b46e0eb6071356a2452c3e029e8276fbc0b2409357ec248645324374dfa03'
GPU_UUID = 'GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0'
VOCAB = 49152
EXECUTION = 'HF eager cache-off teacher-forced B1'
DEPENDENCIES = {'torch', 'transformers', 'safetensors', 'huggingface_hub', 'tokenizers', 'numpy'}
IMPLEMENTATIONS = DEPENDENCIES | {'torch._C', 'concrete_model_class'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'invalid SHA256')
    return value


def absolute(value):
    require(isinstance(value, str) and value and Path(value).is_absolute()
            and '..' not in Path(value).parts, 'absolute canonical path required')
    return Path(value)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs, parse_constant=invalid)


def ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def reference(item):
    require(isinstance(item, dict) and set(item) >= {'path', 'sha256'}, 'invalid artifact reference')
    absolute(item['path'])
    digest(item['sha256'])
    return item


class PathMap:
    """All receipt file reads require an explicit absolute source→local rule."""
    def __init__(self, values):
        self.rules = []
        for value in values:
            require(isinstance(value, str) and '=' in value, 'path map requires OLD=LOCAL')
            old, new = value.split('=', 1)
            old, new = absolute(old), absolute(new)
            require(old not in [a for a, _ in self.rules], 'duplicate path map source')
            self.rules.append((old, new))
        self.rules.sort(key=lambda rule: len(rule[0].parts), reverse=True)

    def resolve(self, value):
        path = absolute(value)
        for old, new in self.rules:
            if path.is_relative_to(old):
                result = new / path.relative_to(old)
                # Symlinks may name local data but cannot escape the mapped root.
                require(result.resolve().is_relative_to(new.resolve()), 'mapped path escapes local root')
                return result
        raise ValueError('no explicit path map for: ' + value)

    def checked(self, item):
        reference(item)
        path = self.resolve(item['path'])
        require(sha(path) == item['sha256'], 'artifact hash differs: ' + str(path))
        return path


def frozen_inputs(cases_path, source_path, tool_path):
    for path, expected in [(cases_path, CASES_SHA), (source_path, SOURCE_SHA), (tool_path, CAPTURE_SHA)]:
        require(sha(path) == expected, 'frozen local input changed: ' + str(path))
    cases, sources = read(cases_path), read(source_path)
    require(cases['schema_version'] == 'riley.common-prefix-divergence-cases.v1', 'case schema')
    require(sources['schema'] == 'riley.common-prefix-reference-source.v1', 'source schema')
    require(len(cases['reference_prompt_token_ids']) == 128 and len(cases['reference_output_token_ids']) == 32, 'reference shape')
    require([c['case_id'] for c in cases['cases']] == ['common-prefix-136-output-08', 'common-prefix-157-output-29'], 'exact two cases required')
    for c, index in zip(cases['cases'], [8, 29]):
        require(c['generated_index'] == index and c['input_length'] == 128 + index
                and c['logits_input_position'] == 127 + index, 'teacher-forced position')
        require(c['input_token_ids'] == cases['reference_prompt_token_ids'] + cases['reference_output_token_ids'][:index], 'teacher-forced prefix differs')
        require(c['reference_token_id'] == cases['reference_output_token_ids'][index], 'reference choice differs')
    return cases, sources


def score(values, case):
    require(len(values) == VOCAB and all(math.isfinite(v) for v in values), 'full finite vocabulary required')
    order = sorted(range(VOCAB), key=lambda token: (-values[token], token))
    maximum = values[order[0]]
    choices = [case['reference_token_id'], *case['observed_token_ids']]
    return {'argmax_lowest_id_on_tie': order[0], 'top1_top2_logit_gap': maximum - values[order[1]],
            'top10': [{'token_id': i, 'logit': values[i]} for i in order[:10]],
            'choices': [{'token_id': i, 'logit': values[i], 'gap_from_max': maximum - values[i],
                         'strictly_greater_count': sum(x > values[i] for x in values),
                         'equal_count': sum(x == values[i] for x in values)} for i in choices]}


def tensor(item, shape, dtype, mapper, expected_path):
    require(dtype in ('float32', 'bfloat16'), 'unsupported tensor dtype')
    require(item['shape'] == shape and item['dtype'] == dtype and item['byte_order'] == 'little', 'tensor shape/dtype/byte order differs')
    size = math.prod(shape) * (4 if dtype == 'float32' else 2)
    require(type(item['bytes']) is int and item['bytes'] == size, 'tensor declared byte size differs')
    path = mapper.checked(item)
    require(path.resolve() == expected_path.resolve(), 'tensor belongs to another case/run')
    raw = path.read_bytes()
    require(len(raw) == size, 'tensor actual byte size differs')
    require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'tensor changed while reading')
    if dtype == 'float32':
        values = [v[0] for v in struct.iter_unpack('<f', raw)]
    else:
        # Zero-fill the low FP32 bits, preserving every BF16 bit including -0.
        values = [struct.unpack('<f', struct.pack('<I', v[0] << 16))[0]
                  for v in struct.iter_unpack('<H', raw)]
    require(len(values) == math.prod(shape) and all(math.isfinite(v) for v in values), 'nonfinite/raw tensor shape')
    return values


def launch_arguments(argv):
    require(isinstance(argv, list) and all(isinstance(v, str) for v in argv) and len(argv) >= 3, 'launch argv')
    absolute(argv[0]); absolute(argv[1])
    require(argv[2] == '_worker', 'fresh worker invocation required')
    allowed = {'--root', '--reference-package', '--output', '--dtype', '--hf-home', '--path-map'}
    fields, maps = {}, []
    require((len(argv) - 3) % 2 == 0, 'launch option arity')
    for flag, value in zip(argv[3::2], argv[4::2]):
        require(flag in allowed, 'unexpected worker launch flag')
        if flag == '--path-map':
            maps.append(value)
        else:
            require(flag not in fields, 'duplicate worker launch flag')
            fields[flag] = value
    require(set(fields) == allowed - {'--path-map'}, 'missing worker launch field')
    for flag in ('--root', '--reference-package', '--output', '--hf-home'):
        absolute(fields[flag])
    return fields, PathMap(maps)


def expected_case_inputs(cases, execution_map):
    # Reconstruct historical capture path names only. These execution rules do
    # not authorize any analyzer file reads or substitute for --path-map.
    items = list(cases['inputs'].values())
    for case in cases['cases']:
        for source in case['source_responses']:
            items.extend([source['record'], source['raw_response']])
    expected = {}
    for item in items:
        reference(item)
        path = absolute(item['path'])
        for old, new in execution_map.rules:
            if path.is_relative_to(old):
                path = new / path.relative_to(old)
                break
        if str(path) in expected:
            require(expected[str(path)] == item['sha256'], 'case provenance path collision')
        expected[str(path)] = item['sha256']
    return expected


def provenance(worker, sources):
    require(worker['reference_source_files'] == sources['files'], 'reference source receipt differs')
    installed = worker['installed_dependencies']
    require(set(installed) == DEPENDENCIES, 'installed dependency inventory set')
    files = {}
    for name, item in installed.items():
        require(isinstance(item['version'], str) and item['version'] and item['files'], 'installed inventory version/files')
        for path, value in item['files'].items():
            absolute(path); digest(value)
            require(path not in files or files[path] == value, 'installed source hash disagreement')
            files[path] = value
    loaded = worker['loaded_implementations']
    require(set(loaded) == IMPLEMENTATIONS, 'loaded implementation set')
    for item in loaded.values():
        reference(item)
        require(files.get(item['path']) == item['sha256'], 'loaded implementation not in newly recorded inventory')
    metadata = worker['reference_metadata']
    require(set(metadata) == {'python_version', 'python_executable_sha256', 'python_platform_system', 'python_platform_machine', 'torch_version', 'transformers_version', 'safetensors_version', 'config_sha256', 'tokenizer_sha256', 'tokenizer_files_sha256'}, 'reference metadata fields')
    require(metadata['python_platform_system'] == 'linux' and metadata['python_platform_machine'] == 'x86_64', 'reference platform')
    for key in ('python_executable_sha256', 'config_sha256', 'tokenizer_sha256'):
        digest(metadata[key])
    require(isinstance(metadata['python_version'], str) and metadata['python_version'], 'reference Python version')
    require(metadata['tokenizer_files_sha256'], 'tokenizer metadata')
    for value in metadata['tokenizer_files_sha256'].values():
        digest(value)
    for name in ('torch', 'transformers', 'safetensors'):
        require(metadata[name + '_version'] == installed[name]['version'].split('+', 1)[0], 'reference/installed version differs')
    for when in ('before', 'after'):
        require(worker['gpu_' + when] == {'uuid': GPU_UUID, 'driver_version': '580.173.02'}, 'capture GPU identity')
        maps = worker['private_maps_' + when]
        require(maps['compute_loaded'] is True and maps['all_observed_vendor_mappings_pinned'] is True
                and maps['private_vendor_files'] and maps['driver_mappings'], 'private compute maps incomplete')
        mapped_files = {}
        for line in maps['driver_mappings']:
            fields = line.split(None, 5)
            require(len(fields) == 6 and not fields[5].endswith(' (deleted)'), 'invalid vendor mapping line')
            path = fields[5]
            absolute(path)
            record = maps['private_vendor_files'][path]
            digest(record['sha256'])
            require(record['device'] == fields[3] and type(record['inode']) is int and record['inode'] == int(fields[4]), 'vendor inode receipt mismatch')
            mapped_files[path] = record
        require(mapped_files == maps['private_vendor_files'] and any(Path(p).name.startswith('libcuda.so') for p in mapped_files), 'unmapped/private CUDA receipt')
    libraries = worker['cuda_library_files']
    require(libraries and all(any(Path(p).name.startswith(prefix) for p in libraries) for prefix in ('libcudart', 'libcublas')), 'CUDA library inventory')
    for path, value in libraries.items():
        absolute(path); digest(value)


def campaign_gate(gate):
    digest(gate['preparation_sha256']); digest(gate['finalization_sha256'])
    reference(gate['restoration'])
    require(gate['performance_or_acceptance_inferred'] is False and gate['process_exit_receipts'], 'serving finalization receipt scope')
    for path, value in gate['process_exit_receipts'].items():
        absolute(path); digest(value)


def validate_run(completion_path, dtype, mapper, cases, sources):
    completion_path = Path(completion_path).resolve(strict=True)
    directory = completion_path.parent
    completion_ref = ref(completion_path)
    completion = read(completion_path)
    require(completion['schema'] == 'riley.common-prefix-hf-logits.v1'
            and completion['performance_claim'] is False and completion['acceptance_gate_changed'] is False, 'capture completion schema/scope')
    worker_path, final_path = mapper.checked(completion['worker']), mapper.checked(completion['finalization'])
    require(worker_path.resolve() == directory / 'worker-receipt.json' and final_path.resolve() == directory / 'finalization.json', 'completion cross-run reference')
    worker, final = read(worker_path), read(final_path)
    require(worker['completed'] is True and worker['backend_closed'] is True and worker['failure'] is None
            and worker['cleanup_failure'] is None and worker['dtype'] == dtype, 'worker incomplete or wrong dtype')
    require(final['failure'] is None and final['cleanup_failure'] is None
            and type(final['cleanup']['returncode']) is int and final['cleanup']['returncode'] == 0 and final['cleanup']['cleanup_verified'] is True
            and final['cleanup']['remaining_owned_pids'] == [], 'owned worker finalization incomplete')
    require(worker['performance_claim'] is False and worker['acceptance_gate_changed'] is False
            and worker['case_manifest_sha256'] == CASES_SHA, 'worker scope/cases')
    require(worker['tool']['sha256'] == CAPTURE_SHA, 'worker capture tool differs')
    reference(worker['tool'])
    launch_ref, process_ref = ref(directory / 'launch.json'), ref(directory / 'process.json')
    launch, process = read(directory / 'launch.json'), read(directory / 'process.json')
    require(launch['tool'] == worker['tool'] and launch['campaign_gate'] == worker['campaign_gate']
            and launch['diagnostic_only'] is True and launch['performance_claim'] is False, 'launch receipt binding')
    fields, execution_map = launch_arguments(launch['argv'])
    require(fields['--dtype'] == dtype and launch['argv'][1] == worker['tool']['path']
            and mapper.resolve(fields['--output']).resolve() == directory, 'launch dtype/tool/output differs')
    require(launch['environment']['HF_HUB_OFFLINE'] == '1' and launch['environment']['TRANSFORMERS_OFFLINE'] == '1'
            and launch['environment']['HF_HOME'] == fields['--hf-home'], 'offline model environment')
    require(type(process['pid']) is int and process['pid'] > 0 and type(process['session_id']) is int
            and process['session_id'] == process['pid'], 'fresh owned worker session')
    require(worker['case_inputs'] == expected_case_inputs(cases, execution_map), 'worker frozen case provenance differs')
    campaign_gate(worker['campaign_gate']); provenance(worker, sources)
    require(len(worker['cases']) == 2 and [c['case_id'] for c in worker['cases']] == [c['case_id'] for c in cases['cases']], 'exact two captured prefixes')
    values, sidecars, raw_refs = {}, [], []
    for captured, case in zip(worker['cases'], cases['cases']):
        require(captured['input_token_ids'] == case['input_token_ids'] and captured['generated_index'] == case['generated_index'], 'captured teacher-forced prefix differs')
        require(captured['execution'] == EXECUTION and captured['retokenized'] is False and captured['free_generation'] is False, 'capture is not exact B1 cache-off teacher forcing')
        path = directory / (case['case_id'] + '.json')
        sidecar_ref = ref(path)
        require(read(path) == captured, 'case JSON differs from hash-bound worker record')
        sidecars.append(sidecar_ref)
        row = {}
        for key, suffix, shape in [('logits', 'logits', [VOCAB]), ('log_probs', 'log_probs', [VOCAB]), ('first_layer_hidden', 'hidden', [case['input_length'], 576])]:
            expected_dtype = 'float32' if dtype == 'fp32' or key == 'log_probs' else 'bfloat16'
            row[key] = tensor(captured[key], shape, expected_dtype, mapper, directory / (case['case_id'] + '.' + suffix + '.bin'))
            raw_refs.append(captured[key])
        require(captured['scores'] == score(row['logits'], case), 'stored scores are not raw tensor-derived')
        values[case['case_id']] = row
    for item in [completion_ref, launch_ref, process_ref, *sidecars]:
        require(ref(item['path']) == item, 'receipt changed during validation')
    for item in [completion['worker'], completion['finalization'], *raw_refs]:
        mapper.checked(item)
    evidence = {'completion': completion_ref, 'worker': ref(worker_path), 'finalization': ref(final_path),
                'launch': launch_ref, 'process': process_ref, 'case_sidecars': sidecars,
                'worker_pid': process['pid'], 'dtype': dtype}
    return worker, values, evidence


def inventory_difference(left, right):
    """Lazy library loading may differ by dtype; shared path bytes may not."""
    common = set(left) & set(right)
    require(all(left[path] == right[path] for path in common), 'shared native library identity differs')
    return {'shared_paths': sorted(common), 'fp32_only': sorted(set(left) - set(right)),
            'bf16_only': sorted(set(right) - set(left))}


def ranks(values):
    result = [0] * len(values)
    for rank, token in enumerate(sorted(range(len(values)), key=lambda i: (-values[i], i)), 1):
        result[token] = rank
    return result


def differences(left, right):
    require(len(left) == len(right) > 0, 'comparison tensor shape')
    delta = [b-a for a, b in zip(left, right)]
    order = sorted(range(len(delta)), key=lambda i: (-abs(delta[i]), i))
    return {'elements': len(delta), 'different_value_count': sum(a != b for a, b in zip(left, right)),
            'different_fp32_bit_count': sum(struct.pack('<f', a) != struct.pack('<f', b) for a, b in zip(left, right)),
            'signed_zero_difference_count': sum(a == b == 0 and math.copysign(1, a) != math.copysign(1, b) for a, b in zip(left, right)),
            'max_absolute_difference': abs(delta[order[0]]), 'mean_absolute_difference': math.fsum(abs(v) for v in delta)/len(delta),
            'root_mean_square_difference': math.sqrt(math.fsum(v*v for v in delta)/len(delta)),
            'minimum_bf16_minus_fp32': min(delta), 'maximum_bf16_minus_fp32': max(delta),
            'largest_absolute_differences': [{'flat_index': i, 'fp32': left[i], 'bf16': right[i], 'bf16_minus_fp32': delta[i]} for i in order[:20]]}


def analyze(fp32_path, bf16_path, cases_path, source_path, tool_path, path_maps, output):
    local_bindings = {name: ref(path) for name, path in [('analyzer', __file__), ('capture_helper', tool_path),
                      ('cases_manifest', cases_path), ('reference_source_inventory', source_path)]}
    cases, sources = frozen_inputs(cases_path, source_path, tool_path)
    mapper = PathMap(path_maps)
    require(Path(fp32_path).resolve().parent != Path(bf16_path).resolve().parent, 'distinct fresh capture directories required')
    left, a, a_ref = validate_run(fp32_path, 'fp32', mapper, cases, sources)
    right, b, b_ref = validate_run(bf16_path, 'bf16', mapper, cases, sources)
    require(a_ref['worker_pid'] != b_ref['worker_pid'], 'distinct fresh worker PID receipts required')
    for key in ('campaign_gate', 'reference_metadata', 'reference_source_files', 'installed_dependencies', 'loaded_implementations'):
        require(left[key] == right[key], 'FP32/BF16 provenance differs: ' + key)
    native_differences = {'cuda_library_files': inventory_difference(left['cuda_library_files'], right['cuda_library_files'])}
    for when in ('before', 'after'):
        native_differences['private_maps_' + when] = inventory_difference(left['private_maps_' + when]['private_vendor_files'], right['private_maps_' + when]['private_vendor_files'])
    report = {'schema': 'riley.common-prefix-logits-analysis.v1', 'analysis_complete': True,
              'diagnostic_only': True, 'performance_claim': False, 'acceptance_gate_changed': False,
              'accuracy_acceptance_decision': None, 'numerical_tolerance': None,
              'execution': EXECUTION, 'actual_vllm_graph_logits_captured': False,
              'scope': 'Full arrays compared descriptively. This does not isolate a kernel cause or establish accuracy/performance.',
              'provenance_scope': {'locally_rehashed': 'analyzer, frozen capture helper/cases/source inventory, both completion/worker/finalization/launch/process/case files and every raw tensor',
                 'receipt_only_not_rehashed': 'historical case source responses, serving campaign/restoration files, reference package and installed dependency/native driver/library files',
                 'installed_inventory_is_new_capture_evidence_not_old_canonical_oracle': True,
                 'worker_freshness_evidence': 'distinct directories/PIDs, launch argv and successful owned-session cleanup receipts; no independent historical process-start-time receipt'},
              **local_bindings,
              'path_maps': [{'source': str(old), 'local': str(new)} for old, new in mapper.rules],
              'captures': {'fp32': a_ref, 'bf16': b_ref}, 'loaded_native_inventory_comparison': native_differences, 'cases': []}
    prepared = []
    for case in cases['cases']:
        name = case['case_id']; x, y = a[name], b[name]
        lr, rr = ranks(x['logits']), ranks(y['logits'])
        row = {'case_id': name, 'input_token_ids': case['input_token_ids'], 'generated_index': case['generated_index'],
               'reference_token_id': case['reference_token_id'], 'observed_serving_token_ids': case['observed_token_ids'],
               'fp32_scores': score(x['logits'], case), 'bf16_scores': score(y['logits'], case),
               'logits': differences(x['logits'], y['logits']), 'log_probs': differences(x['log_probs'], y['log_probs']),
               'first_layer_hidden': differences(x['first_layer_hidden'], y['first_layer_hidden']),
               'rank_definition': '1-based descending logit; equal logits ordered by ascending token ID',
               'different_rank_count': sum(i != j for i, j in zip(lr, rr)),
               'maximum_absolute_rank_change': max(abs(i-j) for i, j in zip(lr, rr)),
               'sum_exp_log_probs': {'fp32': math.fsum(math.exp(v) for v in x['log_probs']), 'bf16': math.fsum(math.exp(v) for v in y['log_probs'])}}
        row['first_layer_hidden']['shape'] = [case['input_length'], 576]
        prepared.append((row, x, y, lr, rr))
    # Do not create successful-looking output before every capture is validated.
    for item in local_bindings.values():
        require(ref(item['path']) == item, 'local source changed during analysis')
    for evidence, worker in [(a_ref, left), (b_ref, right)]:
        for item in [evidence[key] for key in ('completion', 'worker', 'finalization', 'launch', 'process')] + evidence['case_sidecars']:
            require(ref(item['path']) == item, 'capture receipt changed during analysis')
        for captured in worker['cases']:
            for key in ('logits', 'log_probs', 'first_layer_hidden'):
                mapper.checked(captured[key])
    output = Path(output)
    output.mkdir(mode=0o700)
    for row, x, y, lr, rr in prepared:
        path = output / (row['case_id'] + '.vocabulary.tsv')
        with path.open('x', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['token_id', 'fp32_logit', 'bf16_logit', 'logit_bf16_minus_fp32', 'fp32_rank', 'bf16_rank', 'rank_bf16_minus_fp32', 'fp32_log_prob', 'bf16_log_prob', 'log_prob_bf16_minus_fp32'])
            for i in range(VOCAB):
                writer.writerow([i, x['logits'][i], y['logits'][i], y['logits'][i]-x['logits'][i], lr[i], rr[i], rr[i]-lr[i], x['log_probs'][i], y['log_probs'][i], y['log_probs'][i]-x['log_probs'][i]])
        row['full_vocabulary'] = {**ref(path), 'data_rows': VOCAB}
        report['cases'].append(row)
    with (output / 'analysis.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--fp32-completion', required=True)
    parser.add_argument('--bf16-completion', required=True)
    parser.add_argument('--cases', default=str(here / 'numerical-divergence-cases.json'))
    parser.add_argument('--reference-sources', default=str(here / 'common-prefix-reference-source.json'))
    parser.add_argument('--capture-tool', default=str(here / 'capture_common_prefix_logits.py'))
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.fp32_completion, args.bf16_completion, args.cases, args.reference_sources, args.capture_tool, args.path_map, args.output)


if __name__ == '__main__':
    main()
