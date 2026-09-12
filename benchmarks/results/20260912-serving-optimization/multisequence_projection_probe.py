#!/usr/bin/env python3
"""Isolated M2/M4 anchored-GEMM arithmetic experiment; prepare/build/run separately.

Only run launches GPU work. No production edits, remote operations, performance
measurements, fallback algorithms, or serving qualification are performed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CHILD = 'batch8_http_token_observation_check_v2.py'
CHILD_SHA = '0023e42863a7cab7b448098e218e2b2fdd4c23ea3a5820a9e62cadece00d2e1d'
PRECISE_HELPER = 'batch6_projection_probe.py'
PRECISE_HELPER_SHA = 'e306157697d5584b1f5d59826326f1682e6be57c224108fed68e8a2c3146e3d9'
COMMIT = '8329c1aeec6e013f581128888c536e15f8bf7300'
BINARY_SHA = '4c0876dbe080920bcc0d689df16d52b7714f31aa625b603fb9f7f1acc760c741'
PRECISE_SHA = 'fbf8e8cd517abd29502d50c33a7d46fe7738bb7393ffbee7d9b0e532f305d87f'
UUID = '9087e4256acab722b8c9cc0423b39fb0'
DRIVER_SHA = '266916dac6c5e7e4655526570cf303a6cc017880d2552960abc66017a7c98cf4'
NATIVE_BUILD_INFO = 'riley-cuda-native abi=1 nvcc=13.3.73'
SCHEMA = 'riley.multisequence-projection'
SHAPES = (('qkv', 960, 576, 74), ('gate_up', 3072, 576, 74), ('o', 576, 576, 74),
          ('down', 576, 1536, 75), ('head', 49152, 576, 89))
PATTERNS = ('row_fingerprint', 'signed_zero_impulses', 'bounded_cancellation')
MODES = ((2, 2), (4, 4), (4, 3))
COUNTS = {'cases': 1089, 'weight_groups': 121, 'plans': 15, 'layers': 30, 'patterns': 3}
CASE_FLAGS = ('exact_outputs', 'finite_outputs', 'guards_intact', 'inputs_unchanged', 'weights_unchanged',
              'allocation_stats_unchanged_across_execute', 'case_allocations_freed')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def evidence(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def check(item):
    require(set(item) == {'path', 'sha256'} and evidence(item['path']) == item, 'artifact changed: '+str(item))


def load(filename, digest, name):
    path = HERE/filename
    require(sha(path) == digest, 'frozen helper changed: '+filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prerequisites(root, base):
    # This pinned validator reconstructs the entire fresh batch8 build/model/
    # fusion qualification, including the exact private-runtime symlink schema.
    # Import and validation perform no HTTP launch or GPU work.
    child = load(CHILD, CHILD_SHA, '_multisequence_batch8_proofs')
    build, proof = child.validate_prerequisites(root, base)
    require(build['source_commit'] == COMMIT and build['binaries'][str(root/'batch8-target/release/riley')] == BINARY_SHA,
            'batch8 source/binary differs')
    require(sha(root/'batch8-source/kernels/src/graph_numerics_precise.cu') == PRECISE_SHA,
            'accepted precise source differs')
    binding, reference = child.qualifier.reference(base)
    return build, proof, reference


def pattern_bytes(pattern, k, m, active):
    require(pattern in range(3) and k in (576, 1536) and (m, active) in MODES, 'unsupported fixture geometry')
    words = []
    for row in range(m):
        for column in range(k):
            sign = ((row+column) & 1) << 15
            if row >= active: bits = 0
            elif pattern == 0: bits = sign | (0x3e00 + ((row*43+column*29) & 255))
            elif pattern == 1:
                bits = sign
                if column == (row*137+11) % k: bits = (row & 1)*0x8000 | 0x3f80
                elif column == (row*137+12) % k: bits = sign | (1+row)  # raw BF16 subnormal
            else: bits = (((column//2+row) & 1) << 15) | (0x3c00 + ((column*7+row*31) & 511))
            words.append(bits)
    return struct.pack('<'+'H'*len(words), *words)


def group_specs(config):
    require(config.get('num_hidden_layers') == 30 and config.get('hidden_size') == 576
            and config.get('intermediate_size') == 1536 and config.get('vocab_size') == 49152,
            'model config geometry differs')
    require(type(config.get('tie_word_embeddings')) is bool, 'explicit head binding required')
    for layer in range(30):
        prefix = f'model.layers.{layer}.'
        yield layer, 0, [(prefix+'self_attn.q_proj.weight', 576), (prefix+'self_attn.k_proj.weight', 192), (prefix+'self_attn.v_proj.weight', 192)]
        yield layer, 1, [(prefix+'mlp.gate_proj.weight', 1536), (prefix+'mlp.up_proj.weight', 1536)]
        yield layer, 2, [(prefix+'self_attn.o_proj.weight', 576)]
        yield layer, 3, [(prefix+'mlp.down_proj.weight', 576)]
    yield 30, 4, [('model.embed_tokens.weight' if config['tie_word_embeddings'] else 'lm_head.weight', 49152)]


def tensor_slice(handle, header, start, size, name, n, k):
    t = header[name]
    require(t['dtype'] == 'BF16' and t['shape'] == [n, k], 'tensor dtype/shape differs: '+name)
    offsets = t['data_offsets']
    require(len(offsets) == 2 and all(type(x) is int for x in offsets), 'tensor offsets invalid')
    left, right = offsets
    require(0 <= left < right and right-left == n*k*2 and start+right <= size, 'tensor range differs: '+name)
    handle.seek(start+left); raw = handle.read(right-left)
    require(len(raw) == right-left, 'short checkpoint read')
    # Inspect raw exponent bits without converting any fixture value.
    require(all((value[0] & 0x7f80) != 0x7f80 for value in struct.iter_unpack('<H', raw)), 'nonfinite checkpoint tensor')
    return raw, {'tensor': name, 'shape': [n, k], 'data_offsets': offsets, 'file_offset': start+left,
                 'bytes': right-left, 'sha256': hashlib.sha256(raw).hexdigest()}


def fixture_description(model, config, output, create):
    helper = load(PRECISE_HELPER, PRECISE_HELPER_SHA, '_multisequence_safetensors')
    header, start = helper.tensor_header(model)
    files, weights, cases, occupied = {}, [], [], []
    directory = output/'fixtures'
    for k in (576, 1536):
        for pattern in range(3):
            for m, active in MODES:
                path = directory/f'x-{k}-{pattern}-m{m}-a{active}.bf16'
                raw = pattern_bytes(pattern, k, m, active)
                if create:
                    with path.open('xb') as f: f.write(raw)
                require(path.read_bytes() == raw, 'input fixture changed')
                files[str(path)] = sha(path)
    with model.open('rb') as handle:
        for layer, shape, pieces in group_specs(config):
            name, n, k, custom = SHAPES[shape]
            packed, parts = bytearray(), []
            for tensor, width in pieces:
                raw, part = tensor_slice(handle, header, start, model.stat().st_size, tensor, width, k)
                left, right = part['data_offsets']
                require(all(right <= a or left >= b for a, b in occupied), 'selected source tensor ranges overlap')
                occupied.append((left, right)); packed.extend(raw); parts.append(part)
            require(len(packed) == n*k*2, 'packed weight length differs')
            if shape == 4 and config['tie_word_embeddings'] and 'lm_head.weight' in header:
                raw, _ = tensor_slice(handle, header, start, model.stat().st_size, 'lm_head.weight', n, k)
                require(raw == packed, 'optional tied head differs from actual embedding binding')
            path = directory/f'w-{layer:02d}-{name}.bf16'
            if create:
                with path.open('xb') as f: f.write(packed)
            require(path.read_bytes() == packed, 'packed fixture differs from exact checkpoint slices')
            files[str(path)] = sha(path)
            weights.append({'layer': layer, 'shape': shape, 'projection': name, 'n': n, 'k': k,
                            'parts': parts, 'fixture': evidence(path)})
            for pattern in range(3):
                for m, active in MODES:
                    cases.append({'case_id': len(cases), 'layer': layer, 'shape': shape, 'm': m, 'active_rows': active,
                                  'n': n, 'k': k, 'pattern': pattern,
                                  'input': str(directory/f'x-{k}-{pattern}-m{m}-a{active}.bf16'), 'weight': str(path)})
    tsv = ''
    for c in cases:
        values = [c[key] for key in ('case_id','layer','shape','m','active_rows','n','k','pattern','input','weight')]
        require(all(not any(x in str(value) for x in '\t\n\r') for value in values), 'TSV path contains control character')
        tsv += '\t'.join(map(str, values))+'\n'
    index = output/'cases.tsv'
    if create:
        with index.open('x') as f: f.write(tsv)
    require(index.read_text() == tsv, 'case TSV differs from reconstructed fixtures')
    return {'fixture_files': files, 'weight_records': weights, 'case_records': cases, 'case_index': evidence(index),
            'safetensors_data_start': start}


def prepare(root, base, output):
    root, base, output = root.resolve(strict=True), base.resolve(strict=True), output.resolve()
    build, proof, reference = prerequisites(root, base)
    model = Path(next(path for path in reference['model_files'] if path.endswith('/model.safetensors')))
    config = model.with_name('config.json'); cfg = read(config)
    list(group_specs(cfg))
    output.mkdir(mode=0o700); (output/'fixtures').mkdir(mode=0o700)
    fixtures = fixture_description(model, cfg, output, True)
    manifest = {'schema_version': SCHEMA+'-fixtures.v1', **COUNTS, 'root': str(root), 'base': str(base),
                'source_root': build['source_root'], 'source_commit': build['source_commit'], 'source_build': evidence(root/'batch8-build.json'),
                'binaries': build['binaries'], 'source_files': build['source_files'], 'prerequisites': proof, 'reference': reference,
                'runner': evidence(Path(__file__)), 'native_source': evidence(Path(__file__).with_suffix('.cpp')),
                'model': evidence(model), 'model_config': evidence(config),
                'dependencies': [evidence(HERE/CHILD), evidence(HERE/PRECISE_HELPER)],
                'patterns': list(PATTERNS), 'modes': [list(x) for x in MODES], 'guard_bytes_each_side': 256,
                'weights_converted': False, 'gpu_tests_executed': False, 'performance_measured': False,
                'performance_claim_eligible': False, **fixtures}
    write(output/'fixtures.json', manifest)
    return {'manifest': evidence(output/'fixtures.json'), **COUNTS, 'gpu_tests_executed': False}


def validate_manifest(path):
    path = path.resolve(strict=True); m = read(path)
    require(m['schema_version'] == SCHEMA+'-fixtures.v1' and all(m[k] == v for k, v in COUNTS.items()), 'fixture schema/count differs')
    for key in ('runner','native_source','model','model_config','source_build'): check(m[key])
    require(m['runner'] == evidence(__file__) and m['native_source'] == evidence(Path(__file__).with_suffix('.cpp')),
            'probe source changed')
    require(m['dependencies'] == [evidence(HERE/CHILD), evidence(HERE/PRECISE_HELPER)], 'dependency pins changed')
    build, proof, refs = prerequisites(Path(m['root']), Path(m['base']))
    require(m['source_build'] == evidence(Path(m['root'])/'batch8-build.json') and m['prerequisites'] == proof and m['reference'] == refs,
            'current build/reference/proof differs')
    require(all(m[k] == build[k] for k in ('source_root','source_commit','source_files','binaries')), 'source binding differs')
    require(m['model']['sha256'] == refs['model_files'][m['model']['path']]
            and Path(m['model_config']['path']) == Path(m['model']['path']).with_name('config.json'), 'model/config binding differs')
    actual = fixture_description(Path(m['model']['path']), read(m['model_config']['path']), path.parent, False)
    require(all(m[key] == value for key, value in actual.items()), 'fixture description differs')
    require(m['patterns'] == list(PATTERNS) and m['modes'] == [list(x) for x in MODES] and m['guard_bytes_each_side'] == 256
            and all(m[k] is False for k in ('weights_converted','gpu_tests_executed','performance_measured','performance_claim_eligible')),
            'fixture scope differs')
    return m


def production_link(m, nvcc):
    helper = load(PRECISE_HELPER, PRECISE_HELPER_SHA, '_multisequence_production_flags')
    options = helper.production_flags(m, nvcc)
    directory = Path(options['original_compile_cwd'])
    archive = directory/'libriley_cuda_native.a'
    installed = directory.parent/'cuda-native-install/lib/libriley_cuda_native.a'
    require(sha(archive) == sha(installed), 'installed and built native archives differ')
    libs, artifacts = {}, [evidence(installed)]
    for key in ('cudart','driver','cublaslt'):
        path = directory/f'riley-cuda-{key}-Release.path'
        target = Path(path.read_text().strip())
        require(target.is_absolute() and target.is_file(), 'production link library missing')
        libs[key] = evidence(target); artifacts.append(evidence(path))
    optional = directory/'riley-cuda-nvml-Release.path'
    if optional.exists():
        libs['nvml'] = evidence(Path(optional.read_text().strip())); artifacts.append(evidence(optional))
    # Pin the archive's actual members and compare each to the build-directory
    # object; this is a newly recorded archive identity, not a retroactive claim
    # that an old model receipt hashed the static archive.
    ar = Path('/usr/bin/ar'); require(ar.is_file(), 'explicit system ar unavailable')
    members = subprocess.check_output([str(ar), 't', str(archive)], text=True).splitlines()
    objects = sorted((directory/'CMakeFiles/riley_cuda_native.dir/src').glob('*.o'))
    require('gemm.cu.o' in members and len(members) == len(set(members)) and set(members) == {p.name for p in objects},
            'archive member set differs from production objects')
    object_evidence = []
    for obj in objects:
        data = subprocess.check_output([str(ar), 'p', str(archive), obj.name])
        require(hashlib.sha256(data).hexdigest() == sha(obj), 'archive member differs from production object: '+obj.name)
        object_evidence.append(evidence(obj))
    return {'archive': evidence(archive), 'installed_archive': evidence(installed), 'objects': object_evidence,
            'native_build_info_source': evidence(Path(m['source_root'])/'kernels/src/version.cu'),
            'libraries': libs, 'artifacts': artifacts, 'production_options': options, 'ar': evidence(ar),
            'archive_previously_qualified_hash_available': False}


def command(m, cxx, nvcc, link, binary):
    # Host-only harness has no arithmetic GPU code; all device arithmetic is in
    # the unchanged archive. Linking explicitly avoids nvcc's default cudart.
    libs = link['libraries']
    argv = [str(cxx), '-std=c++17', '-O2', '-I'+str(Path(m['source_root'])/'kernels/include'),
            '-I'+str(nvcc.parent.parent/'include'), m['native_source']['path'], link['archive']['path']]
    argv += [libs[key]['path'] for key in ('cublaslt','cudart','driver')]
    if 'nvml' in libs: argv.append(libs['nvml']['path'])
    argv += ['-pthread', '-ldl', '-o', str(binary)]
    return argv


def build_probe(path, cxx, nvcc):
    m = validate_manifest(path); cxx, nvcc = cxx.resolve(strict=True), nvcc.resolve(strict=True)
    link = production_link(m, nvcc)
    directory = path.resolve().parent/'build'; directory.mkdir(mode=0o700)
    binary = directory/'multisequence_projection_probe'
    argv = command(m, cxx, nvcc, link, binary)
    compiler_version = subprocess.check_output([str(cxx),'--version'], text=True)
    nvcc_version = subprocess.check_output([str(nvcc),'--version'], text=True)
    require(re.findall(r'\bV(\d+\.\d+\.\d+)\b',nvcc_version) == ['13.3.73'], 'compiler version differs from frozen batch8 build')
    env = os.environ.copy()
    for key in ('LD_PRELOAD','LD_AUDIT'): env.pop(key,None)
    with (directory/'build.log').open('x') as log:
        subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    require(production_link(m, nvcc) == link, 'archive changed during build')
    validate_manifest(path)
    result = {'schema_version': SCHEMA+'-compile.v1', 'manifest': evidence(path), 'runner': evidence(__file__),
              'native_source': m['native_source'], 'cxx': evidence(cxx), 'nvcc': evidence(nvcc),
              'cxx_version': compiler_version, 'nvcc_version': nvcc_version, 'native_build_info': NATIVE_BUILD_INFO, 'link': link,
              'argv': argv, 'cwd': str(directory), 'binary': evidence(binary), 'log': evidence(directory/'build.log'),
              'environment_sha256': hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest(),
              'production_source_recompiled': False, 'gpu_tests_executed': False, 'performance_measured': False}
    write(directory/'compile.json', result); return result


def validate_compile(path, m):
    r = read(path.parent/'build/compile.json')
    require(r['schema_version'] == SCHEMA+'-compile.v1' and r['manifest'] == evidence(path)
            and r['runner'] == evidence(__file__) and r['native_source'] == m['native_source']
            and r['native_build_info'] == NATIVE_BUILD_INFO
            and re.findall(r'\bV(\d+\.\d+\.\d+)\b',r['nvcc_version']) == ['13.3.73'], 'compile binding differs')
    for key in ('cxx','nvcc','binary','log'): check(r[key])
    require(r['link'] == production_link(m, Path(r['nvcc']['path'])), 'production archive/link evidence changed')
    require(r['argv'] == command(m, Path(r['cxx']['path']), Path(r['nvcc']['path']), r['link'], Path(r['binary']['path']))
            and r['cwd'] == str(path.parent/'build') and Path(r['binary']['path']) == path.parent/'build/multisequence_projection_probe',
            'compile invocation differs')
    require(all(r[k] is False for k in ('production_source_recompiled','gpu_tests_executed','performance_measured')), 'compile scope differs')
    require(subprocess.check_output([r['cxx']['path'],'--version'], text=True) == r['cxx_version']
            and subprocess.check_output([r['nvcc']['path'],'--version'], text=True) == r['nvcc_version'], 'compiler identity changed')
    return r


def validate_metadata(a, shape, m):
    _, n, k, custom = SHAPES[shape]
    expected = {'struct_size': 112, 'backend': 1, 'algorithm_id': 13, 'tile_id': 0, 'stages_id': 0,
                'split_k': 1, 'reduction_scheme': 0, 'cta_swizzling': 0, 'custom_option': custom,
                'deterministic': 1, 'workspace_bytes': 0, 'numerical_implementation_flags': 131585,
                'compute_capability_major': 8, 'compute_capability_minor': 9, 'runtime_version': 13000,
                'cublaslt_version': 130101, 'm': m, 'n': n, 'k': k, 'reserved': [0, 0]}
    require(a == expected and all(type(a[key]) is int for key in expected if key != 'reserved')
            and all(type(value) is int for value in a['reserved']), 'plan metadata differs from frozen anchor')


def validate_records(records, m):
    require(len(records) == 1107 and len(m['case_records']) == 1089
            and all(r['schema_version'] == SCHEMA+'-native.v1' for r in records), 'native record count/schema differs')
    device, end, summary = records[0], records[-2], records[-1]
    require(device['kind'] == 'device' and device['uuid'] == UUID and device['runtime_version'] == 13000
            and device['compute_capability_major'] == 8 and device['compute_capability_minor'] == 9
            and device['abi'] == 1 and device['native_build_info'] == NATIVE_BUILD_INFO, 'native device/build identity differs')
    admitted = {}
    for index, p in enumerate(records[1:16]):
        shape, rows = index//3, (1,2,4)[index%3]
        require(p['kind'] == 'plan' and p['shape'] == shape and p['m'] == rows and p['projection'] == SHAPES[shape][0]
                and p['workspace_cap'] == (16*1024*1024 if rows == 1 else 0), 'plan coverage/config differs')
        require(type(p['admitted']) is bool, 'plan admission not boolean')
        admitted[shape,rows] = p['admitted']
        if p['admitted']:
            require(p['status'] == 0 and p['exact_metadata'] is True, 'admitted plan metadata check failed')
            validate_metadata(p['metadata'], shape, rows)
        else:
            require(rows != 1 and p['status'] == 11 and p['metadata'] is None and p['error'], 'unexpected plan rejection')
    executed, mismatches, exact = 0, 0, True
    for c, r in zip(m['case_records'], records[16:-2]):
        require(r['kind'] == 'case' and all(r[k] == c[k] for k in ('case_id','layer','shape','m','active_rows','n','k','pattern')),
                'native case coverage differs')
        if not admitted[c['shape'],c['m']]:
            require(r['executed'] is False and r['reason'] == 'anchored_descriptor_not_supported', 'unsupported case was executed')
            exact = False; continue
        executed += 1
        require(r['executed'] is True and r['words'] == c['m']*c['n'] and len(r['rows']) == c['m'], 'case output coverage differs')
        count = 0
        for index, row in enumerate(r['rows']):
            require(row['row'] == index and row['words'] == c['n'] and type(row['mismatches']) is int
                    and 0 <= row['mismatches'] <= c['n'], 'row comparison coverage differs')
            count += row['mismatches']
            if row['mismatches']:
                require(type(row['first_mismatch']) is int and 0 <= row['first_mismatch'] < c['n']
                        and all(type(row[k]) is int and 0 <= row[k] < 65536 for k in ('oracle_bits','batched_bits'))
                        and row['oracle_bits'] != row['batched_bits'], 'mismatch evidence differs')
            else: require(all(row[k] is None for k in ('first_mismatch','oracle_bits','batched_bits')), 'equal row has a mismatch')
        require(r['mismatches'] == count and r['exact_outputs'] is (count == 0), 'case mismatch total differs')
        require(all(type(r[k]) is bool for k in CASE_FLAGS), 'case flags not boolean')
        require(all(r[k] is True for k in CASE_FLAGS[2:]), 'memory integrity/accounting failed')
        exact = exact and all(r[k] for k in CASE_FLAGS); mismatches += count
    require(end['kind'] == 'runtime_end' and summary['kind'] == 'summary' and summary['completed'] is True
            and summary['cases'] == 1089 and summary['all_plans_admitted'] is all(admitted.values())
            and summary['all_outputs_exact'] is exact and summary['all_resources_closed'] is True and summary['failure'] == ''
            and summary['performance_measured'] is False and summary['performance_claim_eligible'] is False, 'native summary differs')
    return {'device': device, 'runtime_end': end, 'summary': summary, 'executed_cases': executed,
            'skipped_cases': 1089-executed, 'mismatched_words': mismatches, 'arithmetic_equal': exact}


def validate_maps(text, compiled, runtime):
    observed = {}
    for name, prefix in (('driver','libcuda.so.'), ('cudart','libcudart.so.'), ('cublaslt','libcublasLt.so.')):
        paths = {line.split(None,5)[5] for line in text.splitlines() if len(line.split(None,5)) == 6
                 and Path(line.split(None,5)[5]).name.startswith(prefix)}
        require(len(paths) == 1, 'missing/ambiguous actual loaded library: '+name)
        actual = evidence(Path(next(iter(paths))))
        expected = evidence(runtime['libcuda']['path']) if name == 'driver' else compiled['link']['libraries'][name]
        require(actual == expected, 'actual loaded runtime differs: '+name)
        observed[name] = actual
    require(observed['driver']['sha256'] == DRIVER_SHA, 'private driver differs')
    return observed


def run_probe(path, timeout):
    path = path.resolve(strict=True); m = validate_manifest(path); compiled = validate_compile(path, m)
    root = Path(m['root']); child = load(CHILD, CHILD_SHA, '_multisequence_runtime')
    directory = root/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu'
    runtime = child.qualifier.verify_runtime(root, directory)
    output = path.parent/'run'; output.mkdir(mode=0o700)
    env = os.environ.copy()
    for key in ('LD_PRELOAD','LD_AUDIT'): env.pop(key,None)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys([str(directory), *[str(Path(v['path']).parent) for v in compiled['link']['libraries'].values()]]))
    argv = [compiled['binary']['path'], '--cases', m['case_index']['path'], '--device', '0']
    write(output/'launch.json', {'argv': argv, 'cwd': str(path.parent), 'runtime': runtime,
          'environment_sha256': hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest(),
          'selected_environment': {k:env[k] for k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH')}, 'timeout_seconds': timeout})
    process = None
    try:
        with (output/'native.jsonl').open('x') as log, (output/'stderr.log').open('x') as errors:
            process = subprocess.Popen(argv, cwd=path.parent, env=env, stdout=log, stderr=errors, start_new_session=True)
            try: code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
                raise ValueError('native probe exceeded total timeout; partial raw output preserved')
        require(code == 0, 'native probe failed; raw output/cleanup summary preserved')
        write(output/'process-exit.json', {'pid': process.pid, 'returncode': code, 'owned_process_exited': process.poll() is not None})
        records = [json.loads(line, object_pairs_hook=unique) for line in (output/'native.jsonl').read_text().splitlines()]
        result = validate_records(records,m)
        before = validate_maps(result['device']['maps'],compiled,runtime)
        after = validate_maps(result['runtime_end']['maps'],compiled,runtime)
        require(before == after, 'loaded runtime changed')
        require(child.qualifier.verify_runtime(root,directory) == runtime, 'private runtime changed')
        validate_manifest(path); require(validate_compile(path,m) == compiled, 'compiled probe changed')
        receipt = {'schema_version': SCHEMA+'-result.v1', 'completed': True, 'source_build': m['source_build'],
                   'source_commit': m['source_commit'], 'prerequisites': m['prerequisites'], 'model': m['model'],
                   'manifest': evidence(path), 'compile': evidence(path.parent/'build/compile.json'), 'binary': compiled['binary'],
                   'runner': m['runner'], 'native_source': m['native_source'], 'raw': evidence(output/'native.jsonl'),
                   'stderr': evidence(output/'stderr.log'), 'launch': evidence(output/'launch.json'), 'loaded_libraries': before,
                   'process_exit': evidence(output/'process-exit.json'),
                   **COUNTS, **result, 'supported_descriptors_are_not_a_numerical_proof': True,
                   'full_model_multisequence_qualified': False, 'serving_qualified': False,
                   'performance_measured': False, 'performance_claim_eligible': False}
        write(output/'receipt.json',receipt); return receipt
    except Exception as error:
        write(output/'failure.json', {'completed': False, 'error': str(error), 'returncode': process.returncode if process else None,
              'owned_process_exited': process is None or process.poll() is not None, 'performance_claim_eligible': False})
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        if not (output/'process-exit.json').exists():
            write(output/'process-exit.json', {'pid': process.pid if process else None,
                  'returncode': process.returncode if process else None,
                  'owned_process_exited': process is None or process.poll() is not None})


def validate_receipt(path):
    r = read(path)
    require(r['schema_version'] == SCHEMA+'-result.v1' and r['completed'] is True, 'experiment did not complete')
    require(all(r[k] == v for k,v in COUNTS.items()), 'receipt counts differ')
    for key in ('source_build','model','manifest','compile','binary','runner','native_source','raw','stderr','launch','process_exit'):
        check(r[key])
    manifest_path = Path(r['manifest']['path']); m = validate_manifest(manifest_path)
    compiled = validate_compile(manifest_path,m)
    require(r['compile'] == evidence(manifest_path.parent/'build/compile.json') and r['binary'] == compiled['binary'],
            'receipt compiled probe differs')
    require(all(r[k] == m[k] for k in ('source_build','source_commit','prerequisites','model','runner','native_source')),
            'receipt source/model proof differs')
    records = [json.loads(line,object_pairs_hook=unique) for line in Path(r['raw']['path']).read_text().splitlines()]
    actual = validate_records(records,m)
    require(all(r[k] == v for k,v in actual.items()), 'receipt differs from raw numerical observations')
    launch = read(r['launch']['path']); exited = read(r['process_exit']['path'])
    require(launch['argv'] == [compiled['binary']['path'],'--cases',m['case_index']['path'],'--device','0']
            and launch['cwd'] == str(manifest_path.parent) and type(launch['timeout_seconds']) is int
            and 1 <= launch['timeout_seconds'] <= 3600 and type(exited['pid']) is int and exited['pid'] > 0
            and exited['returncode'] == 0 and exited['owned_process_exited'] is True, 'probe launch/exit differs')
    child = load(CHILD,CHILD_SHA,'_multisequence_receipt_runtime'); root = Path(m['root'])
    directory = root/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu'
    runtime = child.qualifier.verify_runtime(root,directory)
    require(launch['runtime'] == runtime and r['loaded_libraries'] == validate_maps(actual['device']['maps'],compiled,runtime)
            == validate_maps(actual['runtime_end']['maps'],compiled,runtime), 'receipt runtime differs')
    expected_ld = ':'.join(dict.fromkeys([str(directory), *[str(Path(v['path']).parent) for v in compiled['link']['libraries'].values()]]))
    require(launch['selected_environment'] == {'CUDA_VISIBLE_DEVICES':'0','LD_LIBRARY_PATH':expected_ld}, 'runtime search path differs')
    require(r['supported_descriptors_are_not_a_numerical_proof'] is True
            and all(r[k] is False for k in ('full_model_multisequence_qualified','serving_qualified','performance_measured','performance_claim_eligible')),
            'experiment scope differs')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    prepare_parser=sub.add_parser('prepare')
    for flag in ('root','base','output-dir'): prepare_parser.add_argument('--'+flag,type=Path,required=True)
    b=sub.add_parser('build'); b.add_argument('--manifest',type=Path,required=True)
    b.add_argument('--cxx',type=Path,required=True);b.add_argument('--nvcc',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--manifest',type=Path,required=True);r.add_argument('--timeout',type=int,default=1800)
    v=sub.add_parser('validate');v.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.command=='prepare': result=prepare(args.root,args.base,args.output_dir)
        elif args.command=='build': result=build_probe(args.manifest.resolve(strict=True),args.cxx,args.nvcc)
        elif args.command=='validate': result=validate_receipt(args.receipt.resolve(strict=True))
        else:
            require(1 <= args.timeout <= 3600,'timeout outside 1..3600 seconds');result=run_probe(args.manifest,args.timeout)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as e:
        print('error: '+str(e),file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
