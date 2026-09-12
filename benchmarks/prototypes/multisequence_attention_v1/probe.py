#!/usr/bin/env python3
"""Isolated true-row attention experiment: prepare/build/run are separate.

Accepted Batch7 M1 is compiled unchanged. The candidate is mechanically derived
in a distinct namespace. This is synthetic arithmetic/mapping feasibility, not
an owner implementation, model qualification, or performance measurement.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = 'kernels/src/graph_numerics.cu'
COMMIT = '1a2be0df01fe49daa4d4db155ad5c44f34ead6df'
BUILD_SHA = '4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395'
SOURCE_SHA = 'a1bb90862e9eb6bb378ac36843c1d94b05b59f078d4a1f4b9de8c2018a1f666e'
CONVENTIONS_SHA = 'bf8f8deca087d6db53c6cce9238d05b2d014328b3f0000f864cbf30ebc56e7f1'
CONTRACT_SHA = '5bed6d70342dec4f30c85ba67dd41cf560e6b7f4b0de5dedfe59424cecc9b641'
DRIVER_SHA = '266916dac6c5e7e4655526570cf303a6cc017880d2552960abc66017a7c98cf4'
UUID = '9087e4256acab722b8c9cc0423b39fb0'
SCHEMA = 'riley.multisequence-attention'
MODES = ((1, 1), (2, 2), (4, 3), (4, 4))
PATTERNS = ('bounded_fingerprint', 'signed_zero_impulses', 'tail_cancellation')
MAPPINGS = ('identity64', 'reverse64', 'affine7_3_mod64')
COUNTS = {'cases': 34560, 'layers': 30, 'position_tuples': 32, 'patterns': 3, 'mappings': 3, 'modes': 4}
FLAGS = ('exact_outputs', 'finite_outputs', 'guards_intact', 'inputs_unchanged', 'kv_unchanged',
         'oracle_outputs_unchanged', 'inactive_output_zero', 'mapping_invariant', 'all_allocations_freed')
TRANSITION_FLAGS = FLAGS[:-2]
TRANSITIONS = ((0, 0, 0, 2, 0), (15, 17, 1, 3, 2), (29, 5, 2, 2, 1))
INJECTION = ''' const uint32_t row=blockIdx.z;
 out+=row*576;
 const uint32_t active_rows=packet[6]; // fresh common header, byte24
 // Uniform inactive return precedes row metadata/Q/KV loads and every CTA barrier.
 if(row>=active_rows){
  if(threadIdx.x<32)out[blockIdx.x*64+blockIdx.y*32+threadIdx.x]=__ushort_as_bfloat16(0);
  return;
 }
 q+=row*576;
 const uint32_t* metadata=packet+32+row*32; // byte128 + row*128: C10 view
'''
OLD_SIGNATURE = '__global__ void attention_packed_decode_two_warp(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* metadata){\n'
NEW_SIGNATURE = '__global__ void attention_rows_v1(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* packet){\n'
WRAPPER = '''cudaError_t enqueue_rows(cudaStream_t s,const void* q,const void* k,const void* v,void* out,const void* packet,uint32_t bucket) noexcept {
 if(!q||!k||!v||!out||!packet||(bucket!=1&&bucket!=2&&bucket!=4))return cudaErrorInvalidValue;
 attention_rows_v1<<<dim3(9,2,bucket),64,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,(const uint32_t*)packet);
 return cudaGetLastError();
}
'''


def require(ok, message):
    if not ok: raise ValueError(message)


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
        require(key not in result, 'duplicate JSON key'); result[key] = value
    return result


def read(path):
    def invalid(value): raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def check(item):
    require(evidence(item['path']) == item, 'artifact changed: ' + str(item['path']))


def load_conventions(path):
    require(sha(path) == CONVENTIONS_SHA, 'frozen compile conventions differ')
    spec = importlib.util.spec_from_file_location('_nrow_attention_conventions', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def transform(raw):
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'requires exact accepted Batch7 full TU')
    source = raw.decode()
    require(source.count(OLD_SIGNATURE) == 1, 'accepted kernel anchor differs')
    helper_start = source.index('__device__ float exponential(')
    helper_end = source.index('__global__ void attention(')
    start = source.index(OLD_SIGNATURE)
    end = source.index('\n}\n#endif\n', start) + 3
    kernel = source[start:end]
    changed = kernel.replace(OLD_SIGNATURE, NEW_SIGNATURE + INJECTION, 1)
    require(changed.replace(NEW_SIGNATURE + INJECTION, OLD_SIGNATURE, 1) == kernel, 'arithmetic body changed')
    prefix = source[:helper_start]
    helpers = source[helper_start:helper_end]
    generated = (prefix + '#if MODE != 6\n#error "This experiment requires accepted MODE6 arithmetic"\n#endif\n'
                 + 'namespace riley_multisequence_attention_probe {\n' + helpers + changed + WRAPPER
                 + '} // namespace riley_multisequence_attention_probe\n')
    return generated.encode(), {'accepted_kernel_sha256': hashlib.sha256(kernel.encode()).hexdigest(),
        'helper_bytes_sha256': hashlib.sha256(helpers.encode()).hexdigest(),
        'arithmetic_body_preserved_by_reverse_transform': True,
        'changes': ['namespace isolation', 'grid.z row selection and exact Q/output stride576 BF16',
                    'fresh active count from common packet header byte24; fixed cold bucket grid',
                    'C10 metadata pointer at packet byte128+128*row', 'uniform inactive positive-zero output before row reads/barriers']}


def case_rows():
    for i in range(COUNTS['cases']):
        mode = (i//3)%4; bucket, active = MODES[mode]; position = (i//36)%32
        yield {'case_id': i, 'layer': i//1152, 'position_tuple': position, 'pattern': PATTERNS[(i//12)%3],
               'mapping': MAPPINGS[i%3], 'bucket': bucket, 'active_rows': active,
               'positions': [128+(position+11*r)%32 for r in range(active)],
               'numerical_only_position159': any((position+11*r)%32 == 31 for r in range(active))}


def case_index():
    return ''.join(f'{i}\t{i//1152}\t{(i//36)%32}\t{(i//12)%3}\t{(i//3)%4}\t{i%3}\n' for i in range(COUNTS['cases']))


def transition_rows():
    for step, (layer, position, pattern, mode, mapping) in enumerate(TRANSITIONS):
        bucket, active = MODES[mode]
        yield {'step': step, 'layer': layer, 'position_tuple': position, 'pattern': PATTERNS[pattern],
               'mapping': MAPPINGS[mapping], 'bucket': bucket, 'active_rows': active,
               'positions': [128+(position+11*r)%32 for r in range(active)],
               'graph_nodes': 1, 'capture_count': 1, 'parameter_updates': 0, 'replay_index': step+1}


def physical(row, logical, mapping):
    index = row*10 + logical
    return index if mapping == 0 else 63-index if mapping == 1 else (7*index+3)%64


def recipe():
    return {'synthetic_only_no_checkpoint_activations': True, 'q_output_bytes_per_row': 1152,
            'metadata_bytes': 1280, 'row_offset': 128, 'row_stride': 128, 'c10_bytes': 80,
            'block_ids_offset': 16, 'valid_counts_offset': 56, 'shared_physical_blocks': 64,
            'shared_kv_bytes_each': 393216, 'guard_bytes_each_side': 256,
            'modes': [list(x) for x in MODES], 'positions': '128+((tuple+11*row)%32)',
            'position159': 'numerical-only; not admitted by public P128/O1..32 descriptor codec',
            'fixture_authority': 'host-owned synthetic rows and disjoint physical ledger; no live scheduler/KV integration',
            'inactive_device_poison': 'canonical packet validated first, then inactive metadata set to ff and inactive Q to BF16 NaN; not production admission',
            'arithmetic_allocation_count': sum(5+3*c['active_rows'] for c in case_rows()),
            'graph_transition_allocation_count': 17,
            'allocation_count': sum(5+3*c['active_rows'] for c in case_rows())+17,
            'active_count_binding': 'fresh common device header packet[6], byte24; no captured active scalar',
            'retained_graph': {'transitions': list(transition_rows()), 'fixture_coordinates': [list(x) for x in TRANSITIONS],
                'capture_count': 1, 'kernel_nodes': 1, 'parameter_updates': 0,
                'refresh': 'same fixed parent allocations; fresh H2D outside capture; no owner integration'},
            'oracle': 'independent aligned Q/C10/output for each active row; same immutable shared KV pool',
            'candidate_launches_per_case': 1, 'candidate_grid': '9 x 2 x bucket, 64 threads per CTA'}


def source_build(build_path, conventions):
    require(sha(build_path) == BUILD_SHA, 'accepted Batch7 build receipt differs')
    build = conventions.check_snapshot(build_path)
    require(build['source_commit'] == COMMIT and build['source_files'][SOURCE] == SOURCE_SHA, 'not accepted Batch7 source')
    require(sha(Path(build['source_root'])/SOURCE) == SOURCE_SHA, 'accepted attention bytes differ')
    return build


def prepare(build_path, conventions_path, contract_path, output):
    conventions = load_conventions(conventions_path)
    build = source_build(build_path, conventions)
    require(sha(contract_path) == CONTRACT_SHA, 'frozen descriptor contract differs')
    source = Path(build['source_root']); output = Path(output).resolve()
    require(not output.is_relative_to(source), 'output must be outside accepted snapshot')
    generated, lineage = transform((source/SOURCE).read_bytes())
    output.mkdir(mode=0o700)
    for path, raw in [(output/'candidate.cu', generated), (output/'ffi_internal.hpp', (source/'kernels/src/ffi_internal.hpp').read_bytes()), (output/'cases.tsv', case_index().encode())]:
        with path.open('xb') as f: f.write(raw)
    manifest = {'schema_version': SCHEMA+'-fixtures.v1', **COUNTS, 'recipe': recipe(),
        'source_build': evidence(build_path), **{key: build[key] for key in ('source_root', 'source_commit', 'source_files', 'binaries')},
        'runner': evidence(__file__), 'host_source': evidence(HERE/'native.cu'), 'conventions': evidence(conventions_path),
        'descriptor_contract': evidence(contract_path), 'generated_source': evidence(output/'candidate.cu'),
        'generated_header': evidence(output/'ffi_internal.hpp'), 'case_index': evidence(output/'cases.tsv'), 'lineage': lineage,
        'gpu_executed': False, 'performance_claim': False, 'production_integration': False}
    write(output/'fixtures.json', manifest)
    validate_manifest(output/'fixtures.json')
    return manifest


def validate_manifest(path):
    path = Path(path).resolve(strict=True)
    manifest = read(path)
    require(manifest['schema_version'] == SCHEMA+'-fixtures.v1' and all(manifest[k] == v for k,v in COUNTS.items()), 'fixture coverage/schema')
    for key in ('source_build', 'runner', 'host_source', 'conventions', 'descriptor_contract', 'generated_source', 'generated_header', 'case_index'): check(manifest[key])
    require(manifest['runner'] == evidence(__file__) and manifest['host_source'] == evidence(HERE/'native.cu'), 'current helper differs')
    for key, name in [('generated_source','candidate.cu'), ('generated_header','ffi_internal.hpp'), ('case_index','cases.tsv')]:
        require(manifest[key] == evidence(path.parent/name), 'generated file is outside the bound fixture directory')
    require(manifest['descriptor_contract']['sha256'] == CONTRACT_SHA, 'descriptor contract differs')
    build = source_build(manifest['source_build']['path'], load_conventions(manifest['conventions']['path']))
    require(all(manifest[k] == build[k] for k in ('source_root', 'source_commit', 'source_files', 'binaries')), 'build source binding')
    generated, lineage = transform((Path(build['source_root'])/SOURCE).read_bytes())
    require(Path(manifest['generated_source']['path']).read_bytes() == generated and manifest['lineage'] == lineage, 'derived arithmetic/source changed')
    require(Path(manifest['generated_header']['path']).read_bytes() == (Path(build['source_root'])/'kernels/src/ffi_internal.hpp').read_bytes(), 'header copy differs')
    require(Path(manifest['case_index']['path']).read_text() == case_index() and manifest['recipe'] == recipe(), 'case fixture differs')
    require(manifest['gpu_executed'] is False and manifest['performance_claim'] is False and manifest['production_integration'] is False, 'fixture scope')
    return manifest


def compile_commands(manifest, nvcc, options, directory, runtime):
    objects = [directory/name for name in ('oracle.o', 'candidate.o', 'host.o')]
    arch = [flag for flag in options['flags'] if 'arch=compute_89' in flag]
    return [[str(nvcc), *options['flags'], '-x', 'cu', '-c', str(Path(manifest['source_root'])/SOURCE), '-o', str(objects[0])],
            [str(nvcc), *options['flags'], '-x', 'cu', '-c', manifest['generated_source']['path'], '-o', str(objects[1])],
            [str(nvcc), '-std=c++17', '-O2', *arch, '-c', manifest['host_source']['path'], '-o', str(objects[2])],
            [str(nvcc), '--cudart=shared', *map(str, objects), '-L'+str(runtime.parent), '-Xlinker', '-rpath', '-Xlinker', str(runtime.parent), '-o', str(directory/'probe')]]


def build_probe(path, nvcc):
    path = Path(path).resolve(strict=True); manifest = validate_manifest(path)
    nvcc = Path(nvcc).resolve(strict=True)
    conventions = load_conventions(manifest['conventions']['path'])
    options = conventions.production_flags(manifest, nvcc)
    version = subprocess.check_output([str(nvcc), '--version'], text=True)
    require(re.search(r'release 13\.', version), 'requires production CUDA13 compiler')
    runtime = Path(options['runtime_library_link']); directory = path.parent/'build'; directory.mkdir()
    commands = compile_commands(manifest, nvcc, options, directory, runtime)
    with (directory/'compile.log').open('x') as log:
        for command in commands: subprocess.run(command, cwd=directory, stdout=log, stderr=log, check=True)
    require(validate_manifest(path) == manifest, 'source changed during compile')
    for item in options['cmake_artifacts']: check(item)
    result = {'schema_version': SCHEMA+'-build.v1', 'fixture_manifest': evidence(path), 'compiler': evidence(nvcc),
              'compiler_version': version, 'options': options, 'commands': commands, 'runtime': evidence(runtime),
              'runtime_link': str(runtime), 'objects': [evidence(directory/name) for name in ('oracle.o','candidate.o','host.o')],
              'binary': evidence(directory/'probe'), 'log': evidence(directory/'compile.log'), 'gpu_executed': False, 'performance_claim': False}
    write(path.parent/'compile.json', result); return result


def validate_compile(path, manifest_path):
    manifest = validate_manifest(manifest_path); compiled = read(path)
    require(compiled['schema_version'] == SCHEMA+'-build.v1' and compiled['fixture_manifest'] == evidence(manifest_path), 'compile fixture binding')
    for key in ('compiler', 'runtime', 'binary', 'log'): check(compiled[key])
    for item in compiled['objects'] + compiled['options']['cmake_artifacts']: check(item)
    options = load_conventions(manifest['conventions']['path']).production_flags(manifest, Path(compiled['compiler']['path']))
    require(options == compiled['options'], 'production flags changed')
    runtime = Path(options['runtime_library_link'])
    require(str(runtime) == compiled['runtime_link'] and evidence(runtime) == compiled['runtime'], 'shared runtime differs')
    require(compiled['commands'] == compile_commands(manifest, Path(compiled['compiler']['path']), options, Path(path).parent/'build', runtime), 'compile commands differ')
    require(compiled['objects'] == [evidence(Path(path).parent/'build'/name) for name in ('oracle.o','candidate.o','host.o')], 'object set differs')
    require(compiled['binary'] == evidence(Path(path).parent/'build/probe') and compiled['log'] == evidence(Path(path).parent/'build/compile.log'), 'binary/log path differs')
    require(compiled['gpu_executed'] is False and compiled['performance_claim'] is False, 'compile scope')
    return compiled


def validate_maps(lines, driver, runtime):
    files = {}
    for line in lines:
        fields = line.split(None, 5)
        require(len(fields) == 6 and not fields[5].endswith(' (deleted)'), 'native runtime mapping malformed')
        path = Path(fields[5]).resolve(strict=True); info = path.stat()
        require((os.major(info.st_dev), os.minor(info.st_dev)) == tuple(int(x,16) for x in fields[3].split(':')) and info.st_ino == int(fields[4]), 'native runtime mapping identity changed')
        item = evidence(path)
        require(item in (driver, runtime), 'unselected driver/runtime actually loaded')
        files[str(path)] = item
    require(set(files) == {driver['path'], runtime['path']}, 'driver/runtime actual mappings incomplete')
    return files


def validate_raw(path):
    records = [json.loads(line, object_pairs_hook=unique) for line in Path(path).read_text().splitlines() if line]
    require(len(records) == COUNTS['cases']+len(TRANSITIONS)+2 and all(r['schema'] == SCHEMA+'-native.v1' for r in records), 'incomplete native records')
    device, summary = records[0], records[-1]
    require(device['kind'] == 'device' and device['uuid_hex'] == UUID and device['compute_major'] == 8 and device['compute_minor'] == 9 and device['runtime_version'] == 13000, 'GPU/runtime differs')
    failed = 0
    def validate_comparison(row, expected, flags):
        require(row['oracle_invocations'] == expected['active_rows'], 'not repeated M1 comparison')
        require(row['active_words_compared'] == expected['active_rows']*576 and row['inactive_words_checked'] == (expected['bucket']-expected['active_rows'])*576, 'partial output comparison')
        require(all(type(row[k]) is bool for k in flags) and type(row['passed']) is bool and type(row['mismatch_words']) is int and 0 <= row['mismatch_words'] <= expected['active_rows']*576, 'case flag/mismatch shape')
        require(row['exact_outputs'] == (row['mismatch_words'] == 0) and row['passed'] == all(row[k] for k in flags), 'case success flags inconsistent')
        require(('first_mismatch' in row) == (row['mismatch_words'] > 0), 'first mismatch evidence missing/spurious')
        if row['mismatch_words']:
            first = row['first_mismatch']; require(0 <= first['row'] < expected['active_rows'] and 0 <= first['word'] < 576
                and 0 <= first['oracle_bits'] <= 65535 and 0 <= first['candidate_bits'] <= 65535 and first['oracle_bits'] != first['candidate_bits'], 'first mismatch invalid')
        return not row['passed']
    for row, expected in zip(records[1:1+COUNTS['cases']], case_rows()):
        require(row['kind'] == 'case' and all(row[k] == v for k,v in expected.items()), 'case identity/coverage differs')
        require(row['candidate_invocations'] == 1, 'not true-row candidate comparison')
        failed += validate_comparison(row, expected, FLAGS)
    transition_failed = 0
    for row, expected in zip(records[1+COUNTS['cases']:-1], transition_rows()):
        require(row['kind'] == 'graph_transition' and all(row[k] == v for k,v in expected.items()), 'retained graph transition differs')
        transition_failed += validate_comparison(row, expected, TRANSITION_FLAGS)
    require(summary['kind'] == 'summary' and summary['completed'] is True and all(summary[k] == v for k,v in COUNTS.items()), 'native summary incomplete')
    require(summary['graph_transitions'] == len(TRANSITIONS) and summary['failed_graph_transitions'] == transition_failed
            and all(summary[k] == 1 for k in ('graphs_created','graphs_destroyed','execs_created','execs_destroyed')), 'retained graph completion differs')
    require(summary['failed_cases'] == failed and summary['attention_bitwise_equal'] == (failed == transition_failed == 0), 'numeric summary inconsistent')
    require(summary['allocations_created'] == summary['allocations_freed'] == recipe()['allocation_count'] and summary['live_bytes'] == summary['cleanup_errors'] == 0 and summary['stream_destroyed'] is True, 'cleanup incomplete')
    require(summary['error'] == '' and summary['performance_claim'] is False, 'native error/scope')
    return device, summary


def run_probe(path, driver_directory, timeout):
    require(type(timeout) is int and timeout > 0, 'positive bounded timeout required')
    path = Path(path).resolve(strict=True); manifest = validate_manifest(path); compiled = validate_compile(path.parent/'compile.json', path)
    driver_dir = Path(driver_directory).resolve(strict=True); driver = evidence(driver_dir/'libcuda.so.1')
    require(driver['sha256'] == DRIVER_SHA and not os.environ.get('LD_PRELOAD'), 'private driver/LD_PRELOAD differs')
    runtime = compiled['runtime']; env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(driver_dir)+':'+str(Path(compiled['runtime_link']).parent)
    command = [compiled['binary']['path'], '--cases', manifest['case_index']['path'], '--device', '0']
    raw, errors = path.parent/'native-results.jsonl', path.parent/'native-stderr.log'
    require(not any((path.parent/name).exists() for name in ('native-results.jsonl','native-stderr.log','process-exit.json','result.json')), 'run artifacts already exist; no overwrite/retry in place')
    process = None; failure = None; cleanup_failure = None
    try:
        with raw.open('x') as stdout, errors.open('x') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            process.wait(timeout=timeout)
            require(process.returncode == 0, 'native probe incomplete')
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        cleanup_exception = None
        try:
            if process is not None and process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait(timeout=10)
        except BaseException as error:
            cleanup_failure = {'type': type(error).__name__, 'message': str(error)}
            cleanup_exception = error
        finally:
            write(path.parent/'process-exit.json', {'failure': failure, 'cleanup_failure': cleanup_failure,
                'returncode': process.returncode if process else None,
                'owned_process_reaped': process is not None and process.poll() is not None,
                'process_scope': 'single native executable; native source spawns no child processes'})
        if cleanup_exception is not None and failure is None:
            raise cleanup_exception
    require(validate_manifest(path) == manifest and validate_compile(path.parent/'compile.json', path) == compiled, 'probe inputs changed while running')
    device, summary = validate_raw(raw)
    before = validate_maps(device['runtime_maps'], driver, runtime); after = validate_maps(summary['runtime_maps'], driver, runtime)
    require(before == after and evidence(driver_dir/'libcuda.so.1') == driver, 'loaded runtime changed')
    result = {'schema_version': SCHEMA+'-result.v1', 'completed': True, 'attention_bitwise_equal': summary['attention_bitwise_equal'],
              'fixture_manifest': evidence(path), 'compile_receipt': evidence(path.parent/'compile.json'),
              'driver': driver, 'runtime': runtime, 'loaded_native_files': before, 'command': command,
              'native_results': evidence(raw), 'stderr': evidence(errors), 'process_exit': evidence(path.parent/'process-exit.json'),
              'device': device, 'summary': summary, 'gpu_executed': True, 'performance_claim': False, 'production_integration': False}
    write(path.parent/'result.json', result)
    return validate_result(path.parent/'result.json')


def validate_result(path):
    result = read(path)
    require(result['schema_version'] == SCHEMA+'-result.v1' and result['completed'] is True
            and result['gpu_executed'] is True and result['performance_claim'] is False
            and result['production_integration'] is False, 'result scope/completion')
    for key in ('fixture_manifest', 'compile_receipt', 'driver', 'runtime', 'native_results', 'stderr', 'process_exit'): check(result[key])
    manifest_path = Path(result['fixture_manifest']['path']); manifest = validate_manifest(manifest_path)
    compiled = validate_compile(result['compile_receipt']['path'], manifest_path)
    require(result['runtime'] == compiled['runtime'] and result['driver']['sha256'] == DRIVER_SHA, 'result runtime binding')
    process = read(result['process_exit']['path'])
    require(process['failure'] is None and process['cleanup_failure'] is None and type(process['returncode']) is int
            and process['returncode'] == 0 and process['owned_process_reaped'] is True, 'owned probe incomplete')
    device, summary = validate_raw(result['native_results']['path'])
    require(result['device'] == device and result['summary'] == summary
            and result['attention_bitwise_equal'] == summary['attention_bitwise_equal'], 'result/raw summary differs')
    maps = validate_maps(device['runtime_maps'], result['driver'], result['runtime'])
    require(maps == validate_maps(summary['runtime_maps'], result['driver'], result['runtime'])
            and result['loaded_native_files'] == maps, 'native maps changed')
    require(result['command'] == [compiled['binary']['path'], '--cases', manifest['case_index']['path'], '--device', '0'], 'result command differs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    for flag in ('build-receipt','conventions','descriptor-contract','output'): prepare_parser.add_argument('--'+flag, type=Path, required=True)
    build_parser = sub.add_parser('build'); build_parser.add_argument('--manifest', type=Path, required=True); build_parser.add_argument('--nvcc', type=Path, required=True)
    run_parser = sub.add_parser('run'); run_parser.add_argument('--manifest', type=Path, required=True); run_parser.add_argument('--driver-library-dir', type=Path, required=True); run_parser.add_argument('--timeout', type=int, default=3600)
    validate_parser = sub.add_parser('validate'); validate_parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, _frame): raise InterruptedError('signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    if args.command == 'prepare': result = prepare(args.build_receipt, args.conventions, args.descriptor_contract, args.output)
    elif args.command == 'build': result = build_probe(args.manifest, args.nvcc)
    elif args.command == 'run': result = run_probe(args.manifest, args.driver_library_dir, args.timeout)
    else: result = validate_result(args.result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
