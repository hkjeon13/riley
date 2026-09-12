import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2] / 'results/20260912-serving-optimization'
SPEC = importlib.util.spec_from_file_location('common_prefix_analyzer', ROOT / 'analyze_common_prefix_logits.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def remote_ref(path):
    return {'path': '/remote/' + path.parent.name + '/' + path.name, 'sha256': a.sha(path)}


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.sources = a.frozen_inputs(ROOT / 'numerical-divergence-cases.json', ROOT / 'common-prefix-reference-source.json', ROOT / 'capture_common_prefix_logits.py')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapper = a.PathMap(['/remote=' + str(self.root)])
        self.dirs = {dtype: self.make_run(dtype, pid) for dtype, pid in [('fp32', 10001), ('bf16', 10002)]}

    def tensor(self, directory, case, key, dtype):
        suffix = {'first_layer_hidden': 'hidden'}.get(key, key)
        shape = [case['input_length'], 576] if key == 'first_layer_hidden' else [a.VOCAB]
        values = [-4.0] * math.prod(shape)
        values[0] = -0.0
        if key == 'logits':
            values[case['reference_token_id']] = 1.25
            values[case['observed_token_ids'][0]] = 1.0 if dtype == 'float32' else 1.5
        if dtype == 'float32':
            raw = struct.pack('<' + 'f' * len(values), *values)
        else:
            words = [struct.unpack('<I', struct.pack('<f', v))[0] >> 16 for v in values]
            raw = struct.pack('<' + 'H' * len(words), *words)
        path = directory / (case['case_id'] + '.' + suffix + '.bin')
        path.write_bytes(raw)
        return {**remote_ref(path), 'shape': shape, 'dtype': dtype, 'byte_order': 'little', 'bytes': len(raw)}, values

    def make_run(self, dtype, pid):
        directory = self.root / dtype
        directory.mkdir()
        gate = {'preparation_sha256': '1'*64, 'finalization_sha256': '2'*64,
                'restoration': {'path': '/remote/blender/verified.json', 'sha256': '3'*64},
                'process_exit_receipts': {'/remote/campaign/lane/process-exit.json': '4'*64},
                'performance_or_acceptance_inferred': False}
        installed = {name: {'version': '1.0', 'files': {'/unavailable/' + name + '.py': '5'*64}} for name in a.DEPENDENCIES}
        loaded = {name: {'path': '/unavailable/' + name + '.py', 'sha256': '5'*64} for name in a.DEPENDENCIES}
        for name in ('torch._C', 'concrete_model_class'):
            installed['torch']['files']['/unavailable/' + name + '.so'] = '5'*64
            loaded[name] = {'path': '/unavailable/' + name + '.so', 'sha256': '5'*64}
        vendor = {'/private/libcuda.so.580.173.02': {'sha256': '6'*64, 'device': '08:01', 'inode': 123}}
        maps = {'ready': False, 'compute_loaded': True, 'all_observed_vendor_mappings_pinned': True,
                'private_vendor_files': vendor, 'driver_mappings': ['0000-1000 r-xp 0000 08:01 123 /private/libcuda.so.580.173.02']}
        metadata = {'python_version': '3.11.0', 'python_executable_sha256': '7'*64, 'python_platform_system': 'linux',
                    'python_platform_machine': 'x86_64', 'torch_version': '1.0', 'transformers_version': '1.0',
                    'safetensors_version': '1.0', 'config_sha256': '8'*64, 'tokenizer_sha256': '9'*64,
                    'tokenizer_files_sha256': {'tokenizer.json': 'a'*64}}
        worker = {'campaign_gate': gate, 'case_inputs': a.expected_case_inputs(self.cases, a.PathMap([])),
                  'reference_source_files': self.sources['files'], 'tool': {'path': '/remote/capture_common_prefix_logits.py', 'sha256': a.CAPTURE_SHA},
                  'dtype': dtype, 'case_manifest_sha256': a.CASES_SHA, 'performance_claim': False,
                  'acceptance_gate_changed': False, 'reference_metadata': metadata,
                  'gpu_before': {'uuid': a.GPU_UUID, 'driver_version': '580.173.02'},
                  'gpu_after': {'uuid': a.GPU_UUID, 'driver_version': '580.173.02'},
                  'private_maps_before': maps, 'private_maps_after': maps,
                  'installed_dependencies': installed, 'loaded_implementations': loaded,
                  'cuda_library_files': {'/unavailable/libcudart.so.13': 'b'*64, '/unavailable/libcublas.so.13': 'c'*64},
                  'completed': True, 'backend_closed': True, 'failure': None, 'cleanup_failure': None,
                  'scope': 'independent HF logits only', 'cases': []}
        for case in self.cases['cases']:
            row = {key: case[key] for key in ('case_id', 'generated_index', 'input_token_ids')}
            row.update(execution=a.EXECUTION, retokenized=False, free_generation=False)
            for key in ('logits', 'log_probs', 'first_layer_hidden'):
                ty = 'float32' if dtype == 'fp32' or key == 'log_probs' else 'bfloat16'
                item, values = self.tensor(directory, case, key, ty)
                row[key] = item
                if key == 'logits':
                    row['scores'] = a.score(values, case)
            worker['cases'].append(row)
            write(directory / (case['case_id'] + '.json'), row)
        write(directory / 'worker-receipt.json', worker)
        write(directory / 'finalization.json', {'failure': None, 'cleanup_failure': None,
            'cleanup': {'returncode': 0, 'remaining_owned_pids': [], 'cleanup_verified': True}})
        write(directory / 'launch.json', {'argv': ['/python', worker['tool']['path'], '_worker', '--root', '/remote', '--reference-package', '/reference',
            '--output', '/remote/' + dtype, '--dtype', dtype, '--hf-home', '/cache'],
            'environment': {'HF_HOME': '/cache', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
            'tool': worker['tool'], 'campaign_gate': gate, 'diagnostic_only': True, 'performance_claim': False})
        write(directory / 'process.json', {'pid': pid, 'session_id': pid})
        self.rebind(directory)
        return directory

    def rebind(self, directory, worker=None):
        if worker is not None:
            write(directory / 'worker-receipt.json', worker)
            for row in worker['cases']:
                write(directory / (row['case_id'] + '.json'), row)
        write(directory / 'completion.json', {'schema': 'riley.common-prefix-hf-logits.v1',
            'worker': remote_ref(directory / 'worker-receipt.json'), 'finalization': remote_ref(directory / 'finalization.json'),
            'performance_claim': False, 'acceptance_gate_changed': False})

    def validate(self, dtype='fp32'):
        return a.validate_run(self.dirs[dtype] / 'completion.json', dtype, self.mapper, self.cases, self.sources)

    def test_complete_synthetic_capture_reports_full_vocabulary_without_acceptance(self):
        report = a.analyze(self.dirs['fp32'] / 'completion.json', self.dirs['bf16'] / 'completion.json',
            ROOT / 'numerical-divergence-cases.json', ROOT / 'common-prefix-reference-source.json',
            ROOT / 'capture_common_prefix_logits.py', ['/remote=' + str(self.root)], self.root / 'analysis')
        self.assertTrue(report['analysis_complete'])
        self.assertIsNone(report['accuracy_acceptance_decision'])
        self.assertIsNone(report['numerical_tolerance'])
        self.assertFalse(report['performance_claim'])
        self.assertFalse(report['actual_vllm_graph_logits_captured'])
        for row in report['cases']:
            self.assertEqual(row['logits']['elements'], 49152)
            self.assertEqual(row['logits']['different_value_count'], 1)
            self.assertNotEqual(row['fp32_scores']['argmax_lowest_id_on_tie'], row['bf16_scores']['argmax_lowest_id_on_tie'])
            self.assertEqual(len(Path(row['full_vocabulary']['path']).read_text().splitlines()), 49153)
            self.assertEqual(a.sha(row['full_vocabulary']['path']), row['full_vocabulary']['sha256'])
        self.assertFalse(any(name in __import__('sys').modules for name in ('torch', 'numpy')))

    def test_raw_hash_and_whole_vocabulary_bytes_are_required(self):
        path = next(self.dirs['fp32'].glob('*.logits.bin'))
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, 'artifact hash differs'):
            self.validate()

    def test_worker_finalization_and_launch_contracts_fail_closed(self):
        directory = self.dirs['fp32']; original = a.read(directory / 'worker-receipt.json')
        for key, value in [('completed', False), ('backend_closed', False), ('failure', {}), ('cleanup_failure', {})]:
            worker = copy.deepcopy(original); worker[key] = value; self.rebind(directory, worker)
            with self.assertRaisesRegex(ValueError, 'worker incomplete'):
                self.validate()
        self.rebind(directory, original)
        final = a.read(directory / 'finalization.json'); final['cleanup']['remaining_owned_pids'] = [55]
        write(directory / 'finalization.json', final); self.rebind(directory)
        with self.assertRaisesRegex(ValueError, 'finalization incomplete'):
            self.validate()

    def test_stored_scores_and_exact_prefix_cannot_be_replaced(self):
        directory = self.dirs['fp32']; original = a.read(directory / 'worker-receipt.json')
        worker = copy.deepcopy(original); worker['cases'][0]['scores']['top1_top2_logit_gap'] += 1
        self.rebind(directory, worker)
        with self.assertRaisesRegex(ValueError, 'stored scores'):
            self.validate()
        worker = copy.deepcopy(original); worker['cases'][1]['input_token_ids'][-1] += 1
        self.rebind(directory, worker)
        with self.assertRaisesRegex(ValueError, 'teacher-forced prefix'):
            self.validate()

    def test_wrong_shape_dtype_and_case_sidecar_are_rejected(self):
        directory = self.dirs['bf16']; original = a.read(directory / 'worker-receipt.json')
        for key, value in [('shape', [a.VOCAB-1]), ('dtype', 'float32'), ('byte_order', 'big'), ('bytes', 1)]:
            worker = copy.deepcopy(original); worker['cases'][0]['logits'][key] = value
            self.rebind(directory, worker)
            with self.assertRaisesRegex(ValueError, 'tensor'):
                self.validate('bf16')
        self.rebind(directory, original)
        path = directory / (self.cases['cases'][0]['case_id'] + '.json')
        row = a.read(path); row['execution'] = 'vLLM'; write(path, row)
        with self.assertRaisesRegex(ValueError, 'case JSON differs'):
            self.validate('bf16')

    def test_installed_maps_are_receipt_only_but_must_be_internally_bound(self):
        self.validate()  # /unavailable and /private intentionally do not exist.
        directory = self.dirs['fp32']; worker = a.read(directory / 'worker-receipt.json')
        worker['loaded_implementations']['torch']['sha256'] = 'd'*64
        self.rebind(directory, worker)
        with self.assertRaisesRegex(ValueError, 'newly recorded inventory'):
            self.validate()

    def test_missing_path_map_and_cross_run_references_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no explicit path map'):
            a.validate_run(self.dirs['fp32'] / 'completion.json', 'fp32', a.PathMap([]), self.cases, self.sources)
        completion = a.read(self.dirs['fp32'] / 'completion.json')
        completion['worker'] = remote_ref(self.dirs['bf16'] / 'worker-receipt.json')
        write(self.dirs['fp32'] / 'completion.json', completion)
        with self.assertRaisesRegex(ValueError, 'cross-run'):
            self.validate()

    def test_hash_chain_corruption_rejected_even_with_valid_json(self):
        path = self.dirs['fp32'] / 'worker-receipt.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'artifact hash differs'):
            self.validate()

    def test_launch_dtype_environment_and_worker_session_are_checked(self):
        directory = self.dirs['fp32']
        original = a.read(directory / 'launch.json')
        changed = copy.deepcopy(original)
        changed['argv'][changed['argv'].index('--dtype') + 1] = 'bf16'
        write(directory / 'launch.json', changed)
        with self.assertRaisesRegex(ValueError, 'launch dtype'):
            self.validate()
        changed = copy.deepcopy(original); changed['environment']['HF_HUB_OFFLINE'] = '0'
        write(directory / 'launch.json', changed)
        with self.assertRaisesRegex(ValueError, 'offline model'):
            self.validate()
        write(directory / 'launch.json', original)
        write(directory / 'process.json', {'pid': 10001, 'session_id': 10002})
        with self.assertRaisesRegex(ValueError, 'worker session'):
            self.validate()

    def test_frozen_case_input_and_vendor_inode_receipts_are_checked(self):
        directory = self.dirs['fp32']; original = a.read(directory / 'worker-receipt.json')
        worker = copy.deepcopy(original); worker['case_inputs'].pop(next(iter(worker['case_inputs'])))
        self.rebind(directory, worker)
        with self.assertRaisesRegex(ValueError, 'case provenance'):
            self.validate()
        worker = copy.deepcopy(original); worker['private_maps_after']['private_vendor_files']['/private/libcuda.so.580.173.02']['inode'] += 1
        self.rebind(directory, worker)
        with self.assertRaisesRegex(ValueError, 'inode receipt'):
            self.validate()


class ScalarAndPathTests(unittest.TestCase):
    def test_bf16_signed_zero_subnormal_and_nonfinite_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.bin'
            path.write_bytes(struct.pack('<4H', 0, 0x8000, 1, 0x8001))
            item = {'path': '/raw/raw.bin', 'sha256': a.sha(path), 'shape': [4], 'dtype': 'bfloat16', 'byte_order': 'little', 'bytes': 8}
            mapper = a.PathMap(['/raw=' + tmp])
            values = a.tensor(item, [4], 'bfloat16', mapper, path)
            self.assertEqual([struct.unpack('<I', struct.pack('<f', x))[0] for x in values], [0, 0x80000000, 0x10000, 0x80010000])
            for word in (0x7f80, 0xff80, 0x7fc0):
                path.write_bytes(struct.pack('<4H', 0, 0x8000, 1, word)); item['sha256'] = a.sha(path)
                with self.assertRaisesRegex(ValueError, 'nonfinite'):
                    a.tensor(item, [4], 'bfloat16', mapper, path)

    def test_tie_rank_and_signed_zero_difference_are_descriptive(self):
        values = [-4.0]*a.VOCAB; values[3] = values[4] = 1; values[7] = .5
        score = a.score(values, {'reference_token_id': 4, 'observed_token_ids': [7]})
        self.assertEqual(score['argmax_lowest_id_on_tie'], 3)
        self.assertEqual(score['choices'][0]['equal_count'], 2)
        self.assertEqual(a.ranks(values)[4], 2)
        difference = a.differences([0.0, 1.0], [-0.0, 2.0])
        self.assertEqual(difference['different_value_count'], 1)
        self.assertEqual(difference['different_fp32_bit_count'], 2)
        self.assertEqual(difference['signed_zero_difference_count'], 1)

    def test_explicit_absolute_longest_map_no_implicit_local_fallback(self):
        mapper = a.PathMap(['/old=/local', '/old/raw=/specific'])
        self.assertEqual(mapper.resolve('/old/raw/f'), Path('/specific/f'))
        for value in ('relative=/local', '/old=relative', '/a/../b=/local', '/old'):
            with self.assertRaises(ValueError): a.PathMap([value])
        with self.assertRaises(ValueError): mapper.resolve('/older/f')
        with self.assertRaises(ValueError): a.PathMap(['/old=/a', '/old=/b'])

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'bad.json'
            for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                p.write_text(value)
                with self.assertRaises(ValueError): a.read(p)

    def test_frozen_local_capture_helper_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tool.py'; path.write_text('')
            with self.assertRaisesRegex(ValueError, 'frozen local input changed'):
                a.frozen_inputs(ROOT/'numerical-divergence-cases.json', ROOT/'common-prefix-reference-source.json', path)

    def test_lazy_library_set_differences_are_reported_shared_bytes_must_match(self):
        result = a.inventory_difference({'/a': '0'*64}, {'/a': '0'*64, '/b': '1'*64})
        self.assertEqual(result['bf16_only'], ['/b'])
        with self.assertRaisesRegex(ValueError, 'shared native library identity'):
            a.inventory_difference({'/a': '0'*64}, {'/a': '1'*64})


if __name__ == '__main__':
    unittest.main()
