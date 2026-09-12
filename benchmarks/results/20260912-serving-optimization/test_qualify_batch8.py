"""CPU-only adapter regressions; fixture success never represents GPU execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import qualify_batch8 as q


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / 'reference'
        self.base.mkdir()
        self.build = {'source_root': str(self.root / 'batch8-source'), 'source_commit': 'new',
                      'source_files': {'kernels/src/graph_numerics.cu': 'native'}, 'binaries': {'server': 'binary'}}
        self.binding = {'environment': {'gpu': {'uuid': 'GPU-12345678-1234-1234-1234-123456789abc',
                                               'model': 'RTX 4090', 'pci_bus_id': '00000000:01:00.0'}}}
        self.refs = {'binding': {'path': 'ref', 'sha256': 'ref'}, 'model_files': {}}
        self.runtime = {'library_directory': str(self.root / 'driver'), 'receipt': {'path': 'r', 'sha256': 'r'}}
        self.helpers = {'runner': {'path': 'runner', 'sha256': 'runner'}}
        self.write('batch8-build.json', self.build)
        gpu = {'uuid': self.binding['environment']['gpu']['uuid'], 'name': 'RTX 4090',
               'driver_version': '580.173.02', 'pci_bus_id': '00000000:01:00.0', 'device_index': 0}
        results = []
        for name, argv, is_gpu in q.commands():
            text = ('test llama::' + name + ' ... ok\n' if is_gpu else '')
            if name in q.frozen.PARITY_MARKERS:
                text += q.frozen.PARITY_MARKERS[name] + '\n'
            text += 'test result: ok. 1 passed; 0 failed; 0 ignored;\n'
            path = self.root / ('batch8-' + name + '.log')
            path.write_text(text)
            results.append({'name': name, 'argv': argv, 'passed': True, **q.evidence(path),
                            'passed_tests': 1, 'failed_tests': 0, 'ignored_tests': 0})
        self.receipt = {'schema_version': 'riley.batch8-model-correctness.v1', 'passed': True, **self.build,
                        'source_clean': True, 'build': q.evidence(self.root / 'batch8-build.json'),
                        'references': self.refs, 'helpers': self.helpers, 'runtime': self.runtime,
                        'runtime_environment': q.runtime_environment(self.root, Path(self.runtime['library_directory'])),
                        'gpu_before': gpu, 'gpu_after': gpu, 'checks': results[:4], 'server_lib_tests': results[4],
                        'profile_unit_tests': results[5], 'gpu_tests_executed': True,
                        'vllm_reference_tokens_exact': True, 'full_logits_and_kv_exact': True,
                        'http_correctness_qualified': False, 'performance_claim_eligible': False, 'performance_measured': False}
        self.write('batch8-model-tests.json', self.receipt)
        self.reference_patch = patch.object(q, 'reference', return_value=(self.binding, self.refs))
        self.runtime_patch = patch.object(q, 'verify_runtime', return_value=self.runtime)
        self.helper_patch = patch.object(q, 'helper_evidence', return_value=self.helpers)
        for p in (self.reference_patch, self.runtime_patch, self.helper_patch):
            p.start(); self.addCleanup(p.stop)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_fresh_six_check_contract_and_counts(self):
        self.assertEqual(len(q.commands()), 6)
        self.assertEqual(q.validate_model(self.root, self.base, self.build), self.receipt)
        self.assertIn('prompts=3', q.frozen.PARITY_MARKERS[q.GPU_TESTS[1]])
        self.assertIn('full_initialized_kv_snapshots=96', q.frozen.PARITY_MARKERS[q.GPU_TESTS[1]])
        self.assertIn('invalid_shape_stage_cases=42', q.frozen.PARITY_MARKERS[q.GPU_TESTS[0]])

    def test_rejects_stale_build_missing_legacy_bad_gpu_and_scope(self):
        mutations = [lambda r: r.update(source_commit='old'), lambda r: r['checks'].pop(),
                     lambda r: r['gpu_after'].update(uuid='GPU-wrong'), lambda r: r.update(performance_measured=True),
                     lambda r: r['profile_unit_tests'].update(passed_tests=0),
                     lambda r: r.update(http_correctness_qualified=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.receipt); mutation(bad); self.write('batch8-model-tests.json', bad)
                with self.assertRaises((ValueError, KeyError)):
                    q.validate_model(self.root, self.base, self.build)

    def test_rehashed_log_missing_parity_cannot_pass(self):
        item = self.receipt['checks'][1]
        path = Path(item['path'])
        path.write_text(path.read_text().replace('full_initialized_kv_snapshots=96', 'full_initialized_kv_snapshots=95'))
        item.update(q.evidence(path)); self.write('batch8-model-tests.json', self.receipt)
        with self.assertRaisesRegex(ValueError, 'exact parity evidence'):
            q.validate_model(self.root, self.base, self.build)

    def fusion(self):
        driver = self.root / 'batch8_fusion_probe.py'; driver.write_text('# fixture validator\n')
        value = {'schema_version': 'riley.batch8-fusion-correctness.v1', 'passed': True,
                 'gpu_tests_executed': True, **self.build, 'source_build': q.evidence(self.root / 'batch8-build.json'),
                 'candidate_entry': q.ENTRY, 'candidate_source_sha256': 'native', 'runner': q.evidence(driver),
                 'cases': 8640, 'layers': 30, 'positions': 32, 'patterns': 3, 'mappings': 3,
                 **dict.fromkeys(q.FLAGS, True), 'performance_measured': False, 'performance_claim_eligible': False,
                 'device': {'uuid_hex': '12345678123412341234123456789abc', 'runtime_version': 13000}}
        module = types.SimpleNamespace(__file__=str(driver), validate_receipt=lambda path, build: value)
        return value, module

    def test_fusion_hook_current_source_complete_scope(self):
        value, module = self.fusion()
        with patch.dict(sys.modules, {'batch8_fusion_probe': module}):
            self.assertEqual(q.validate_fusion(self.root, self.build, self.binding), value)
            for field, invalid in [('candidate_entry', 'old'), ('candidate_source_sha256', 'old'),
                                   ('cases', 8639), ('inactive_kv_unchanged', False), ('performance_measured', True)]:
                original = value[field]; value[field] = invalid
                with self.subTest(field=field), self.assertRaises(ValueError):
                    q.validate_fusion(self.root, self.build, self.binding)
                value[field] = original
            value['device']['uuid_hex'] = '00' * 16
            with self.assertRaisesRegex(ValueError, 'device/runtime'):
                q.validate_fusion(self.root, self.build, self.binding)

    def test_missing_fusion_emits_no_qualification(self):
        with patch.object(q, 'validate_build', return_value=self.build), patch.object(q, 'validate_fusion', side_effect=ValueError('missing probe')):
            with self.assertRaisesRegex(ValueError, 'missing probe'):
                q.qualify(self.root, self.base)
        self.assertFalse((self.root / 'batch8-qualification.json').exists())

    def test_exact_diff_rejects_inherited_http_change(self):
        correct = '\n'.join('M\t' + name for name in sorted(q.CHANGED))
        with patch.object(q, 'git', return_value=correct):
            q.exact_diff(self.root, q.PARENT, q.CHANGED)
        for bad in (correct + '\nM\tcrates/riley-server/src/openai.rs', correct.replace('M\t', 'A\t', 1)):
            with patch.object(q, 'git', return_value=bad), self.assertRaises(ValueError):
                q.exact_diff(self.root, q.PARENT, q.CHANGED)

    def test_failed_run_preserves_log_and_emits_no_receipt(self):
        for path in self.root.glob('batch8-*.log'): path.unlink()
        (self.root / 'batch8-model-tests.json').unlink()
        with patch.object(q, 'validate_build', return_value=self.build), patch.object(q, 'gpu_identity', return_value={}), \
             patch.object(q.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['cargo'])):
            with self.assertRaises(subprocess.CalledProcessError):
                q.run_checks(self.root, self.base, Path(self.runtime['library_directory']))
        self.assertTrue((self.root / ('batch8-' + q.GPU_TESTS[0] + '.log')).exists())
        self.assertFalse((self.root / 'batch8-model-tests.json').exists())


if __name__ == '__main__':
    unittest.main()
