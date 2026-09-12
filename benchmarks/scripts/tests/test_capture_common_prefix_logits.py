from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2] / 'results/20260912-serving-optimization'
SPEC = importlib.util.spec_from_file_location('common_prefix_capture', ROOT / 'capture_common_prefix_logits.py')
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class PrefixTests(unittest.TestCase):
    def test_actual_frozen_sources_bind_exact_common_prefixes(self):
        document = capture.read(ROOT / 'numerical-divergence-cases.json')
        self.assertEqual(capture.sha(ROOT / 'numerical-divergence-cases.json'), capture.CASES_SHA)
        self.assertEqual(len(capture.case_inputs(document, [(Path('/tmp/riley-opt-260912'), ROOT / 'raw')])), 206)

    def test_a_branch_conditioned_prefix_cannot_be_called_common(self):
        document = capture.read(ROOT / 'numerical-divergence-cases.json')
        document['cases'][0]['input_token_ids'][-1] += 1
        with self.assertRaisesRegex(ValueError, 'common prefix differs'):
            capture.case_inputs(document, [(Path('/tmp/riley-opt-260912'), ROOT / 'raw')])

    def test_corrupted_raw_provenance_is_rejected(self):
        document = capture.read(ROOT / 'numerical-divergence-cases.json')
        document['cases'][0]['source_responses'][0]['raw_response']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            capture.case_inputs(document, [(Path('/tmp/riley-opt-260912'), ROOT / 'raw')])

    def test_rank_counts_preserve_ties_without_declaring_accuracy(self):
        values = [-4.0]*capture.VOCAB
        values[3] = values[4] = 1.0
        values[7] = 0.5
        result = capture.scores(values, {'reference_token_id': 4, 'observed_token_ids': [7]})
        self.assertEqual(result['argmax_lowest_id_on_tie'], 3)
        self.assertEqual(result['top1_top2_logit_gap'], 0)
        self.assertEqual(result['choices'][0]['equal_count'], 2)
        self.assertEqual(result['choices'][1]['strictly_greater_count'], 2)
        self.assertEqual(result['choices'][1]['gap_from_max'], 0.5)
        self.assertNotIn('passed', result)

    def test_full_finite_vocabulary_is_mandatory(self):
        case = {'reference_token_id': 0, 'observed_token_ids': [1]}
        for values in ([1.0]*10, [float('nan')]*capture.VOCAB, [float('inf')]*capture.VOCAB):
            with self.assertRaisesRegex(ValueError, 'full finite vocabulary'):
                capture.scores(values, case)

    def test_longest_explicit_path_map_wins(self):
        rules = [(Path('/old'), Path('/a')), (Path('/old/raw'), Path('/b'))]
        self.assertEqual(capture.mapped('/old/raw/file', rules), Path('/b/file'))
        self.assertEqual(capture.mapped('/older/file', rules), Path('/older/file'))


@dataclass
class Metadata:
    model: str = 'test double'


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'out'
        self.args = SimpleNamespace(root=str(self.root), output=str(self.output), dtype='fp32',
            reference_package=str(self.root / 'reference'), path_map=[], hf_home=str(self.root / 'hf'), timeout=1)
        self.gate = {'stopped': True}
        self.builder = SimpleNamespace(campaign_gate=lambda: self.gate)
        self.session = SimpleNamespace(process_runtime_environment=Mock(), verify_private_maps=lambda *_: {'compute_loaded': True})
        self.prereq = (self.builder, self.gate, self.session, {'child_only_overrides': {}},
                       {'cases': [{'input_token_ids': [1]}]}, {}, self.root / 'reference', {})

    def test_worker_writes_primary_and_cleanup_errors(self):
        self.output.mkdir()
        backend = SimpleNamespace(metadata=Metadata(), _device='test', _model=object(),
            _torch=SimpleNamespace(long='test', tensor=Mock(side_effect=RuntimeError('capture failed'))),
            close=Mock(side_effect=RuntimeError('close failed')))
        with patch.object(capture, 'prerequisites', return_value=self.prereq), \
             patch.object(capture, 'load_backend', return_value=backend), \
             patch.object(capture, 'gpu_identity', return_value={}), \
             patch.object(capture, 'dependencies', return_value={}), \
             patch.object(capture, 'loaded_implementations', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'close failed'):
                capture.worker(self.args)
        receipt = capture.read(self.output / 'worker-receipt.json')
        self.assertFalse(receipt['completed'])
        self.assertFalse(receipt['backend_closed'])
        self.assertEqual(receipt['failure']['message'], 'capture failed')
        self.assertEqual(receipt['cleanup_failure']['message'], 'close failed')

    def test_parent_keeps_finalization_when_cleanup_fails(self):
        capture.write(self.root / 'token-serving-round14-plan.json', {'base_environment': {}})
        process = Mock(pid=123, returncode=2)
        with patch.object(capture, 'prerequisites', return_value=self.prereq), \
             patch.object(capture.subprocess, 'Popen', return_value=process), \
             patch.object(capture.os, 'getsid', return_value=123), \
             patch.object(capture, 'stop_child', side_effect=RuntimeError('cleanup failed')):
            with self.assertRaisesRegex(ValueError, 'diagnostic incomplete'):
                capture.run(self.args)
        receipt = capture.read(self.output / 'finalization.json')
        self.assertEqual(receipt['failure']['message'], 'common-prefix worker failed')
        self.assertEqual(receipt['cleanup_failure']['message'], 'cleanup failed')
        self.assertFalse((self.output / 'completion.json').exists())

    def test_exited_leader_still_triggers_owned_group_cleanup(self):
        process = Mock(pid=123, returncode=0)
        with patch.object(capture.os, 'killpg') as kill, patch.object(Path, 'iterdir', return_value=[]):
            result = capture.stop_child(process)
        self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (123, signal.SIGKILL))
        self.assertTrue(result['cleanup_verified'])


if __name__ == '__main__':
    unittest.main()
