"""The diagnostic build may start only after every serving process is accounted for."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'results/20260912-serving-optimization/build_decode_profile_batch8.py'
SPEC = importlib.util.spec_from_file_location('batch8_diagnostic_builder', SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class CampaignGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'token-serving-round14'
        self.directory.mkdir()
        self.plan = self.root / 'token-serving-round14-plan.json'
        self.plan.write_text('{}\n')
        self.plan_ref = {'path': str(self.plan), 'sha256': builder.sha(self.plan)}
        self.write(self.directory / 'preparation.json', {'plan': self.plan_ref})
        restored = self.root / 'blender-round14/verified.json'
        self.write(restored, {'alive_and_listening': True,
            'commands_and_gui_environment_match': True,
            'all_relaunched_processes_have_pinned_vendor_maps': True,
            'host_or_global_configuration_modified': False, 'processes': [1, 2, 3]})
        self.write(self.directory / 'finalization.json', {
            'failure': 'a failed campaign can still be diagnosed',
            'restoration': {'path': str(restored), 'sha256': builder.sha(restored)}})
        self.lane = self.directory / 'pair01/baseline'
        self.write(self.lane / 'launch.json', {})
        self.write(self.lane / 'process.json', {'pid': 123})
        self.write(self.lane / 'process-exit.json', {'cleanup': {
            'cleanup_verified': True, 'remaining_owned_pids': []}})
        self.addCleanup(patch.stopall)
        patch.object(builder, 'ROOT', self.root).start()
        patch.object(builder, 'CAMPAIGN_PLAN_SHA', self.plan_ref['sha256']).start()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_failed_campaign_with_all_processes_closed_can_be_diagnosed(self):
        result = builder.campaign_gate()
        self.assertFalse(result['performance_or_acceptance_inferred'])
        self.assertEqual(len(result['process_exit_receipts']), 1)

    def test_missing_exit_is_not_silently_omitted(self):
        for filename in ('launch.json', 'process.json'):
            with self.subTest(filename=filename):
                path = self.directory / 'pair02/candidate' / filename
                self.write(path, {})
                with self.assertRaisesRegex(AssertionError, 'each launched lane'):
                    builder.campaign_gate()
                path.unlink()

    def test_preparation_must_bind_exact_plan(self):
        for key in ('path', 'sha256'):
            with self.subTest(key=key):
                reference = dict(self.plan_ref, **{key: 'wrong'})
                self.write(self.directory / 'preparation.json', {'plan': reference})
                with self.assertRaises(AssertionError):
                    builder.campaign_gate()

    def test_living_process_or_failed_cleanup_blocks_build(self):
        for cleanup in ({'cleanup_verified': True, 'remaining_owned_pids': [123]},
                        {'cleanup_verified': False, 'remaining_owned_pids': []}):
            with self.subTest(cleanup=cleanup):
                self.write(self.lane / 'process-exit.json', {'cleanup': cleanup})
                with self.assertRaises(AssertionError):
                    builder.campaign_gate()

    def test_changed_cleanup_receipt_changes_gate_identity(self):
        before = builder.campaign_gate()
        path = self.lane / 'process-exit.json'
        data = json.loads(path.read_text())
        data['failure'] = 'new information'
        self.write(path, data)
        self.assertNotEqual(before, builder.campaign_gate())


if __name__ == '__main__':
    unittest.main()
