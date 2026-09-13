import pathlib
import tempfile
import unittest
from host_phase_analysis import parse


class HostPhaseTests(unittest.TestCase):
    def fixture(self, staged, split_submit=0):
        lines = []
        for kind, steps in [('decode', 8), ('prefill_or_mixed', 0), ('paired_decode', 2)]:
            lines.append(f'RILEY_HOST_PHASE kind={kind} steps={steps} scheduled_tokens={steps*32} plan_ns=1 execute_wall_ns=100 sample_ns=1 commit_publish_ns=1')
        counts = {'retain_encode': 10, 'sync_transfer': 8, 'buffered_submit': 2+split_submit,
                  'buffered_wait': 2+staged, 'read_validate': 10+staged, 'future_prepare': 0}
        for kind, calls in counts.items():
            lines.append(f'RILEY_RUNTIME_PHASE kind={kind} calls={calls} wall_ns=1')
        return '\n'.join(lines)

    def test_legacy_and_staged_pair_counts_preserve_wall_accounting(self):
        for staged in (0, 2):
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory)/'server.log'
                path.write_text(self.fixture(staged))
                report = parse(path)
                self.assertEqual(report['total_observed_ns'], 309)
                self.assertEqual(report['adapter_and_other_execute_ns'], 294)

    def test_split_submission_preserves_two_waits_per_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)/'server.log'
            path.write_text(self.fixture(2, 2))
            self.assertEqual(parse(path)['total_observed_ns'], 309)
            path.write_text(self.fixture(2, 1))
            with self.assertRaises(AssertionError):
                parse(path)

    def test_partial_staged_or_unmatched_wait_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)/'server.log'
            for data in (self.fixture(1), self.fixture(2).replace('kind=buffered_wait calls=4', 'kind=buffered_wait calls=3')):
                path.write_text(data)
                with self.assertRaises(AssertionError):
                    parse(path)


if __name__ == '__main__':
    unittest.main()
