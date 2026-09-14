import copy
import json
from pathlib import Path
import unittest

from check_attention_natural_screen import adjudicate


class NaturalScreenContract(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'results/20260913-flashinfer-natural-screen/metrics.json'
        self.report = json.loads(path.read_text())

    def test_historical_kl_failure_is_failure(self):
        result = adjudicate(self.report)
        self.assertTrue(result['conditions']['nll'])
        self.assertFalse(result['conditions']['kl_from_fp32'])
        self.assertFalse(result['relative_screen_passed'])

    def test_identity_pass_does_not_promote_quality(self):
        for row in self.report['passages']:
            row['candidate'] = copy.deepcopy(row['baseline'])
        self.report['aggregate']['candidate'] = copy.deepcopy(self.report['aggregate']['baseline'])
        self.report['relative_screen_passed'] = True
        result = adjudicate(self.report)
        self.assertTrue(result['relative_screen_passed'])
        self.assertFalse(result['general_quality_accepted'])
        self.assertFalse(result['serving_qualified'])

    def test_corrupt_or_overclaiming_evidence_rejected(self):
        mutations = [
            lambda r: r.update(relative_screen_passed=True),
            lambda r: r.update(general_quality_accepted=True),
            lambda r: r.update(serving_measured=True),
            lambda r: r.update(targets=255),
            lambda r: r['passages'][1].update(index=0),
            lambda r: r['passages'][0]['candidate'].update(nll=float('nan')),
            lambda r: r['passages'][0]['candidate'].update(kl_from_fp32=float('inf')),
            lambda r: r['passages'][0]['candidate'].update(argmax_fp32_matches=33),
            lambda r: r['aggregate']['baseline'].update(nll=0),
            lambda r: r['logit_hashes'].update(candidate='missing'),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                report = copy.deepcopy(self.report)
                mutation(report)
                with self.assertRaises(ValueError):
                    adjudicate(report)


if __name__ == '__main__':
    unittest.main()
