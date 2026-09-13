import pathlib,tempfile,unittest
from mixed_batch_cost import parse
class CostTests(unittest.TestCase):
    def check(self,text):
        with tempfile.TemporaryDirectory()as d:
            p=pathlib.Path(d)/'log';p.write_text(text);return parse(p)
    def fixture(self):
        return 'RILEY_HOST_PHASE kind=decode steps=1\nRILEY_HOST_PHASE kind=prefill_or_mixed steps=2\nRILEY_MIXED_COST rows_upper=32 decode_rows=0 prefill_requests=2 context_upper=128 count=2 execute_wall_ns=3000000 min_ns=1000000 max_ns=2000000\nRILEY_MIXED_COST_OVERFLOW count=1\n'
    def test_mean_and_overflow_accounting(self):
        r=self.check(self.fixture());self.assertEqual(r['rows'][0]['mean_execute_wall_ms'],1.5);self.assertEqual(r['overflow_steps'],1)
    def test_lost_samples_rejected(self):
        with self.assertRaises(AssertionError):self.check(self.fixture().replace('OVERFLOW count=1','OVERFLOW count=0'))
    def test_invalid_aggregate_rejected(self):
        with self.assertRaises(AssertionError):self.check(self.fixture().replace('execute_wall_ns=3000000','execute_wall_ns=6000000'))
if __name__=='__main__':unittest.main()
