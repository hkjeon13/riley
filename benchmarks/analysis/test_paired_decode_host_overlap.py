import unittest
from paired_decode_host_overlap import measure

class PairHostOverlapTests(unittest.TestCase):
    def fixture(self):
        return [dict(cpu_start=0, cpu_end=2, gpu_end=100, correlation=1, future=False, stage='ordinary_decode'),
                dict(cpu_start=5, gpu_end=200, correlation=2, future=True),
                dict(cpu_start=250, gpu_end=400, correlation=3, future=False)]

    def test_union_clips_overlap_and_excludes_wait_time(self):
        r = measure(self.fixture(), [(10, 102), (150, 203)], {1: [(10,100)], 2: [(101, 140), (130, 170)]})
        self.assertEqual(r['rows'][0]['between_waits_ns'], 48)
        self.assertEqual(r['rows'][0]['overlapped_device_activity_ns'], 48)

    def test_no_overlap_is_not_claimed(self):
        r = measure(self.fixture(), [(10, 202), (220, 230)], {1: [(10,100)], 2: [(101, 200)]})
        self.assertEqual(r['pairs_with_device_overlap'], 0)

    def test_missing_or_unfenced_wait_rejected(self):
        for waits in [[(10, 102)], [(10, 99), (150, 203)]]:
            with self.assertRaises(AssertionError):
                measure(self.fixture(), waits, {1: [(10,100)], 2: [(101, 200)]})

    def test_successor_preparation_overlaps_predecessor_gpu_activity(self):
        launches = self.fixture(); launches[1]['cpu_start'] = 70
        r = measure(launches, [(80,102),(150,203)], {1:[(20,100)],2:[(101,200)]})
        self.assertEqual(r['rows'][0]['between_launches_ns'],68)
        self.assertEqual(r['rows'][0]['preparation_device_overlap_ns'],50)
