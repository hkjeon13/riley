import unittest
from overlap_headroom import gap_accounting, union_ns

class HeadroomAccountingTests(unittest.TestCase):
    def test_union_does_not_double_count_overlapping_gpu_work(self):
        self.assertEqual(union_ns([(8,12),(0,10),(15,20),(2,4)]),17)
        self.assertEqual(union_ns([]),0)

    def test_inter_graph_gap_is_separate_from_inside_graph_idle(self):
        rows=[dict(gpu_start=0,gpu_end=100,gpu_union_ns=80,cpu_start=-10),
              dict(gpu_start=140,gpu_end=240,gpu_union_ns=90,cpu_start=130)]
        result=gap_accounting(rows)
        self.assertEqual(result['optimistic_trace_speedup_if_all_gaps_vanish'],1.2)
        self.assertAlmostEqual(result['post_gpu_until_next_cpu_launch']['sum_ms'],30/1e6)
        self.assertAlmostEqual(result['remaining_gap_after_cpu_launch_starts']['sum_ms'],10/1e6)
        self.assertAlmostEqual(result['inside_graph_unoccupied_ms'],30/1e6)

    def test_already_submitted_launch_does_not_create_negative_host_gap(self):
        rows=[dict(gpu_start=0,gpu_end=100,gpu_union_ns=100,cpu_start=-10),
              dict(gpu_start=140,gpu_end=240,gpu_union_ns=100,cpu_start=90)]
        result=gap_accounting(rows)
        self.assertEqual(result['post_gpu_until_next_cpu_launch']['sum_ms'],0)
        self.assertAlmostEqual(result['remaining_gap_after_cpu_launch_starts']['sum_ms'],40/1e6)

    def test_parallel_spans_and_incomplete_windows_are_rejected(self):
        row=dict(gpu_start=0,gpu_end=100,gpu_union_ns=100,cpu_start=-10)
        with self.assertRaises(ValueError):gap_accounting([row])
        with self.assertRaises(ValueError):gap_accounting([row,dict(row,gpu_start=90,gpu_end=200)])

if __name__=='__main__':unittest.main()
