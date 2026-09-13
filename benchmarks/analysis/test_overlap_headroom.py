import unittest
import sqlite3
import tempfile
from pathlib import Path
from overlap_headroom import gap_accounting, union_ns, analyze

class HeadroomAccountingTests(unittest.TestCase):
    def test_internal_stream_merge_retains_union_and_rejects_launch_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'trace.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript("CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,correlationId INTEGER,globalTid INTEGER,nameId INTEGER,returnValue INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,globalPid INTEGER,contextId INTEGER,streamId INTEGER);")
                db.execute("INSERT INTO StringIds VALUES (1,'cudaGraphLaunch')")
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?,?)',[(0,1,1,10,1,0),(150,151,2,10,1,0)])
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)',[(10,100,1,1,1,1),(50,120,1,1,1,2),(160,200,2,1,1,1),(180,220,2,1,1,2)])
            with self.assertRaises(ValueError):
                analyze(path)
            report, rows = analyze(path, merge_graph_streams=True)
            self.assertEqual(report['matched_device_launches'],2)
            self.assertEqual(rows[0]['streams'],[1,2])
            self.assertEqual(rows[0]['gpu_union_ns'],110)
            self.assertEqual(report['groups'][0]['inter_graph_gap']['sum_ms'],40/1e6)
            with sqlite3.connect(path) as db:
                db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=110 WHERE correlationId=2 AND streamId=1')
            with self.assertRaisesRegex(ValueError,'overlapping graph spans'):
                analyze(path, merge_graph_streams=True)

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
