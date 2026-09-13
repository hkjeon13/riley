import json,pathlib,sqlite3,subprocess,sys,tempfile,unittest

SCRIPT=pathlib.Path(__file__).with_name('paired_decode_trace_analysis.py')

class OwnedProfilerTests(unittest.TestCase):
    def test_detached_session_agent_and_descendants_are_scoped(self):
        from paired_decode_serving_trace import owned_processes
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            for pid,parent,group,argv in [(10,9,10,[]),(11,10,10,[]),(20,1,20,['nsys','--start-agent','--session-name','profile-10']),(21,20,21,[]),(30,1,30,['nsys','--start-agent','--session-name','profile-99']),(31,30,31,[])]:
                p=root/str(pid);p.mkdir();fields=['S',str(parent),str(group)]+['0']*16+['123']
                (p/'stat').write_text(f'{pid} (process with space) '+' '.join(fields))
                (p/'statm').write_text('100 2')
                (p/'cmdline').write_bytes('\0'.join(argv).encode())
            self.assertEqual(set(owned_processes(10,root)),{10,11,20,21})

class PairTraceTests(unittest.TestCase):
    def run_fixture(self,invalid=False):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'trace.sqlite';output=pathlib.Path(td)/'report.json'
            with sqlite3.connect(path) as db:
                db.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,correlationId INTEGER,globalTid INTEGER,nameId INTEGER,returnValue INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,globalPid INTEGER,contextId INTEGER,streamId INTEGER,demangledName INTEGER);')
                db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'cudaGraphLaunch_v10000'),(2,'riley_shared32_model::embedding()'),(3,'riley_future_token::resolve()')])
                for i in range(10):
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?,?)',(i*100000,i*100000+200,i,10,1,0))
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)',(i*100000+10000,i*100000+30000,i,1,1,1,3 if i in ([2,3,7] if invalid else [3,7]) else 2))
            result=subprocess.run([sys.executable,str(SCRIPT),str(path),str(output)],capture_output=True,text=True)
            return result,json.loads(output.read_text()) if output.exists() else None

    def test_pair_boundaries_and_exact_gap_accounting(self):
        result,report=self.run_fixture();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(report['selected']['launches'],8)
        self.assertEqual({k:v['gpu_gap']['count'] for k,v in report['gaps'].items()},{'ordinary':3,'within_pair':2,'after_pair':2})
        self.assertAlmostEqual(report['gaps']['within_pair']['gpu_gap']['sum_ms'],.16)
        self.assertAlmostEqual(report['gaps']['within_pair']['post_gpu_until_cpu_launch']['sum_ms'],.14)
        self.assertEqual(report['stage_graph_spans']['future_decode']['count'],2)

    def test_consecutive_successors_rejected(self):
        result,report=self.run_fixture(invalid=True)
        self.assertNotEqual(result.returncode,0);self.assertIsNone(report)
        self.assertIn('lacks immediate ordinary decode predecessor',result.stderr)

if __name__=='__main__':unittest.main()
