import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('nrow_precise_probe',HERE/'probe.py')
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)


class SourceTests(unittest.TestCase):
    def test_both_original_arithmetic_bodies_are_preserved_exactly(self):
        raw=(HERE/'lineage/accepted_batch7_graph_numerics_precise.cu').read_bytes()
        generated,lineage=p.transform(raw);self.assertEqual(generated,(HERE/'candidate_preview.cu').read_bytes())
        for name,old,new in [('rope',p.ROPE_SIGNATURE,p.ROPE_NEW),('swiglu',p.SWIGLU_SIGNATURE,p.SWIGLU_NEW)]:
            source=raw.decode();start=source.index(old);original=source[start:source.index('\n}',start)+2]+'\n'
            changed=generated.decode();start=changed.index(new);end=changed.index('\n}',start)+2
            self.assertEqual(changed[start:end]+'\n',original.replace(old,new,1))
            self.assertEqual(lineage[name+'_kernel_sha256'],hashlib.sha256(original.encode()).hexdigest())
        self.assertTrue(lineage['arithmetic_body_preserved_by_reverse_transform'])
        self.assertNotIn('namespace riley_cuda_internal',generated.decode())
        for bad in (raw+b'\n',raw.replace(b'a*c-b*s',b'a*c+b*s')):
            with self.assertRaisesRegex(ValueError,'unknown accepted'):p.transform(bad)

    def test_common_support_exact_extraction_has_no_main_kernels_or_exports(self):
        attention=HERE.parent/'multisequence_attention_v1'
        support=p.fixture_support((attention/'native.cu').read_bytes())
        self.assertEqual(support,(HERE/'fixture_support_preview.hpp').read_bytes())
        self.assertEqual(p.sha(attention/'probe.py'),p.SUPPORT_SHA)
        self.assertIn(b'validate_descriptor(f,c);',support)
        self.assertIn(b'f.physical_owner[id]==r',support)
        self.assertNotIn(b'__global__',support);self.assertNotIn(b'int main(',support)
        self.assertNotIn(b'enqueue_',support);self.assertNotIn(b'RetainedGraph',support)
        with self.assertRaisesRegex(ValueError,'unfrozen attention'):
            p.fixture_support((attention/'native.cu').read_bytes()+b'\n')

    def test_inactive_header_only_branch_and_all_zero_destinations(self):
        for text in (p.ROPE_NEW,p.SWIGLU_NEW):
            branch=text.index('if(row>=active_rows)');ret=text.index('return;',branch)
            self.assertLess(text.index('packet[6]'),branch)
            self.assertLess(ret,text.index('const __nv_bfloat16*',branch))
            self.assertNotIn('metadata',text[:ret]);self.assertNotIn('packed[',text[:ret])
        self.assertNotIn('active_rows',p.WRAPPERS)
        self.assertNotIn('Memcpy',p.WRAPPERS);self.assertNotIn('Synchronize',p.WRAPPERS)
        self.assertIn('packed+row*960',p.ROPE_NEW);self.assertIn('packet+32+row*32',p.ROPE_NEW)
        self.assertIn('packed+row*3072',p.SWIGLU_NEW);self.assertIn('g+1536',p.SWIGLU_NEW)
        rope=[i for block in range(2) for thread in range(256) for i in range(thread+block*256,576,512)]
        self.assertEqual(sorted(rope),list(range(576)))
        self.assertEqual([block*256+thread for block in range(6) for thread in range(256)],list(range(1536)))

    def test_case_geometry_complete_and_row_kv_addresses_unique(self):
        cases=list(p.case_rows());self.assertEqual(len(cases),3456)
        self.assertEqual(len(p.case_index().splitlines()),3456)
        self.assertEqual({c['synthetic_seed'] for c in cases},{0,15,29})
        self.assertFalse(p.recipe()['checkpoint_activations'])
        self.assertEqual(p.recipe()['normal_allocations'],107136);self.assertEqual(p.recipe()['graph_allocations'],43)
        for bucket,active in p.MODES:
            selected=[c for c in cases if (c['bucket'],c['active_rows'])==(bucket,active)]
            self.assertEqual(len(selected),864)
            for row in range(active):self.assertEqual({c['positions'][row] for c in selected},set(range(128,160)))
        for mapping in range(3):
            physical=lambda r,b:(r*10+b) if mapping==0 else 63-(r*10+b) if mapping==1 else (7*(r*10+b)+3)%64
            self.assertEqual(len({physical(r,b) for r in range(4) for b in range(10)}),40)
            for t in range(32):
                addresses=[]
                for r in range(4):
                    pos=128+(t+11*r)%32
                    addresses += [((physical(r,pos//16)*3+h)*16+pos%16)*64+d for h in range(3) for d in range(64)]
                self.assertEqual(len(set(addresses)),768)
                self.assertTrue(all(0<=a<196608 for a in addresses))

    def test_native_cold_graph_and_full_compare_contract(self):
        source=(HERE/'native.cu').read_text();graph=source[source.index('size_t graph_transitions('):]
        self.assertEqual(source.count('cudaStreamBeginCapture('),1)
        self.assertLess(graph.index('graph.capture(workspace)'),graph.index('for(size_t i='))
        self.assertLess(graph.index('workspace.refresh(cases[i])'),graph.index('cudaGraphLaunch('))
        self.assertLess(graph.index('graph.close();'),graph.index('workspace.close();'))
        self.assertNotIn('SetParams',source);self.assertNotIn('cudaGraphExecUpdate',source)
        self.assertIn('require(count==2,',source);self.assertIn('cudaGraphNodeTypeKernel',source)
        self.assertIn('state.created==allocated',source)
        self.assertEqual([r['active_rows'] for r in p.transition_rows()],[3,4,3])
        self.assertEqual([r['positions'] for r in p.transition_rows()],[[128,139,150],[145,156,135,146],[133,144,155]])
        self.assertIn('actual[OKEYS]==cache_before[0]',source)
        self.assertIn('actual[OVALUES]==cache_before[1]',source)
        self.assertIn('kCache/2,2',source);self.assertIn('kCache/2,3',source)
        self.assertIn('if(!host.kv_written[i])',source)
        self.assertIn('first_word=(bo-kGuard)/2+i',source)

    def test_both_numerical_tus_receive_full_identical_precise_flags(self):
        flags=['-O3','-DNDEBUG','-std=c++17','-Xcompiler=-fno-exceptions','--generate-code=arch=compute_89,code=[compute_89,sm_89]']
        manifest={'source_root':'/accepted','generated':{'candidate.cu':{'path':'/out/candidate.cu'}},'host_source':{'path':'/host/native.cu'}}
        commands=p.commands(manifest,Path('/nvcc'),{'flags':flags},Path('/out/build'),Path('/runtime/libcudart.so'))
        self.assertEqual(commands[0][1:1+len(flags)],flags);self.assertEqual(commands[1][1:1+len(flags)],flags)
        self.assertIn('/accepted/'+p.SOURCE,commands[0]);self.assertIn('/out/candidate.cu',commands[1])
        self.assertIn('-I/out',commands[2]);self.assertIn('--cudart=shared',commands[3])
        self.assertNotIn('--use_fast_math',sum(commands,[]))


class RawTests(unittest.TestCase):
    def records(self):
        schema=p.SCHEMA+'-native.v1'
        device={'schema':schema,'kind':'device','uuid_hex':p.UUID,'compute_major':8,'compute_minor':9,'runtime_version':13000,'runtime_maps':[]}
        rows=[]
        for transition,expected_rows in ((False,p.case_rows()),(True,p.transition_rows())):
            for c in expected_rows:
                a,b=c['active_rows'],c['bucket']
                rows.append({'schema':schema,'kind':'graph_transition' if transition else 'case',**c,**{k:True for k in p.FLAGS},
                  'passed':True,'candidate_invocations':2,'oracle_invocations':2*a,'q_words_compared':a*576,'product_words_compared':a*1536,
                  'key_words_compared':196608,'value_words_compared':196608,'inactive_words_checked':(b-a)*2112,
                  'mapping_invariant_checked':not transition,'allocations_created':43 if transition else 11+8*a,
                  'mismatch_words':0,'mismatch_by_region':[0,0,0,0]})
        summary={'schema':schema,'kind':'summary','completed':True,'precise_bitwise_equal':True,**p.COUNTS,
                 'failed_cases':0,'failed_graph_transitions':0,'graph_transitions':3,'graphs_created':1,'graphs_destroyed':1,
                 'execs_created':1,'execs_destroyed':1,'allocations_created':107179,'allocations_freed':107179,
                 'live_bytes':0,'cleanup_errors':0,'stream_destroyed':True,'error':'','performance_claim':False,'runtime_maps':[]}
        return [device,*rows,summary]

    def validate(self,records):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'native.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in records))
            return p.validate_raw(path)

    def test_all_records_and_exact_signed_zero_mismatch_remains_failed(self):
        rows=self.records();self.assertTrue(self.validate(rows)[1]['precise_bitwise_equal'])
        rows[1].update(mismatch_words=1,mismatch_by_region=[0,1,0,0],exact_outputs=False,passed=False,
                       first_mismatch={'region':1,'word':1535,'oracle_bits':0,'candidate_bits':32768})
        rows[-1].update(failed_cases=1,precise_bitwise_equal=False)
        self.assertFalse(self.validate(rows)[1]['precise_bitwise_equal'])

    def test_partial_coverage_inconsistent_bits_and_case_matrix_fail(self):
        rows=self.records()
        for field,value in [('q_words_compared',575),('product_words_compared',1535),('key_words_compared',196607),
                            ('value_words_compared',196607),('candidate_invocations',1),('case_id',7),('synthetic_seed',1),
                            ('mismatch_words',1),('exact_outputs',False),('mapping_invariant_checked',False)]:
            original=rows[1][field];rows[1][field]=value
            with self.assertRaises(ValueError):self.validate(rows)
            rows[1][field]=original
        with self.assertRaises(ValueError):self.validate(rows[:-2]+rows[-1:])

    def test_retained_graph_and_cleanup_are_separately_required(self):
        rows=self.records();row=rows[-4]
        for field,value in [('active_rows',4),('graph_nodes',1),('capture_count',3),('replay_index',2),('parameter_updates',1),('allocations_created',44)]:
            original=row[field];row[field]=value
            with self.assertRaises(ValueError):self.validate(rows)
            row[field]=original
        for field,value in [('allocations_freed',107178),('cleanup_errors',1),('execs_destroyed',0),('stream_destroyed',False)]:
            original=rows[-1][field];rows[-1][field]=value
            with self.assertRaises(ValueError):self.validate(rows)
            rows[-1][field]=original
        row.update(mismatch_words=1,mismatch_by_region=[0,0,1,0],exact_outputs=False,passed=False,
                   first_mismatch={'region':2,'word':196607,'oracle_bits':0,'candidate_bits':32768})
        rows[-1].update(failed_graph_transitions=1,precise_bitwise_equal=False)
        self.assertFalse(self.validate(rows)[1]['precise_bitwise_equal'])


class CleanupTests(unittest.TestCase):
    def failure(self,kill_effect=None,final_wait=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);directory=Path(temp.name)
        manifest_path=directory/'fixtures.json';manifest_path.write_text('{}')
        manifest={'generated':{'cases.tsv':{'path':'/cases'}},'support_driver':{'path':'/support'}}
        compiled={'runtime':{'path':'/runtime/libcudart.so','sha256':'a'*64},'runtime_link':'/runtime/libcudart.so','binary':{'path':'/binary'}}
        driver=directory/'driver';driver.mkdir();(driver/'libcuda.so.1').write_text('driver')
        process=Mock(pid=999,returncode=None);process.poll.side_effect=lambda:process.returncode
        waits=[subprocess.TimeoutExpired('/binary',1),final_wait]
        def wait(**_kwargs):
            value=waits.pop(0)
            if isinstance(value,BaseException):raise value
            process.returncode=0 if value is None else value;return process.returncode
        process.wait.side_effect=wait
        with patch.object(p,'validate_manifest',return_value=manifest),patch.object(p,'validate_compile',return_value=compiled), \
             patch.object(p,'load',return_value=Mock()),patch.object(p,'DRIVER_SHA',p.sha(driver/'libcuda.so.1')), \
             patch.dict(p.os.environ,{'LD_PRELOAD':''}),patch.object(p.subprocess,'Popen',return_value=process), \
             patch.object(p.os,'killpg',side_effect=kill_effect) as kill:
            with self.assertRaises(subprocess.TimeoutExpired):p.run_probe(manifest_path,driver,1)
        self.assertFalse((directory/'result.json').exists())
        return p.read(directory/'process-exit.json'),kill

    def test_signal_exit_race_preserves_primary_and_reaps(self):
        receipt,kill=self.failure(ProcessLookupError())
        self.assertEqual(receipt['failure']['type'],'TimeoutExpired');self.assertIsNone(receipt['cleanup_failure'])
        self.assertTrue(receipt['owned_process_reaped']);self.assertEqual(kill.call_args.args,(999,signal.SIGTERM))

    def test_cleanup_failure_is_retained_without_success_result(self):
        receipt,_=self.failure(PermissionError('blocked'))
        self.assertEqual(receipt['failure']['type'],'TimeoutExpired');self.assertEqual(receipt['cleanup_failure']['type'],'PermissionError')
        self.assertFalse(receipt['owned_process_reaped'])


if __name__=='__main__':unittest.main()
