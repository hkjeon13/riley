import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('nrow_attention_probe', HERE/'probe.py')
p = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)


class SourceAndGeometryTests(unittest.TestCase):
    def test_exact_batch7_source_and_only_mapping_transform(self):
        raw=(HERE/'lineage/accepted_batch7_graph_numerics.cu').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), p.SOURCE_SHA)
        generated,lineage=p.transform(raw)
        self.assertEqual(generated,(HERE/'candidate_preview.cu').read_bytes())
        source=raw.decode(); start=source.index(p.OLD_SIGNATURE); end=source.index('\n}\n#endif\n',start)+3
        expected=source[start:end]
        added=generated.decode(); start=added.index(p.NEW_SIGNATURE); end=added.index('cudaError_t enqueue_rows', start)
        restored=added[start:end].replace(p.NEW_SIGNATURE+p.INJECTION,p.OLD_SIGNATURE,1)
        self.assertEqual(restored,expected)
        self.assertEqual(hashlib.sha256(expected.encode()).hexdigest(),lineage['accepted_kernel_sha256'])
        self.assertNotIn('fused_rope',added)
        self.assertEqual(added.count('namespace riley_multisequence_attention_probe {'),1)
        self.assertEqual(added.count('<<<dim3(9,2,bucket),64,0,s>>>'),1)
        self.assertIn('#error "This experiment requires accepted MODE6 arithmetic"',added)

    def test_source_mutation_or_fusion_append_is_rejected(self):
        raw=(HERE/'lineage/accepted_batch7_graph_numerics.cu').read_bytes()
        for changed in [raw+b'\n', raw.replace(b'den+=__shfl_xor_sync',b'den-=__shfl_xor_sync',1), raw+b'// fusion8']:
            with self.assertRaisesRegex(ValueError,'exact accepted Batch7'): p.transform(changed)

    def test_inactive_returns_before_metadata_q_kv_and_barriers(self):
        text=(HERE/'candidate_preview.cu').read_text()
        body=text[text.index(p.NEW_SIGNATURE):]
        inactive=body.index('if(row>=active_rows)'); early_return=body.index('return;',inactive)
        self.assertLess(early_return,body.index('q+=row*576'))
        self.assertLess(early_return,body.index('const uint32_t* metadata='))
        self.assertLess(early_return,body.index('__syncthreads'))
        self.assertIn('packet+32+row*32',body)
        self.assertNotIn('k[',body[:early_return]); self.assertNotIn('v[',body[:early_return])
        writes=[head*64+half*32+thread for head in range(9) for half in range(2) for thread in range(32)]
        self.assertEqual(sorted(writes),list(range(576)))
        active=[head*64+(half*4+warp*2+block)*8+2*t+j for head in range(9) for half in range(2) for warp in range(2) for block in range(2) for t in range(4) for j in range(2)]
        self.assertEqual(sorted(active),list(range(576)))

    def test_full_reserved_and_live_block_maps_are_disjoint(self):
        for mapping in range(3):
            full=[p.physical(row,logical,mapping) for row in range(4) for logical in range(10)]
            self.assertEqual(len(set(full)),40); self.assertTrue(all(0<=v<64 for v in full))
            for seed in range(32):
                live=[]
                for row in range(4):
                    position=128+(seed+11*row)%32
                    ids=[p.physical(row,i,mapping) for i in range(position//16+1)]
                    valid=[16]*(len(ids)-1)+[position%16+1]
                    self.assertEqual(sum(valid),position+1)
                    live+=ids
                self.assertEqual(len(live),len(set(live)))

    def test_complete_case_matrix_and_all_tail_positions_per_row(self):
        cases=list(p.case_rows()); self.assertEqual(len(cases),34560)
        self.assertEqual(len(p.case_index().splitlines()),34560)
        self.assertEqual(p.recipe()['arithmetic_allocation_count'],432000)
        self.assertEqual(p.recipe()['graph_transition_allocation_count'],17)
        self.assertEqual(p.recipe()['allocation_count'],432017)
        for mode in p.MODES:
            selected=[c for c in cases if (c['bucket'],c['active_rows'])==mode]
            self.assertEqual(len(selected),8640)
            for row in range(mode[1]): self.assertEqual({c['positions'][row] for c in selected},set(range(128,160)))
        for c in cases:
            self.assertEqual(c['numerical_only_position159'],159 in c['positions'])
            self.assertEqual(len(c['positions']),len(set(c['positions'])))

    def test_helpers_and_synthetic_sample_have_exact_accepted_lineage(self):
        accepted=(HERE/'lineage/accepted_batch7_graph_numerics.cu').read_text(); generated=(HERE/'candidate_preview.cu').read_text()
        helpers=accepted[accepted.index('__device__ float exponential('):accepted.index('__global__ void attention(')]
        self.assertIn(helpers,generated)
        historical=HERE.parents[1]/'results/20260912-serving-optimization/batch7_attention_probe.cu'
        old=historical.read_text(); native=(HERE/'native.cu').read_text()
        sample=old[old.index('// Integer BF16 encodings only.'):old.index('struct Fixtures {')]
        self.assertIn(sample,native)
        self.assertIn('validate_descriptor(f,c);',native)
        self.assertLess(native.index('validate_descriptor(f,c);'),native.index('f.packet=f.canonical;'))
        self.assertIn('f.physical_owner[id]==r',native)
        self.assertIn('std::fill(f.packet.begin()+128+active*128,f.packet.begin()+640,0xff)',native)
        self.assertNotIn('cudaMemcpyDeviceToDevice',native)

    def test_numerical_translation_units_use_identical_complete_flags(self):
        flags=['-O3','-DNDEBUG','-std=c++17','-Xcompiler=-fno-exceptions','--use_fast_math','-gencode=arch=compute_89,code=sm_89']
        manifest={'source_root':'/accepted','generated_source':{'path':'/out/candidate.cu'},'host_source':{'path':'/probe/native.cu'}}
        commands=p.compile_commands(manifest,Path('/nvcc'),{'flags':flags},Path('/out/build'),Path('/runtime/libcudart.so'))
        self.assertEqual(commands[0][1:1+len(flags)],flags);self.assertEqual(commands[1][1:1+len(flags)],flags)
        self.assertIn('/accepted/kernels/src/graph_numerics.cu',commands[0])
        self.assertIn('/out/candidate.cu',commands[1])
        self.assertNotIn('--use_fast_math',commands[2]);self.assertIn('--cudart=shared',commands[3])
        self.assertNotIn('--cudart=static',commands[3])

    def test_fresh_header_count_and_one_retained_bucket4_graph(self):
        candidate=(HERE/'candidate_preview.cu').read_text(); native=(HERE/'native.cu').read_text()
        self.assertNotIn('active_rows',p.NEW_SIGNATURE)
        self.assertNotIn('active_rows',p.WRAPPER)
        self.assertIn('const uint32_t active_rows=packet[6];',p.INJECTION)
        self.assertLess(p.INJECTION.index('packet[6]'),p.INJECTION.index('if(row>=active_rows)'))
        self.assertNotIn('cudaMemcpy',p.WRAPPER);self.assertNotIn('Synchronize',p.WRAPPER)
        self.assertIn('<<<dim3(9,2,bucket),64,0,s>>>',candidate)
        transitions=list(p.transition_rows())
        self.assertEqual([r['active_rows'] for r in transitions],[3,4,3])
        self.assertEqual([r['positions'] for r in transitions],[[128,139,150],[145,156,135,146],[133,144,155]])
        self.assertEqual({r['bucket'] for r in transitions},{4})
        self.assertEqual(native.count('cudaStreamBeginCapture('),1)
        self.assertEqual(native.count('retained.capture('),1)
        graph=native[native.index('size_t graph_transitions('):]
        self.assertLess(graph.index('retained.capture('),graph.index('for(size_t step='))
        self.assertLess(graph.index('packet.upload(f.packet)'),graph.index('cudaGraphLaunch('))
        self.assertLess(graph.index('retained.close();'),graph.index('q.close();'))
        self.assertIn('require(nodes==1,',native)
        self.assertIn('cudaGraphNodeTypeKernel',native)
        self.assertNotIn('SetParams',native);self.assertNotIn('cudaGraphExecUpdate',native)


class RawRecordTests(unittest.TestCase):
    def records(self):
        device={'schema':p.SCHEMA+'-native.v1','kind':'device','uuid_hex':p.UUID,'compute_major':8,'compute_minor':9,'runtime_version':13000,'runtime_maps':[]}
        rows=[]
        for c in p.case_rows():
            rows.append({'schema':p.SCHEMA+'-native.v1','kind':'case',**c,**{k:True for k in p.FLAGS},'passed':True,
                'candidate_invocations':1,'oracle_invocations':c['active_rows'],'active_words_compared':c['active_rows']*576,
                'inactive_words_checked':(c['bucket']-c['active_rows'])*576,'mismatch_words':0})
        transitions=[]
        for c in p.transition_rows():
            transitions.append({'schema':p.SCHEMA+'-native.v1','kind':'graph_transition',**c,
                **{k:True for k in p.TRANSITION_FLAGS},'passed':True,'oracle_invocations':c['active_rows'],
                'active_words_compared':c['active_rows']*576,'inactive_words_checked':(4-c['active_rows'])*576,'mismatch_words':0})
        summary={'schema':p.SCHEMA+'-native.v1','kind':'summary','completed':True,'attention_bitwise_equal':True,**p.COUNTS,
                 'failed_cases':0,'allocations_created':432017,'allocations_freed':432017,'live_bytes':0,'cleanup_errors':0,
                 'graph_transitions':3,'failed_graph_transitions':0,'graphs_created':1,'graphs_destroyed':1,'execs_created':1,'execs_destroyed':1,
                 'stream_destroyed':True,'error':'','performance_claim':False,'runtime_maps':[]}
        return [device,*rows,*transitions,summary]

    def validate(self,records):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in records))
            return p.validate_raw(path)

    def test_full_complete_records_and_mismatch_observation(self):
        records=self.records();self.assertTrue(self.validate(records)[1]['attention_bitwise_equal'])
        row=records[1];row.update(mismatch_words=1,exact_outputs=False,passed=False,
            first_mismatch={'row':0,'word':7,'oracle_bits':0,'candidate_bits':32768})
        records[-1].update(failed_cases=1,attention_bitwise_equal=False)
        self.assertFalse(self.validate(records)[1]['attention_bitwise_equal'])

    def test_partial_outputs_wrong_matrix_and_cleanup_cannot_pass(self):
        records=self.records()
        for field,value in [('active_words_compared',575),('candidate_invocations',2),('inactive_words_checked',1),('case_id',3)]:
            saved=records[1][field];records[1][field]=value
            with self.assertRaises(ValueError):self.validate(records)
            records[1][field]=saved

    def test_retained_graph_contract_and_mismatch_are_separately_checked(self):
        records=self.records();row=records[-4]
        for field,value in [('active_rows',4),('replay_index',2),('graph_nodes',2),('capture_count',3),('parameter_updates',1),('active_words_compared',575)]:
            saved=row[field];row[field]=value
            with self.assertRaises(ValueError):self.validate(records)
            row[field]=saved
        records[-1]['execs_destroyed']=0
        with self.assertRaises(ValueError):self.validate(records)
        records[-1]['execs_destroyed']=1
        row.update(mismatch_words=1,exact_outputs=False,passed=False,
            first_mismatch={'row':2,'word':575,'oracle_bits':0,'candidate_bits':32768})
        records[-1].update(failed_graph_transitions=1,attention_bitwise_equal=False)
        self.assertFalse(self.validate(records)[1]['attention_bitwise_equal'])
        records[-1]['allocations_freed']=431999
        with self.assertRaisesRegex(ValueError,'cleanup incomplete'):self.validate(records)


class CleanupTests(unittest.TestCase):
    def run_failure(self,wait_effect,kill_effect=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);directory=Path(temp.name)
        manifest_path=directory/'fixtures.json';manifest_path.write_text('{}')
        manifest={'case_index':{'path':'/cases'}}
        compiled={'runtime':{'path':'/runtime/libcudart.so','sha256':'a'*64},'runtime_link':'/runtime/libcudart.so','binary':{'path':'/binary'}}
        driver_dir=directory/'driver';driver_dir.mkdir();(driver_dir/'libcuda.so.1').write_text('driver')
        process=Mock(pid=999,returncode=None);process.poll.side_effect=lambda:process.returncode
        pending=list(wait_effect)
        def wait(**_kwargs):
            value=pending.pop(0)
            if isinstance(value,BaseException):raise value
            process.returncode=0 if value is None else value
            return process.returncode
        process.wait.side_effect=wait
        with patch.object(p,'validate_manifest',return_value=manifest),patch.object(p,'validate_compile',return_value=compiled), \
             patch.object(p,'DRIVER_SHA',p.sha(driver_dir/'libcuda.so.1')),patch.dict(p.os.environ,{'LD_PRELOAD':''}), \
             patch.object(p.subprocess,'Popen',return_value=process),patch.object(p.os,'killpg',side_effect=kill_effect) as kill:
            with self.assertRaises(subprocess.TimeoutExpired):p.run_probe(manifest_path,driver_dir,1)
        return p.read(directory/'process-exit.json'),kill,process

    def test_exited_signal_race_does_not_mask_primary_or_skip_receipt(self):
        timeout=subprocess.TimeoutExpired('/binary',1)
        receipt,kill,_=self.run_failure([timeout,None],ProcessLookupError())
        self.assertEqual(receipt['failure']['type'],'TimeoutExpired');self.assertIsNone(receipt['cleanup_failure'])
        self.assertTrue(receipt['owned_process_reaped'])
        self.assertEqual(kill.call_args.args,(999,signal.SIGTERM))

    def test_cleanup_error_preserves_primary_and_records_unreaped(self):
        timeout=subprocess.TimeoutExpired('/binary',1)
        receipt,_,_=self.run_failure([timeout],PermissionError('blocked'))
        self.assertEqual(receipt['failure']['type'],'TimeoutExpired')
        self.assertEqual(receipt['cleanup_failure']['type'],'PermissionError')
        self.assertFalse(receipt['owned_process_reaped'])


if __name__=='__main__':unittest.main()
