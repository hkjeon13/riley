import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('multisequence_probe_under_test', HERE/'multisequence_projection_candidates_v1.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def metadata(shape, m, arm=-1):
    _, n, k, custom = p.SHAPES[shape]
    candidate=m!=1
    return {'struct_size':112,'backend':1,'algorithm_id':31 if candidate else 13,'tile_id':3 if candidate else 0,
            'stages_id':2 if candidate else 0,'split_k':0 if candidate else 1,
            'reduction_scheme':0,'cta_swizzling':1 if candidate else 0,'custom_option':2 if candidate else custom,
            'deterministic':1,'workspace_bytes':4096 if candidate and arm==1 else 0,
            'numerical_implementation_flags':65537 if candidate else 131585,'compute_capability_major':8,
            'compute_capability_minor':9,'runtime_version':13000,'cublaslt_version':130101,'m':m,'n':n,'k':k,'reserved':[0,0]}


def sample_records(rejected=False, mismatch=False):
    def r(kind, **kw): return {'schema_version':p.SCHEMA+'-native.v1','kind':kind,**kw}
    records=[r('device',uuid=p.UUID,runtime_version=13000,compute_capability_major=8,
               compute_capability_minor=9,abi=1,native_build_info=p.NATIVE_BUILD_INFO,maps='mock')]
    for shape in range(5):
        for slot in range(5):
            arm=-1 if slot==0 else (slot-1)//2
            m=1 if slot==0 else (2,4)[(slot-1)%2]
            reject=rejected and shape==0 and m==2 and arm==0
            row=r('plan',shape=shape,projection=p.SHAPES[shape][0],m=m,arm=arm,flags=0,
                  selection='strict_plan_create',m1_oracle=m==1,workspace_cap=16*1024*1024 if m==1 else p.WORKSPACE_CAPS[arm],
                  status=11 if reject else 0,admitted=not reject,metadata=None if reject else metadata(shape,m,arm))
            if reject:row.update(error='unsupported',error_stage=10,error_domain=1,error_native_code=0)
            else:row['metadata_valid']=True
            records.append(row)
    cases=[]
    for group in range(121):
        shape,layer=(group%4,group//4) if group<120 else (4,30)
        _,n,k,_=p.SHAPES[shape]
        for pattern in range(3):
            for m,active in p.MODES:
                for arm,cap in enumerate(p.WORKSPACE_CAPS):
                    c={'case_id':len(cases),'layer':layer,'shape':shape,'m':m,'active_rows':active,'n':n,'k':k,
                       'pattern':pattern,'arm':arm,'workspace_cap':cap}
                    cases.append(c)
                    if rejected and shape==0 and m==2 and arm==0:
                        records.append(r('case',**c,executed=False,reason='strict_descriptor_not_supported'));continue
                    rows=[{'row':row,'words':n,'mismatches':0,'first_mismatch':None,'oracle_bits':None,'batched_bits':None} for row in range(m)]
                    count=int(mismatch and c['case_id']==0)
                    if count:rows[0].update(mismatches=1,first_mismatch=0,oracle_bits=0,batched_bits=32768)
                    workspace=metadata(shape,m,arm)['workspace_bytes']
                    records.append(r('case',**c,executed=True,words=m*n,mismatches=count,rows=rows,
                        workspace_bytes=workspace,oracle_workspace_bytes=0,workspace_parent_bytes=workspace+512,
                        oracle_workspace_parent_bytes=512,workspace_span_dtype='U8',workspace_span_offset=256,
                        workspace_passed=True,case_device_allocations=5+2*m,
                        **{key:count==0 if key=='exact_outputs' else True for key in p.CASE_FLAGS}))
    records += [r('runtime_end',maps='mock'),r('summary',completed=True,cases=2178,all_plans_admitted=not rejected,
               all_outputs_exact=not(rejected or mismatch),all_resources_closed=True,performance_measured=False,
               performance_claim_eligible=False,failure='')]
    return records,{'case_records':cases}


class ProbeTests(unittest.TestCase):
    def test_frozen_v2_preserved_and_candidate_uses_public_strict_api(self):
        self.assertEqual(p.sha(HERE/'multisequence_projection_probe_v2.py'),
                         'be721d7beb08b66113553d3d97fc8bf4592ef35c533017473d9e453d5d973c5f')
        self.assertEqual(p.sha(HERE/'multisequence_projection_probe_v2.cpp'),
                         '962e3e8ff6cc58249254c75e62599b65e44521ebd4932cb546197738551ae3e9')
        native=(HERE/'multisequence_projection_candidates_v1.cpp').read_text()
        self.assertNotIn('riley_cuda_gemm_plan_create_anchored(',native)
        self.assertEqual(native.count('auto status=riley_cuda_gemm_plan_create('),1)
        self.assertIn('c.flags=0',native)
        self.assertIn('scratch=span(workspace,RILEY_CUDA_DTYPE_U8)',native)
        self.assertIn('plan,&a,&b,&c,&scratch,stream',native)
        self.assertIn('oracle_workspace=s.allocate(Bytes())',native)
        self.assertIn('s.download(workspace,workspace_guards)',native)
        self.assertIn('s.download(oracle_workspace,workspace_guards)',native)

    def test_prepare_then_validate_preserves_count_and_names(self):
        # Reproduce V1's real prepare/validate failure before checking V2.
        # Only remote proofs and large checkpoint materialization are doubled;
        # the manifest assembly, disk roundtrip, evidence and validators are real.
        old_spec=importlib.util.spec_from_file_location('projection_frozen_v1',HERE/'multisequence_projection_probe.py')
        old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();base=root/'base';base.mkdir()
            (root/'batch8-build.json').write_text('{}')
            model=root/'model.safetensors';model.write_bytes(b'small mocked checkpoint')
            config=model.with_name('config.json')
            config.write_text(json.dumps({'num_hidden_layers':30,'hidden_size':576,'intermediate_size':1536,
                                         'vocab_size':49152,'tie_word_embeddings':True}))
            build={'source_root':str(root/'batch8-source'),'source_commit':p.COMMIT,'binaries':{},'source_files':{}}
            refs={'model_files':{str(model):p.sha(model)}};proof={'mock_remote_proof':True}
            def materialize(_model,_config,output,create):
                index=output/'cases.tsv'
                if create:index.write_text('mocked fixture materialization\n')
                self.assertEqual(index.read_text(),'mocked fixture materialization\n')
                return {'fixture_files':{},'weight_records':[],'case_records':[],
                        'case_index':p.evidence(index),'safetensors_data_start':0}
            for module,label in ((old,'v1'),(p,'v2')):
                with patch.object(module,'prerequisites',return_value=(build,proof,refs)), \
                     patch.object(module,'fixture_description',side_effect=materialize) as fixtures:
                    returned=module.prepare(root,base,root/label)
                    path=Path(returned['manifest']['path'])
                    if module is old:
                        with self.assertRaisesRegex(ValueError,'fixture schema/count differs'):module.validate_manifest(path)
                        self.assertEqual(fixtures.call_count,1)
                        continue
                    manifest=module.validate_manifest(path)
                    self.assertEqual([call.args[3] for call in fixtures.call_args_list],[True,False])
                    self.assertEqual(manifest['schema_version'],p.SCHEMA+'-fixtures.v1')
                    self.assertEqual(manifest['pattern_count'],3);self.assertEqual(returned['pattern_count'],3)
                    self.assertEqual(manifest['patterns'],list(p.PATTERNS))
                    self.assertEqual(manifest['arm_count'],2);self.assertEqual(manifest['arms'],list(p.ARMS))
                    self.assertTrue(all(manifest[k]==v for k,v in p.COUNTS.items()))
                    self.assertEqual(manifest['native_source'],p.evidence(HERE/'multisequence_projection_candidates_v1.cpp'))
                    original=path.read_text()
                    for key,value in [('pattern_count',2),('pattern_count',list(p.PATTERNS)),
                                      ('patterns',3),('patterns',list(reversed(p.PATTERNS))),
                                      ('arms',[p.ARMS[0]]),('arm_count',1),
                                      ('schema_version','riley.multisequence-projection-fixtures.v2')]:
                        changed=json.loads(original);changed[key]=value;path.write_text(json.dumps(changed))
                        with self.subTest(key=key,value=value),self.assertRaises(ValueError):module.validate_manifest(path)
                    path.write_text(original)

    def test_case_plan_coverage(self):
        records,m=sample_records();r=p.validate_records(records,m)
        self.assertEqual(r['executed_cases'],2178);self.assertTrue(r['arithmetic_equal'])
        self.assertEqual(len({(c['layer'],c['shape']) for c in m['case_records']}),121)

    def test_native_build_identity_rejected(self):
        records,m=sample_records();records[0]['native_build_info']='riley-cuda-native abi=1 nvcc=13.0.0'
        with self.assertRaisesRegex(ValueError,'native device/build identity'):p.validate_records(records,m)

    def test_unsupported_is_observation_without_fallback(self):
        records,m=sample_records(rejected=True);r=p.validate_records(records,m)
        self.assertFalse(r['arithmetic_equal']);self.assertEqual(r['skipped_cases'],90)
        records[2]['status']=10
        with self.assertRaisesRegex(ValueError,'unexpected plan rejection'):p.validate_records(records,m)

    def test_bitwise_mismatch_signed_zero_preserved(self):
        records,m=sample_records(mismatch=True);r=p.validate_records(records,m)
        self.assertEqual(r['mismatched_words'],1);self.assertFalse(r['arithmetic_equal'])
        records[26]['rows'][0]['batched_bits']=0
        with self.assertRaisesRegex(ValueError,'mismatch evidence'):p.validate_records(records,m)

    def test_metadata_keeps_m1_exact_and_candidates_strict(self):
        for field in metadata(0,1):
            a=metadata(0,1);a[field]=[1,0] if field=='reserved' else a[field]+1
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_metadata(a,0,1)
        for arm,cap in enumerate(p.WORKSPACE_CAPS):
            p.validate_metadata(metadata(0,2,arm),0,2,cap)
            for field,value in [('workspace_bytes',cap+1),('split_k',2),('reduction_scheme',1),('deterministic',0),
                                ('runtime_version',12000),('reserved',[1,0]),('algorithm_id',-1)]:
                a=metadata(0,2,arm);a[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):p.validate_metadata(a,0,2,cap)

    def test_workspace_is_exact_for_each_arm_and_admission_is_not_equality(self):
        records,m=sample_records();result=p.validate_records(records,m)
        self.assertEqual([a['executed_cases'] for a in result['arm_results']],[1089,1089])
        self.assertEqual([a['admitted_plans'] for a in result['arm_results']],[10,10])
        self.assertEqual(records[26]['workspace_bytes'],0);self.assertEqual(records[27]['workspace_bytes'],4096)
        for index,field,value in [(2,'workspace_cap',16*1024*1024),(3,'arm',1),(4,'flags',1)]:
            changed=copy.deepcopy(records);changed[index][field]=value
            with self.subTest(index=index,field=field),self.assertRaises(ValueError):p.validate_records(changed,m)
        for field,value in [('workspace_bytes',0),('workspace_parent_bytes',512),('workspace_span_dtype','BF16'),
                            ('workspace_span_offset',0),('workspace_passed',False),('oracle_workspace_parent_bytes',0),
                            ('case_device_allocations',7),('workspace_guards_intact',False)]:
            changed=copy.deepcopy(records);changed[27][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_records(changed,m)
        bad,m=sample_records(mismatch=True);result=p.validate_records(bad,m)
        self.assertFalse(result['arm_results'][0]['arithmetic_equal']);self.assertTrue(result['arm_results'][1]['arithmetic_equal'])
        unsupported,m=sample_records(rejected=True);result=p.validate_records(unsupported,m)
        self.assertEqual(result['arm_results'][0]['skipped_cases'],90)
        self.assertFalse(result['arm_results'][0]['arithmetic_equal']);self.assertTrue(result['arm_results'][1]['arithmetic_equal'])

    def test_case_missing_duplicate_inactive_row_rejected(self):
        original,m=sample_records()
        for mutation in ('missing','duplicate','last_row'):
            records=copy.deepcopy(original)
            if mutation=='missing':records.pop(26)
            elif mutation=='duplicate':records[27]=copy.deepcopy(records[26])
            else:records[30]['rows'][-1]['row']=0
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):p.validate_records(records,m)

    def test_guards_cleanup_accounting_fail_closed(self):
        original,m=sample_records()
        for key in p.CASE_FLAGS[2:]:
            records=copy.deepcopy(original);records[26][key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):p.validate_records(records,m)
        original[-1]['all_resources_closed']=False
        with self.assertRaises(ValueError):p.validate_records(original,m)

    def test_fixture_raw_bits_and_heterogeneous_padding(self):
        for k in (576,1536):
            for pattern in range(3):
                data=p.pattern_bytes(pattern,k,4,3)
                rows=[data[row*k*2:(row+1)*k*2] for row in range(4)]
                self.assertEqual(len(set(rows[:3])),3);self.assertEqual(rows[-1],bytes(k*2))
                self.assertEqual(p.pattern_bytes(pattern,k,2,2),b''.join(rows[:2]))
                self.assertEqual(p.pattern_bytes(pattern,k,4,4)[:k*2*3],b''.join(rows[:3]))
                words=[x[0] for x in struct.iter_unpack('<H',data)]
                self.assertTrue(all((x&0x7f80)!=0x7f80 for x in words))
                if pattern==1:
                    self.assertIn(0x8000,words);self.assertIn(1,words)

    def test_tied_and_untied_head_binding(self):
        config={'num_hidden_layers':30,'hidden_size':576,'intermediate_size':1536,'vocab_size':49152,'tie_word_embeddings':True}
        groups=list(p.group_specs(config));self.assertEqual(len(groups),121)
        self.assertEqual(groups[-1][2],[('model.embed_tokens.weight',49152)])
        config['tie_word_embeddings']=False
        self.assertEqual(list(p.group_specs(config))[-1][2],[('lm_head.weight',49152)])
        del config['tie_word_embeddings']
        with self.assertRaises(ValueError):list(p.group_specs(config))

    def test_tensor_slices_preserve_bits_and_reject_range_dtype(self):
        bits=struct.pack('<4H',0,0x8000,1,0xbf80)
        header={'test':{'dtype':'BF16','shape':[2,2],'data_offsets':[0,8]}}
        raw,part=p.tensor_slice(io.BytesIO(bits),header,0,8,'test',2,2)
        self.assertEqual(raw,bits);self.assertEqual(part['bytes'],8)
        for mutation in ('dtype','range','nonfinite'):
            h=copy.deepcopy(header);b=bits
            if mutation=='dtype':h['test']['dtype']='F16'
            elif mutation=='range':h['test']['data_offsets']=[1,9]
            else:b=struct.pack('<4H',0,0,0,0x7f80)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):p.tensor_slice(io.BytesIO(b),h,0,8,'test',2,2)

    def test_json_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):json.loads('{"passed":true,"passed":false}',object_pairs_hook=p.unique)

    def test_run_owned_process_success_and_failure_cleanup(self):
        # Control-flow doubles only. No CUDA process, remote call or compiler.
        for mode in ('success','native_failure','timeout_term','timeout_kill'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); manifest_path=root/'fixtures.json';manifest_path.write_text('{}')
                records,m=sample_records();m.update(root=str(root),base=str(root),case_index={'path':'mock-cases'},
                    source_build=p.evidence(manifest_path),source_commit=p.COMMIT,prerequisites={},model=p.evidence(manifest_path),
                    runner=p.evidence(p.__file__),native_source=p.evidence(HERE/'multisequence_projection_candidates_v1.cpp'))
                (root/'build').mkdir();(root/'build/compile.json').write_text('{}')
                compiled={'binary':p.evidence(manifest_path),'link':{'libraries':{'cudart':{'path':'/mock/libcudart.so'}}}}
                runtime={'libcuda':{'path':'/mock/libcuda.so'}}
                fake=SimpleNamespace(returncode=None,pid=123,wait_calls=0,terminated=0,killed=0)
                def wait(timeout):
                    fake.wait_calls+=1
                    if mode.startswith('timeout') and fake.wait_calls==1:raise subprocess.TimeoutExpired('mock',timeout)
                    if mode=='timeout_kill' and fake.wait_calls==2:raise subprocess.TimeoutExpired('mock',timeout)
                    fake.returncode=2 if mode=='native_failure' else 0
                    return fake.returncode
                def terminate():fake.terminated+=1
                def kill():fake.killed+=1
                fake.wait=wait;fake.terminate=terminate;fake.kill=kill;fake.poll=lambda:fake.returncode
                def spawn(*args,**kwargs):
                    kwargs['stdout'].write(''.join(json.dumps(row)+'\n' for row in records));kwargs['stdout'].flush()
                    return fake
                child=SimpleNamespace(qualifier=SimpleNamespace(verify_runtime=lambda *a:runtime))
                with patch.object(p,'validate_manifest',return_value=m),patch.object(p,'validate_compile',return_value=compiled),\
                     patch.object(p,'load',return_value=child),patch.object(p,'validate_maps',return_value={}),\
                     patch.object(p.subprocess,'Popen',side_effect=spawn) as popen:
                    if mode=='success':self.assertTrue(p.run_probe(manifest_path,1)['completed'])
                    else:
                        with self.assertRaises(ValueError):p.run_probe(manifest_path,1)
                        self.assertFalse((root/'run/receipt.json').exists());self.assertTrue((root/'run/failure.json').exists())
                    self.assertEqual(popen.call_count,1)
                    self.assertTrue(p.read(root/'run/process-exit.json')['owned_process_exited'])
                    self.assertEqual(fake.terminated,int(mode.startswith('timeout')))
                    self.assertEqual(fake.killed,int(mode=='timeout_kill'))

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable')
    def test_host_syntax_actual_public_abi_only_cuda_uuid_stub(self):
        # Syntax only, not a CUDA compilation or link/runtime/numerical proof.
        root=HERE.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/'cuda.h').write_text('struct CUuuid { char bytes[16]; };\nconstexpr int CUDA_SUCCESS=0;\nextern "C" int cuDeviceGetUuid(CUuuid*,int);\n')
            command=['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsyntax-only','-I'+str(root/'kernels/include'),'-I'+str(d),str(HERE/'multisequence_projection_candidates_v1.cpp')]
            subprocess.run(command,check=True,capture_output=True,text=True)
            (d/'size.cpp').write_text('#include "riley_cuda.h"\nstatic_assert(sizeof(RileyCudaGemmAlgorithmInfo)==112);\n')
            subprocess.run(['c++','-std=c++17','-fsyntax-only','-I'+str(root/'kernels/include'),str(d/'size.cpp')],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
