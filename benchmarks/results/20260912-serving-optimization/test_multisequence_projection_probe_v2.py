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
spec = importlib.util.spec_from_file_location('multisequence_probe_under_test', HERE/'multisequence_projection_probe_v2.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def metadata(shape, m):
    _, n, k, custom = p.SHAPES[shape]
    return {'struct_size':112,'backend':1,'algorithm_id':13,'tile_id':0,'stages_id':0,'split_k':1,
            'reduction_scheme':0,'cta_swizzling':0,'custom_option':custom,'deterministic':1,'workspace_bytes':0,
            'numerical_implementation_flags':131585,'compute_capability_major':8,'compute_capability_minor':9,
            'runtime_version':13000,'cublaslt_version':130101,'m':m,'n':n,'k':k,'reserved':[0,0]}


def sample_records(rejected=False, mismatch=False):
    def r(kind, **kw): return {'schema_version':p.SCHEMA+'-native.v1','kind':kind,**kw}
    records=[r('device',uuid=p.UUID,runtime_version=13000,compute_capability_major=8,
               compute_capability_minor=9,abi=1,native_build_info=p.NATIVE_BUILD_INFO,maps='mock')]
    for shape in range(5):
        for m in (1,2,4):
            reject = rejected and shape == 0 and m == 2
            row=r('plan',shape=shape,projection=p.SHAPES[shape][0],m=m,workspace_cap=16*1024*1024 if m==1 else 0,
                  status=11 if reject else 0,admitted=not reject,metadata=None if reject else metadata(shape,m))
            if reject: row.update(error='unsupported',error_stage=10,error_domain=1,error_native_code=0)
            else: row['exact_metadata']=True
            records.append(row)
    cases=[]
    for group in range(121):
        shape,layer=(group%4,group//4) if group<120 else (4,30)
        _,n,k,_=p.SHAPES[shape]
        for pattern in range(3):
            for m,active in p.MODES:
                c={'case_id':len(cases),'layer':layer,'shape':shape,'m':m,'active_rows':active,'n':n,'k':k,'pattern':pattern}
                cases.append(c)
                if rejected and shape==0 and m==2:
                    records.append(r('case',**c,executed=False,reason='anchored_descriptor_not_supported')); continue
                rows=[{'row':row,'words':n,'mismatches':0,'first_mismatch':None,'oracle_bits':None,'batched_bits':None} for row in range(m)]
                count=int(mismatch and c['case_id']==0)
                if count: rows[0].update(mismatches=1,first_mismatch=0,oracle_bits=0,batched_bits=32768)
                records.append(r('case',**c,executed=True,words=m*n,mismatches=count,rows=rows,
                                 **{key:count==0 if key=='exact_outputs' else True for key in p.CASE_FLAGS}))
    records += [r('runtime_end',maps='mock'),r('summary',completed=True,cases=1089,all_plans_admitted=not rejected,
               all_outputs_exact=not(rejected or mismatch),all_resources_closed=True,performance_measured=False,
               performance_claim_eligible=False,failure='')]
    return records,{'case_records':cases}


class ProbeTests(unittest.TestCase):
    def test_v1_preserved_and_native_copy_is_exact(self):
        self.assertEqual(p.sha(HERE/'multisequence_projection_probe.py'),
                         'ecdda89c60d8e04b53e6e69d0101c5d511f36d91a0f693771670e386dee1855c')
        self.assertEqual(p.sha(HERE/'multisequence_projection_probe.cpp'),
                         '962e3e8ff6cc58249254c75e62599b65e44521ebd4932cb546197738551ae3e9')
        self.assertEqual((HERE/'multisequence_projection_probe.cpp').read_bytes(),
                         (HERE/'multisequence_projection_probe_v2.cpp').read_bytes())
        self.assertEqual(p.sha(HERE/'test_multisequence_projection_probe.py'),
                         '0f6afe1d0b39f6a1862c96a808738706ffbc96b2c7f01a3b5556cfa19de8b9b9')

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
                    self.assertEqual(manifest['schema_version'],p.SCHEMA+'-fixtures.v2')
                    self.assertEqual(manifest['pattern_count'],3);self.assertEqual(returned['pattern_count'],3)
                    self.assertEqual(manifest['patterns'],list(p.PATTERNS))
                    self.assertTrue(all(manifest[k]==v for k,v in p.COUNTS.items()))
                    self.assertEqual(manifest['native_source'],p.evidence(HERE/'multisequence_projection_probe_v2.cpp'))
                    original=path.read_text()
                    for key,value in [('pattern_count',2),('pattern_count',list(p.PATTERNS)),
                                      ('patterns',3),('patterns',list(reversed(p.PATTERNS))),
                                      ('schema_version',p.SCHEMA+'-fixtures.v1')]:
                        changed=json.loads(original);changed[key]=value;path.write_text(json.dumps(changed))
                        with self.subTest(key=key,value=value),self.assertRaises(ValueError):module.validate_manifest(path)
                    path.write_text(original)

    def test_case_plan_coverage(self):
        records,m=sample_records();r=p.validate_records(records,m)
        self.assertEqual(r['executed_cases'],1089);self.assertTrue(r['arithmetic_equal'])
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
        records[16]['rows'][0]['batched_bits']=0
        with self.assertRaisesRegex(ValueError,'mismatch evidence'):p.validate_records(records,m)

    def test_metadata_gate_every_field(self):
        for field in metadata(0,2):
            a=metadata(0,2);a[field]=[1,0] if field=='reserved' else a[field]+1
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_metadata(a,0,2)

    def test_case_missing_duplicate_inactive_row_rejected(self):
        original,m=sample_records()
        for mutation in ('missing','duplicate','last_row'):
            records=copy.deepcopy(original)
            if mutation=='missing':records.pop(16)
            elif mutation=='duplicate':records[17]=copy.deepcopy(records[16])
            else:records[18]['rows'][-1]['row']=0
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):p.validate_records(records,m)

    def test_guards_cleanup_accounting_fail_closed(self):
        original,m=sample_records()
        for key in p.CASE_FLAGS[2:]:
            records=copy.deepcopy(original);records[16][key]=False
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
                    runner=p.evidence(p.__file__),native_source=p.evidence(HERE/'multisequence_projection_probe_v2.cpp'))
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
            command=['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsyntax-only','-I'+str(root/'kernels/include'),'-I'+str(d),str(HERE/'multisequence_projection_probe_v2.cpp')]
            subprocess.run(command,check=True,capture_output=True,text=True)
            (d/'size.cpp').write_text('#include "riley_cuda.h"\nstatic_assert(sizeof(RileyCudaGemmAlgorithmInfo)==112);\n')
            subprocess.run(['c++','-std=c++17','-fsyntax-only','-I'+str(root/'kernels/include'),str(d/'size.cpp')],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
