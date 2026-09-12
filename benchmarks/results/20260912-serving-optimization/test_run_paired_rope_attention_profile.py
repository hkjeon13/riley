"""Focused CPU contract tests; no remote, process launch, GPU or native build."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_diagnostic_tests',HERE/'run_paired_rope_attention_profile.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def records(mode='rope_attention'):
    result=[]
    for phase,capture in [('prefill',10),('decode',20)]:
        result.append({'schema':m.OP_SCHEMA,'kind':'capture','capture_id':capture,'phase':phase})
    for index in range(1,513):
        position=127+(index-1)%32;capture=10 if position==127 else 20
        replay={'schema':m.WHOLE_SCHEMA,'kind':'replay','replay_id':index,'capture_id':capture,'position':position,
                'event_status':0,'launch_status':0,'completion_status':0,'event_setup_ns':10,'host_staging_ns':11,
                'host_launch_ns':12,'host_wait_ns':13,'cuda_graph_span_ns':14}
        result.append(replay)
        if mode!='off' and position>=128:
            op={k:replay[k] for k in ('replay_id','capture_id','position','launch_status','completion_status')}
            op.update(schema=m.OP_SCHEMA,kind='operation',operator='rope_attention',cuda_sum_ns=60,
                      intervals=[{'layer':i,'event_status':0,'cuda_span_ns':2} for i in range(30)])
            result.append(op)
    return result


def report(mode='rope_attention'):
    def met(n,value):return {'measured_count':n,'unmeasured_count':0,'median':value,'sum':n*value}
    captures=[]
    for phase in ('prefill','decode'):
        count=60 if mode!='off' and phase=='decode' else 0
        captures.append({'phase':phase,'operator':mode,'source_sha256':'source','tool_sha256':'tool',
                         'historical_tool_sha256':'historical','projection_events_compiled':False,
                         'expected_event_records':count,'recorded_event_records':count})
    return {'captures':captures,'whole_graph':{'failed_replays':0,'projection_groups':[],
            'groups':{phase:{'replays':n,'successful_replays':n,'metrics':{'cuda_graph_span_ns':met(n,100)}}
                      for phase,n in [('prefill',6),('decode',186)]}},
            'operator_groups':[] if mode=='off' else [{'operator':'rope_attention','replays':186,'layers':list(range(30)),
                          'sum_across_selected_layers_ns':met(186,60)}]}


class NativeTests(unittest.TestCase):
    def test_complete_inventory_and_exact_warmup_boundary(self):
        raw=records();full=m.validate_native(raw,'rope_attention')
        self.assertEqual((full['replays'],full['prefill'],full['decode'],full['operator_replays']),(512,16,496,496))
        kept=[r for r in raw if 'replay_id' not in r or r['replay_id']>=321]
        actual=m.validate_native(kept,'rope_attention',True)
        self.assertEqual((actual['replays'],actual['prefill'],actual['decode'],actual['operator_replays']),(192,6,186,186))
        self.assertEqual(m.validate_native(records('off'),'off')['operator_replays'],0)

    def test_replay_missing_duplicate_reordered_or_wrong_position_rejected(self):
        initial=records()
        changes=[lambda r:r.pop(2),lambda r:r.insert(2,copy.deepcopy(r[2])),
                 lambda r:r[2].update(replay_id=2),lambda r:r[2].update(position=128),
                 lambda r:r[2].update(capture_id=20),lambda r:r[2].update(event_status=1),
                 lambda r:r[2].update(cuda_graph_span_ns=None)]
        for change in changes:
            raw=copy.deepcopy(initial);change(raw)
            with self.subTest(change=change),self.assertRaises(ValueError):m.validate_native(raw,'rope_attention')

    def test_combined_interval_missing_failed_or_wrong_layer_never_reconstructs_success(self):
        initial=records();index=next(i for i,r in enumerate(initial) if r['kind']=='operation')
        changes=[lambda r:r.pop(index),lambda r:r[index].update(cuda_sum_ns=None),
                 lambda r:r[index].update(cuda_sum_ns=61),lambda r:r[index].update(operator='attention'),
                 lambda r:r[index]['intervals'][0].update(event_status=1),
                 lambda r:r[index]['intervals'][0].update(layer=1),
                 lambda r:r[index]['intervals'][0].update(cuda_span_ns=float('nan')),
                 lambda r:r[index].update(capture_id=10)]
        for change in changes:
            raw=copy.deepcopy(initial);change(raw)
            with self.subTest(change=change),self.assertRaises(ValueError):m.validate_native(raw,'rope_attention')
        with self.assertRaises(ValueError):m.validate_native(initial,'off')

    def test_warmup_failure_and_prefill_projection_are_rejected(self):
        raw=records();raw[2]['completion_status']=1
        with self.assertRaises(ValueError):m.validate_native(raw,'rope_attention')
        raw=records();raw.append({'schema':m.WHOLE_SCHEMA,'kind':'projection'})
        with self.assertRaisesRegex(ValueError,'projection hooks'):m.validate_native(raw,'rope_attention')

    def test_extraction_summarizes_only_last_six_and_preserves_inventory(self):
        for mode in ('off','rope_attention'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);raw=root/'server.log'
                raw.write_text('server startup\n'+''.join(json.dumps(r)+'\n' for r in records(mode)))
                seen=[]
                def summarize(path):seen.extend(m.native_records(path));return report(mode)
                bundle={'builder':SimpleNamespace(GRAPH_SHA='source',TOOL_SHA='tool',HISTORICAL_SHA='historical'),
                        'tool':SimpleNamespace(summarize=summarize)}
                _,proof=m.extract_and_summarize(raw,root,bundle,mode)
                self.assertEqual(proof['retained']['first_replay_id'],321)
                self.assertEqual(min(r['replay_id'] for r in seen if 'replay_id' in r),321)
                self.assertEqual(len([r for r in seen if r['kind']=='capture']),2)
                self.assertFalse(proof['performance_claim_eligible'])

    def test_wrong_capture_tool_or_missing_metric_rejects_summary(self):
        for mutate in [lambda r:r['captures'][0].update(tool_sha256='wrong'),
                       lambda r:r['whole_graph']['groups']['decode']['metrics']['cuda_graph_span_ns'].update(unmeasured_count=1),
                       lambda r:r['operator_groups'][0]['sum_across_selected_layers_ns'].update(measured_count=185)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);raw=root/'server.log';raw.write_text(''.join(json.dumps(r)+'\n' for r in records()))
                r=report();mutate(r)
                bundle={'builder':SimpleNamespace(GRAPH_SHA='source',TOOL_SHA='tool',HISTORICAL_SHA256='historical',HISTORICAL_SHA='historical'),
                        'tool':SimpleNamespace(summarize=lambda _:r)}
                with self.assertRaises(ValueError):m.extract_and_summarize(raw,root,bundle,'rope_attention')
                self.assertFalse((root/'native-validation.json').exists())


    def test_both_frozen_real_summarizers_accept_only_bound_full_records(self):
        scripts=HERE.parents[1]/'scripts'
        for name,filename in [('baseline','profile_decode_rope_attention_baseline.py'),('candidate','profile_decode_operators_batch8.py')]:
            spec=importlib.util.spec_from_file_location('paired_real_'+name,scripts/filename)
            tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
            for mode in ('off','rope_attention'):
                with self.subTest(name=name,mode=mode),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);raw=records(mode)
                    extra=[]
                    for capture in raw[:2]:
                        count=30 if mode!='off' else 0
                        edges=2*count if capture['phase']=='decode' else 0
                        capture.update(operator=mode,layer_mask=(1<<30)-1 if count else 0,interval_count=count,
                                       expected_event_records=edges,recorded_event_records=edges,source_sha256=tool.SOURCE_SHA256,
                                       tool_sha256=m.sha(scripts/filename),historical_tool_sha256=tool.HISTORICAL_SHA256,
                                       projection_events_compiled=False)
                        if name=='baseline':capture.update(baseline_source_commit=tool.BASELINE_SOURCE_COMMIT,baseline_build_sha256=tool.BASELINE_BUILD_SHA256)
                        extra.append(dict(schema=m.OP_SCHEMA,kind='inventory',capture_id=capture['capture_id'],inventory_status=0,event_record_nodes=edges,event_wait_nodes=0))
                    raw=extra+raw
                    path=root/'server.log';path.write_text(''.join(json.dumps(r)+'\n' for r in raw))
                    builder=SimpleNamespace(GRAPH_SHA=tool.SOURCE_SHA256,TOOL_SHA=m.sha(scripts/filename),HISTORICAL_SHA=tool.HISTORICAL_SHA256)
                    summary,validation=m.extract_and_summarize(path,root,{'builder':builder,'tool':tool},mode)
                    self.assertEqual(summary['whole_graph']['groups']['decode']['replays'],186)
                    self.assertEqual(validation['retained']['prefill'],6)


class BuildBindingTests(unittest.TestCase):
    def test_completed_build_receipt_binds_binary_graph_tool_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tool=root/'tools/profile.py';tool.parent.mkdir();tool.write_text('# fixture')
            (tool.parent/'profile_owned_graph.py').write_text('# historical fixture')
            target=root/'target';(target/'release').mkdir(parents=True);binary=target/'release/riley';binary.write_bytes(b'binary')
            (root/'builder.py').write_text('# pinned fixture')
            base={'build_environment':{'CARGO_TARGET_DIR':'/original','CUDA_HOME':'/cuda'}}
            m.write(root/'base.json',base);(root/'diag-build.log').write_text('build passed')
            gate={'finalized':True};verified=[]
            builder=SimpleNamespace(ROOT=root,SOURCE=root/'source',TARGET=target,TOOL=tool,BASE_COMMIT='c'*40,
                BASE_BUILD_SHA=m.sha(root/'base.json'),GRAPH='kernels/src/graph_resources.cu',GRAPH_SHA='a'*64,INSTRUMENTED_SHA='b'*64,
                TOOL_SHA=m.sha(tool),HISTORICAL_SHA=m.sha(tool.parent/'profile_owned_graph.py'),
                campaign_gate=lambda:gate,verify_base=lambda b:verified.append('base'),verify_instrumented=lambda b:verified.append('instrumented'))
            instrument=dict(source_root=str(builder.SOURCE),source_file=builder.GRAPH,operator_enqueue_count_per_interval=2,
                selected_decode_events_per_layer=2,prefill_operator_event_count=0,original_sha256=builder.GRAPH_SHA,instrumented_sha256=builder.INSTRUMENTED_SHA,
                tool_sha256=builder.TOOL_SHA,historical_tool_sha256=builder.HISTORICAL_SHA,applied=True,projection_events=False,
                synchronizations_added=0,performance_claim_eligible=False,baseline_build=m.evidence(root/'base.json'),baseline_source_commit=builder.BASE_COMMIT)
            m.write(root/'diag-instrumentation.json',instrument)
            build=dict(measurement_finalization=gate,builder_sha256='d'*64,base_source_commit=builder.BASE_COMMIT,base_build_sha256=builder.BASE_BUILD_SHA,
                source_root=str(builder.SOURCE),source_clean=False,instrumented_source_sha256=builder.INSTRUMENTED_SHA,
                tool_sha256=builder.TOOL_SHA,historical_tool_sha256=builder.HISTORICAL_SHA,instrumentation=instrument,
                binary=str(binary),binary_sha256=m.sha(binary),build_log_sha256=m.sha(root/'diag-build.log'),
                execution_started=False,performance_claim_eligible=False,
                build_argv=['cargo','build','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley'],
                build_environment=dict(base['build_environment'],CARGO_TARGET_DIR=str(target),PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:/bin'))
            path=root/'diag-build.json';m.write(path,build)
            lane={'build':m.evidence(root/'base.json')}
            spec=dict(builder='builder.py',builder_sha='d'*64,base='base.json',prefix='diag',graph='b'*64)
            def load(path,expected,name):return builder if path.name=='builder.py' else SimpleNamespace(SOURCE_SHA256=builder.GRAPH_SHA,HISTORICAL_SHA256=builder.HISTORICAL_SHA)
            with patch.dict(m.LANES,{'baseline':spec}),patch.object(m,'load',side_effect=load):
                result=m.validate_build(root,'baseline',gate,lane)
                self.assertEqual(verified,['base','instrumented']);self.assertEqual(result['build']['binary'],str(binary))
                for key,value in [('binary_sha256','0'*64),('measurement_finalization',{}),('instrumented_source_sha256','0'*64),
                                  ('tool_sha256','0'*64),('execution_started',True),('performance_claim_eligible',True)]:
                    bad=copy.deepcopy(build);bad[key]=value;path.write_text(json.dumps(bad))
                    with self.subTest(key=key),self.assertRaises(ValueError):m.validate_build(root,'baseline',gate,lane)
                path.write_text(json.dumps(build))
                with self.assertRaisesRegex(ValueError,'qualified API lane'):
                    m.validate_build(root,'baseline',gate,{'build':{'path':'elsewhere','sha256':'x'}})


class EnvironmentAndLifecycleTests(unittest.TestCase):
    def test_environment_uses_frozen_clean_validator_without_mutating_it(self):
        calls=[]
        def clean(parent,lane,runtime):calls.append((parent,lane,runtime));return {'HOME':'/home/x','PATH':'/bin','LD_LIBRARY_PATH':'/private'}
        core=SimpleNamespace(lane_environment=clean);parent={};lane={};runtime={}
        env=m.diagnostic_environment(core,parent,lane,runtime,'rope_attention')
        self.assertEqual(len(calls),1)
        self.assertIs(core.lane_environment,clean)
        self.assertEqual(env['RILEY_DECODE_OPERATOR'],'rope_attention')
        self.assertEqual(env['RILEY_OWNED_GRAPH_PROJECTION'],'off')
        self.assertEqual(env['RILEY_DECODE_OPERATOR_LAYERS'],'all')
        self.assertEqual((parent,lane,runtime),({},{},{}))
        with self.assertRaises(ValueError):m.diagnostic_environment(core,parent,lane,runtime,'qkv')

    def context(self,root,fail_stop=False,fail_restore=False):
        session=SimpleNamespace(SNAPSHOT=root/'snapshot.json')
        session.validate_runtime=lambda:{'v':1};session.check=lambda _:['a','b','c'];session.bound_runtime=lambda:{'v':1}
        actions=[]
        def action(plan,verb,out):
            actions.append(verb)
            if verb=='stop':
                session.SNAPSHOT.write_text('{}')
                if fail_stop:raise RuntimeError('partial stop')
            if verb=='restore' and fail_restore:raise RuntimeError('restore pending')
        class Watchdog:
            def __init__(self,*_):pass
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def check(self):pass
        core=SimpleNamespace(shared=SimpleNamespace(check_port=lambda _:None),session_action=action,Watchdog=Watchdog,
                             verify_restoration=lambda *_:{'path':'restored','sha256':'x'})
        context={'core':core,'session':session,'run_plan':{},'parent':{'lanes':{'baseline':{'port':1},'candidate':{'port':2}}}}
        return context,{'runtime':{'v':1},'sessions':['a','b','c']},actions

    def test_startup_failure_closes_only_created_server_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);process=SimpleNamespace(pid=123)
            core=SimpleNamespace(lane_argv=lambda *_:['/owned/riley'],lane_environment=lambda *_:{'HOME':'/home/x','PATH':'/bin'},
                shared=SimpleNamespace(check_port=lambda _:None,wait_ready=lambda *_:(_ for _ in ()).throw(RuntimeError('readiness failed'))),
                cooldown=lambda *_:{})
            parent={'lanes':{'baseline':{'argv':['/original/riley'],'port':19431,'cwd':'/original'}}}
            bundle={'build':{'source_root':str(root),'binary':'/owned/riley'}}
            context=dict(core=core,session=object(),runtime={},parent=parent,bundles={'baseline':bundle},binding={},
                         run_plan={'startup_timeout_seconds':1})
            cleanup={'cleanup_verified':True,'remaining_owned_pids':[],'returncode':0}
            with patch.object(m.subprocess,'Popen',return_value=process) as popen,patch.object(m.os,'getsid',return_value=123),\
                 patch.object(m,'cleanup_owned',return_value=cleanup) as close,self.assertRaisesRegex(ValueError,'incomplete'):
                m.run_case(context,{},root/'case','baseline','off',SimpleNamespace(check=lambda:None))
            close.assert_called_once();self.assertIs(close.call_args.args[1],process)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            final=m.read(root/'case/process-exit.json')
            self.assertEqual(final['cleanup'],cleanup);self.assertEqual(final['failure']['message'],'readiness failed')
            self.assertFalse((root/'case/completion.json').exists())
            with patch.object(m.subprocess,'Popen',return_value=process),patch.object(m.os,'getsid',return_value=123),\
                 patch.object(m,'cleanup_owned',side_effect=RuntimeError('owned cleanup failed')),self.assertRaises(ValueError):
                m.run_case(context,{},root/'failed-cleanup','baseline','off',SimpleNamespace(check=lambda:None))
            failure=m.read(root/'failed-cleanup/process-exit.json')['failure']
            self.assertEqual(failure['primary']['message'],'readiness failed')
            self.assertEqual(failure['cleanup']['message'],'owned cleanup failed')
            self.assertFalse(failure['owned_cleanup_verified'])

    def test_changed_preflight_prevents_any_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,prepared,actions=self.context(root)
            with patch.object(m,'unchanged',side_effect=ValueError('pin changed')),self.assertRaises(ValueError):
                m.measure(context,prepared,root)
            self.assertEqual(actions,[])
            self.assertFalse((root/'completion.json').exists())

    def test_partial_stop_still_restores_no_success_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,prepared,actions=self.context(root,fail_stop=True)
            with patch.object(m,'unchanged'),self.assertRaisesRegex(ValueError,'incomplete'):
                m.measure(context,prepared,root)
            self.assertEqual(actions,['stop','restore'])
            self.assertIsNotNone(m.read(root/'finalization.json')['failure'])
            self.assertFalse((root/'completion.json').exists())

    def test_lane_failure_or_interruption_preserves_prior_completion_and_restores(self):
        for exception in (RuntimeError('token mismatch'),KeyboardInterrupt()):
            with self.subTest(exception=exception),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);context,prepared,actions=self.context(root)
                calls=[]
                def case(ctx,prep,directory,name,mode,watchdog):
                    calls.append(name)
                    if len(calls)==2:raise exception
                    directory.mkdir();m.write(directory/'completion.json',{'completed':True})
                    return {},report(mode)
                with patch.object(m,'unchanged'),patch.object(m,'run_case',side_effect=case),self.assertRaises(ValueError):
                    m.measure(context,prepared,root)
                self.assertEqual(actions,['stop','restore'])
                self.assertEqual(len(m.read(root/'finalization.json')['completed_processes']),1)
                self.assertFalse((root/'comparison.json').exists())

    def test_failed_restore_keeps_watchdog_responsibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,prepared,actions=self.context(root,fail_stop=True,fail_restore=True)
            with patch.object(m,'unchanged'),self.assertRaises(ValueError):m.measure(context,prepared,root)
            self.assertTrue(m.read(root/'finalization.json')['failure']['independent_watchdog_remains_responsible'])
            self.assertFalse((root/'completion.json').exists())

    def test_eight_fresh_cases_preserve_ab_ba_order_and_compare_only_after_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,prepared,actions=self.context(root)
            m.write(root/'preparation.json',prepared)
            calls=[]
            def case(ctx,prep,directory,name,mode,watchdog):
                calls.append((directory.name,name,mode));directory.mkdir();m.write(directory/'completion.json',{'completed':True})
                return {},report(mode)
            with patch.object(m,'unchanged'),patch.object(m,'run_case',side_effect=case):result=m.measure(context,prepared,root)
            self.assertEqual(calls,m.SCHEDULE);self.assertEqual(actions,['stop','restore'])
            self.assertEqual(len(result['processes']),8)
            compared=m.read(root/'comparison.json')
            self.assertEqual([p['order'] for p in compared['paired_combined_intervals']],['AB','BA'])
            self.assertTrue(all(p['candidate_over_baseline']==1 for p in compared['paired_combined_intervals']))
            self.assertIsNone(compared['candidate_acceptance'])
            self.assertFalse(result['performance_claim_eligible'])

    def test_default_cli_preparation_never_measures_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'
            with patch.object(m,'prepare',return_value=({}, {'prepared':True})),patch.object(m,'measure') as measured:
                m.main(['--output',str(out)]);measured.assert_not_called()
                with self.assertRaisesRegex(ValueError,'output exists'):m.main(['--output',str(out)])


if __name__=='__main__':unittest.main()
