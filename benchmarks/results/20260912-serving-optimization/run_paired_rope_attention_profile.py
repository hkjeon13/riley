"""Prepare/run eight isolated C1 combined RoPE+attention diagnostics.

Default is read-only preparation. --measure explicitly pauses only Round15's
verified three Blender successors and restores them in finally. Requires both
completed diagnostic builds and generated Round15 helper; never builds anything.
This is instrumentation evidence, never a serving performance/acceptance gate.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading

ROOT = Path('/tmp/riley-opt-260912')
FACTORY_SHA = 'e7ee002a0c2d77d76e2262b24703501fe9b9ea305491894d31460879096ca876'
MODULE_PINS = {
    'run_serving_token_optimization_v4.py': '4b51034247b4301e390897788ffa3f7c0f1ce26bb313a9fc718afebc6524ee0e',
    'run_serving_concurrency.py': '73f8fb3fa09d623dbeb36616ebe0ee7d45e291ca3da6e515e3b56ddf45f27a6f',
    'run_serving_optimization.py': 'e1d0f34c11d58a577232738d645cf5821fc6d71132757269d1d02429965deee8',
    'serving_token_client.py': 'a2a4a35569d6b892542097c119e2be8a6402beb60ab1565aa95658794c364766',
    'serving_token_client_v2.py': '2bc9238265666b99654f4f4e456ca0bb10c45b8bed8f28493452b89c1f450fcf',
}
LANES = {
    'baseline': {'builder': 'build_decode_profile_baseline_combined.py',
                 'builder_sha': 'dcd9796c78897dc65d863362fd3b1382b94280cb8ba86df03cc8ec73e5a75017',
                 'base': 'http-token-build.json', 'prefix': 'decode7-combined-profile',
                 'graph': '91d473d01047dc8ad6dad475a567b92d3a290e33bb0a7ed20359d719642cd485'},
    'candidate': {'builder': 'build_decode_profile_batch8.py',
                  'builder_sha': 'c21db7598264a10735b9f38fd3dba5c69233f63fa94ebf2a00c5bafe7380b67a',
                  'base': 'batch8-build.json', 'prefix': 'decode8-profile',
                  'graph': '604b99c668729667cd39ef3d42f3fcd5c0dfd8088cc8f3277ae4d6f85b27f372'},
}
SCHEDULE = [('off-before-baseline', 'baseline', 'off'), ('off-before-candidate', 'candidate', 'off'),
            ('combined-ab-baseline', 'baseline', 'rope_attention'), ('combined-ab-candidate', 'candidate', 'rope_attention'),
            ('combined-ba-candidate', 'candidate', 'rope_attention'), ('combined-ba-baseline', 'baseline', 'rope_attention'),
            ('off-after-baseline', 'baseline', 'off'), ('off-after-candidate', 'candidate', 'off')]
WHOLE_SCHEMA = 'riley.owned-graph-diagnostic.v1'
OP_SCHEMA = 'riley.decode-operator-diagnostic.v1'
SCHEMA = 'riley.paired-rope-attention-diagnostic.v1'
WORKLOAD = {'offered_concurrency': 1, 'warmup_nonstream': 5, 'warmup_stream': 5,
            'retained_stream': 6, 'prompt_tokens': 128, 'output_tokens': 32,
            'requests_per_process': 16, 'total_replays': 512, 'retained_replay_id_min': 321,
            'retained_replays': 192, 'retained_prefill': 6, 'retained_decode': 186}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def evidence(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load(path, expected, name):
    require(sha(path) == expected, 'frozen helper changed: ' + str(path))
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def validate_build(root, name, gate, parent_lane):
    spec = LANES[name]
    builder = load(root/spec['builder'], spec['builder_sha'], 'paired_builder_' + name)
    require(__debug__ and builder.ROOT == root and builder.campaign_gate() == gate, 'diagnostic build campaign gate differs')
    base = read(root/spec['base'])
    require(parent_lane['build'] == evidence(root/spec['base']), 'diagnostic base differs from qualified API lane')
    builder.verify_base(base)
    builder.verify_instrumented(base)
    path = root/(spec['prefix'] + '-build.json')
    build = read(path)
    instrumentation_path = root/(spec['prefix'] + '-instrumentation.json')
    instrument = read(instrumentation_path)
    require(build['measurement_finalization'] == gate and build['builder_sha256'] == spec['builder_sha']
            and build['base_source_commit'] == builder.BASE_COMMIT and build['base_build_sha256'] == builder.BASE_BUILD_SHA,
            'diagnostic build source/gate identity differs')
    require(build['source_root'] == str(builder.SOURCE) and build['source_clean'] is False
            and build['instrumented_source_sha256'] == spec['graph'] == builder.INSTRUMENTED_SHA,
            'diagnostic source identity differs')
    require(build['tool_sha256'] == builder.TOOL_SHA and build['historical_tool_sha256'] == builder.HISTORICAL_SHA
            and build['instrumentation'] == instrument, 'instrumentation receipt differs')
    require(instrument['source_root'] == str(builder.SOURCE) and instrument['source_file'] == builder.GRAPH,
            'instrumentation source path differs')
    require(instrument['original_sha256'] == builder.GRAPH_SHA and instrument['instrumented_sha256'] == spec['graph']
            and instrument['tool_sha256'] == builder.TOOL_SHA and instrument['historical_tool_sha256'] == builder.HISTORICAL_SHA
            and instrument['applied'] is True and instrument['projection_events'] is False
            and instrument['synchronizations_added'] == 0 and instrument['performance_claim_eligible'] is False,
            'unsupported instrumentation')
    if name == 'baseline':
        require(instrument['baseline_build'] == evidence(root/spec['base'])
                and instrument['baseline_source_commit'] == builder.BASE_COMMIT
                and instrument['operator_enqueue_count_per_interval'] == 2
                and instrument['selected_decode_events_per_layer'] == 2
                and instrument['prefill_operator_event_count'] == 0, 'combined baseline source binding differs')
    binary = builder.TARGET/'release/riley'
    require(build['binary'] == str(binary) and build['binary_sha256'] == sha(binary)
            and build['build_log_sha256'] == sha(root/(spec['prefix'] + '-build.log'))
            and build['execution_started'] is False and build['performance_claim_eligible'] is False,
            'diagnostic binary/build receipt differs')
    require(build['build_argv'] == ['cargo','build','--release','-p','riley-server','--features','server,bench,cuda','--bin','riley'],
            'diagnostic build command differs')
    expected_env = dict(base['build_environment'], CARGO_TARGET_DIR=str(builder.TARGET))
    recorded_env = build['build_environment']
    require(set(recorded_env) == set(expected_env) | {'PATH'}
            and all(recorded_env[k] == v for k,v in expected_env.items())
            and recorded_env['PATH'].startswith('/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'), 'diagnostic build environment differs')
    tool = load(builder.TOOL, builder.TOOL_SHA, 'paired_operator_' + name)
    require(tool.SOURCE_SHA256 == builder.GRAPH_SHA and tool.HISTORICAL_SHA256 == builder.HISTORICAL_SHA, 'profiler module source differs')
    files = [root/spec['builder'], root/spec['base'], path, instrumentation_path,
             root/(spec['prefix']+'-build.log'), binary, builder.TOOL, builder.TOOL.with_name('profile_owned_graph.py')]
    return {'builder': builder, 'tool': tool, 'build': build, 'base': base,
            'files': {str(p.resolve()): sha(p) for p in files}}


def diagnostic_environment(core, parent, lane, runtime, mode):
    require(mode in ('off', 'rope_attention'), 'unsupported diagnostic mode')
    # Validate the uninstrumented explicit environment first, then add only the
    # four fixed diagnostic keys. Never bypass or mutate V4's environment rules.
    env = core.lane_environment(parent, lane, runtime)
    env.update(RILEY_OWNED_GRAPH_PROFILE='1', RILEY_OWNED_GRAPH_PROJECTION='off',
               RILEY_DECODE_OPERATOR=mode, RILEY_DECODE_OPERATOR_LAYERS='all')
    require({k for k in env if k.startswith('RILEY_')} == {'RILEY_OWNED_GRAPH_PROFILE','RILEY_OWNED_GRAPH_PROJECTION',
            'RILEY_DECODE_OPERATOR','RILEY_DECODE_OPERATOR_LAYERS'}, 'unexpected diagnostic environment')
    return env


def prepare(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT and __debug__, 'authorized root and nonoptimized Python required')
    for name, expected in MODULE_PINS.items():
        require(sha(root/name) == expected, 'controller dependency changed: ' + name)
    core = load(root/'run_serving_token_optimization_v4.py', MODULE_PINS['run_serving_token_optimization_v4.py'], 'paired_token_core')
    require(Path(core.tokens.__file__).resolve() == root/'serving_token_client_v2.py'
            and Path(core.qualified_tokens.__file__).resolve() == root/'serving_token_client.py'
            and Path(core.legacy.__file__).resolve() == root/'run_serving_concurrency.py'
            and Path(core.shared.__file__).resolve() == root/'run_serving_optimization.py', 'controller import shadowed')
    builder = load(root/LANES['candidate']['builder'], LANES['candidate']['builder_sha'], 'paired_campaign_gate')
    gate = builder.campaign_gate()  # Before any output, process, GPU query or pause.
    parent_path = root/'token-serving-round14-plan.json'
    parent, previous = read(parent_path), read(root/'token-serving-round14/preparation.json')
    core.validate_workload(parent)
    core.unchanged(parent_path, parent, previous)
    values, reference = core.reference_data(parent, parent['immutable_files'])
    core.validate_measurement_client(parent, parent['immutable_files'])
    factory = load(root/'prepare_remote_session_round15.py', FACTORY_SHA, 'paired_session_factory')
    prep_path = root/'round15-session-preparation.json'
    factory_receipt = read(prep_path)
    require(factory_receipt['schema_version'] == 'riley.round15-session-preparation.v1'
            and factory_receipt['generator'] == evidence(root/'prepare_remote_session_round15.py')
            and factory_receipt['campaign_gate'] == gate and factory_receipt['lifecycle_ast_identical'] is True
            and factory_receipt['session_action_executed'] is False and factory_receipt['live_process_check_executed'] is False,
            'generated session receipt differs')
    helper = root/'remote_session_round15.py'
    require(factory_receipt['helper'] == evidence(helper), 'generated session helper hash differs')
    previous_pins = {name: factory_receipt['predecessor_inputs']['blender-round14/' + name]
                     for name in ('session.json', 'runtime.json', 'verified.json')}
    require(helper.read_text() == factory.render((root/'remote_session_round14.py').read_text(), previous_pins,
            factory_receipt['predecessor_inputs'], gate), 'generated session code is not the frozen factory output')
    session = core.load_module(helper, 'paired_round15_session')
    require(session.CAMPAIGN == root and session.ROOT == root/'blender-round15' and not session.ROOT.exists(), 'Round15 is wrong or already used')
    require(factory.gate_inputs(root, builder) == (gate, factory_receipt['predecessor_inputs'])
            and session.GENERATION_PINS == factory_receipt['predecessor_inputs'] and session.GENERATION_GATE == gate,
            'generated predecessor gate/pins differ')
    runtime = session.validate_runtime()
    require(runtime['schema_version'] == 'riley.round15-private-runtime.v1'
            and runtime['proven_sessions'] == factory_receipt['proven_successors'], 'generated runtime successor chain differs')
    sessions = session.check(runtime)
    require(len(sessions) == 3, 'three authorized sessions required')
    core.check_refs(runtime)
    bases = {}
    for name in LANES:
        lane = parent['lanes'][name]
        require(lane['kind'] == 'riley', 'qualified Riley lane required')
        core.validate_riley(lane, parent, values, parent['immutable_files'])
        bases[name] = validate_build(root, name, gate, lane)
        core.shared.check_port(lane['port'])
        diagnostic_environment(core, parent, lane, runtime, 'rope_attention')
    require(parent['nvidia_smi']['sha256'] == runtime['files'].get(parent['nvidia_smi']['path']), 'private GPU query executable differs')
    files = {str(root/name): expected for name, expected in MODULE_PINS.items()}
    for path in (Path(__file__).resolve(), root/'prepare_remote_session_round15.py', prep_path, helper):
        files[str(path)] = sha(path)
    for bundle in bases.values(): files.update(bundle['files'])
    for path, expected in factory_receipt['predecessor_inputs'].items(): files[str(root/path)] = expected
    files.update({str(path): value for path,value in runtime['files'].items()})
    run_plan = {key: parent[key] for key in ('nvidia_smi','startup_timeout_seconds','request_timeout_seconds','cooldown_timeout_seconds')}
    run_plan['session'] = {'helper': evidence(helper), 'python': parent['session']['python'], 'root': str(session.ROOT)}
    prepared = {'schema': SCHEMA, 'diagnostic_started': False, 'performance_claim_eligible': False,
                'parent_plan': evidence(parent_path), 'parent_preparation': evidence(root/'token-serving-round14/preparation.json'),
                'parent_scope': 'qualified source/model/runtime/token reference only; old serving metrics are not reinterpreted',
                'campaign_gate': gate, 'factory': evidence(prep_path), 'session': run_plan['session'],
                'runtime': runtime, 'sessions': sessions, 'reference_sha256': reference.sha256,
                'immutable_files': files, 'resolved_targets': {p:str(Path(p).resolve(strict=True)) for p in files},
                'workload': WORKLOAD, 'schedule': SCHEDULE, 'condition': core.CONDITION,
                'diagnostic_scope': 'single combined CUDA interval per layer; observed with event perturbation; no serving acceptance claim'}
    context = dict(root=root, core=core, parent=parent, previous=previous, parent_path=parent_path, builder=builder,
                   session=session, runtime=runtime, reference=reference, request=values['request'], binding=values['binding'],
                   run_plan=run_plan, bundles=bases)
    unchanged(context, prepared)
    return context, prepared


def unchanged(context, prepared):
    for path, expected in prepared['immutable_files'].items():
        require(sha(path) == expected and str(Path(path).resolve(strict=True)) == prepared['resolved_targets'][path], 'diagnostic input changed: ' + path)
    require(context['builder'].campaign_gate() == prepared['campaign_gate'], 'Round14 finalization changed')
    context['core'].unchanged(context['parent_path'], context['parent'], context['previous'])
    for name,bundle in context['bundles'].items():
        bundle['builder'].verify_base(bundle['base'])
        bundle['builder'].verify_instrumented(bundle['base'])


def native_records(log):
    result = []
    for line in Path(log).read_text().splitlines():
        if WHOLE_SCHEMA not in line and OP_SCHEMA not in line:
            continue
        row = json.loads(line)
        require(row.get('schema') in (WHOLE_SCHEMA, OP_SCHEMA), 'unknown native diagnostic schema')
        result.append(row)
    return result


def validate_native(records, mode, retained=False):
    require(mode in ('off','rope_attention'), 'unsupported mode')
    first, last = (321,512) if retained else (1,512)
    whole = [r for r in records if r['schema']==WHOLE_SCHEMA and r['kind']=='replay']
    require([r['replay_id'] for r in whole] == list(range(first,last+1)), 'native replay inventory/order differs')
    captures = [r for r in records if r['schema']==OP_SCHEMA and r['kind']=='capture']
    require(len(captures)==2 and {r['phase'] for r in captures}=={'prefill','decode'}
            and len({r['capture_id'] for r in captures})==2, 'exact two graph captures required')
    by_phase={r['phase']:r for r in captures}
    for r in whole:
        expected_position=127+(r['replay_id']-1)%32
        require(r['position']==expected_position and r['capture_id']==by_phase['prefill' if expected_position==127 else 'decode']['capture_id'],
                'replay request/position/capture binding differs')
        require(all(r[k]==0 for k in ('event_status','launch_status','completion_status'))
                and all(type(r[k]) in (float,int) and math.isfinite(r[k]) and r[k]>=0
                        for k in ('event_setup_ns','host_staging_ns','host_launch_ns','host_wait_ns','cuda_graph_span_ns')),
                'missing or failed whole-graph timing')
    operations=[r for r in records if r['schema']==OP_SCHEMA and r['kind']=='operation']
    expected=[r for r in whole if r['position']>=128] if mode!='off' else []
    require([r['replay_id'] for r in operations]==[r['replay_id'] for r in expected], 'operator replay inventory differs')
    for op,r in zip(operations,expected):
        require(all(op[k]==r[k] for k in ('capture_id','position','launch_status','completion_status'))
                and op['operator']=='rope_attention' and [v['layer'] for v in op['intervals']]==list(range(30)), 'combined layer/replay binding differs')
        spans=[v['cuda_span_ns'] for v in op['intervals']]
        require(all(v['event_status']==0 for v in op['intervals']) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in spans)
                and type(op['cuda_sum_ns']) in (int,float) and math.isfinite(op['cuda_sum_ns'])
                and abs(sum(spans)-op['cuda_sum_ns'])<=.031, 'missing/failed combined interval or sum')
    require(not any(r['schema']==WHOLE_SCHEMA and r['kind']=='projection' for r in records), 'prefill projection hooks forbidden')
    return {'replays':len(whole), 'prefill':sum(r['position']==127 for r in whole), 'decode':sum(r['position']>=128 for r in whole),
            'operator_replays':len(operations), 'first_replay_id':first, 'last_replay_id':last}


def extract_and_summarize(log, directory, bundle, mode):
    records=native_records(log)
    full=validate_native(records,mode)
    # Both warmup transports execute five requests. Select only the six final
    # requests, preserving graph/capture inventories needed by the frozen tool.
    retained=[r for r in records if 'replay_id' not in r or r['replay_id']>=321]
    kept=validate_native(retained,mode,True)
    path=directory/'retained-native.log'
    with path.open('x') as stream:
        for r in retained:stream.write(json.dumps(r,allow_nan=False)+'\n')
    report=bundle['tool'].summarize(path)
    captures=report['captures']
    for r in captures:
        require(r['operator']==mode and r['source_sha256']==bundle['builder'].GRAPH_SHA
                and r['tool_sha256']==bundle['builder'].TOOL_SHA and r['historical_tool_sha256']==bundle['builder'].HISTORICAL_SHA
                and r['projection_events_compiled'] is False, 'capture instrumentation identity differs')
        expected=60 if mode!='off' and r['phase']=='decode' else 0
        require(r['expected_event_records']==r['recorded_event_records']==expected, 'capture event edges differ')
    require(report['whole_graph']['failed_replays']==0 and not report['whole_graph']['projection_groups'], 'native diagnostic failures')
    for phase,count in [('prefill',6),('decode',186)]:
        group=report['whole_graph']['groups'][phase]
        require(group['replays']==group['successful_replays']==count, 'retained phase count differs')
        require(all(v['measured_count']==count and v['unmeasured_count']==0 for v in group['metrics'].values()), 'missing whole-graph metric')
    if mode=='off':require(report['operator_groups']==[], 'off mode captured operators')
    else:
        require(len(report['operator_groups'])==1, 'only combined family allowed')
        group=report['operator_groups'][0]
        require(group['operator']=='rope_attention' and group['replays']==186 and len(group['layers'])==30
                and group['sum_across_selected_layers_ns']['measured_count']==186
                and group['sum_across_selected_layers_ns']['unmeasured_count']==0, 'incomplete combined summary')
    write(directory/'native-summary.json',report)
    result={'full':full,'retained':kept,'raw_log':evidence(log),'retained_log':evidence(path),
            'summary':evidence(directory/'native-summary.json'),'performance_claim_eligible':False}
    write(directory/'native-validation.json',result)
    return report,result


def cleanup_owned(core, process, log):
    core.shared.stop_owned_process(process)
    members,zombies=[],[]
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit() and core.owned(int(entry.name),process.pid):
            try:
                state=(entry/'stat').read_text().rsplit(') ',1)[1].split()[0]
                (zombies if state=='Z' else members).append(int(entry.name))
            except FileNotFoundError:pass
    require(not members and process.returncode==0, 'owned Riley session did not close gracefully')
    return {'returncode':process.returncode,'remaining_owned_pids':members,'zombie_pids':zombies,
            'cleanup_verified':True,'log':evidence(log)}


def run_case(context, prepared, directory, name, mode, watchdog):
    core,session,runtime=context['core'],context['session'],context['runtime']
    directory.mkdir(mode=0o700)
    parent_lane=context['parent']['lanes'][name];bundle=context['bundles'][name]
    lane=dict(parent_lane,argv=list(parent_lane['argv']),cwd=bundle['build']['source_root'])
    lane['argv'][0]=bundle['build']['binary']
    argv=core.lane_argv(lane,{'offered_concurrency':1,'vllm_token_budget':128},directory)
    env=diagnostic_environment(core,context['parent'],parent_lane,runtime,mode)
    core.shared.check_port(lane['port'])
    start=core.cooldown(context['run_plan'],env,context['binding'],directory/'cooldown.json')
    watchdog.check()
    write(directory/'launch.json',{'argv':argv,'cwd':lane['cwd'],'environment':env,'lane':name,'mode':mode,'fresh_process':True,
                                   'workload':WORKLOAD,'start_gpu':start,'diagnostic_only':True,'performance_claim_eligible':False})
    process=None;cleanup=None;failure=None;monitor=None
    stop=threading.Event();errors=[];samples=[];responses=[]
    log=directory/'server.log'
    try:
        with log.open('x') as stream:
            process=subprocess.Popen(argv,cwd=lane['cwd'],env=env,stdout=stream,stderr=stream,start_new_session=True)
            write(directory/'process.json',{'pid':process.pid,'session_id':os.getsid(process.pid)})
            core.shared.wait_ready(process,lane['port'],context['run_plan']['startup_timeout_seconds'])
            write(directory/'driver-maps-before.json',core.compute_maps(context['run_plan'],env,context['binding'],session,runtime,process))
            with core.tokens.TokenHttpClient() as client:
                def inspect():
                    while not stop.wait(5):
                        try:
                            watchdog.check()
                            require(process.poll() is None,'diagnostic server exited')
                            sample=core.gpu_snapshot(context['run_plan'],env,context['binding']);samples.append(sample)
                            require(all(core.owned(row['pid'],process.pid) for row in sample['compute_processes']),'foreign CUDA compute contaminated diagnostic')
                        except Exception as error:
                            errors.append(str(error));client.abort_pending(str(error));return
                monitor=threading.Thread(target=inspect,daemon=True);monitor.start()
                payload={'model':context['reference'].model,'prompt':context['request']['prompt'],'max_tokens':32,'temperature':0,'top_p':1}
                for phase,streaming,count in [('warmup-nonstream',False,5),('warmup-stream',True,5),('retained',True,6)]:
                    watchdog.check()
                    def one():
                        require(not errors,'diagnostic monitor failed')
                        return client.request(lane['port'],payload,context['reference'],streaming=streaming,mode='strict',
                                              timeout_seconds=context['run_plan']['request_timeout_seconds'])
                    rows,accounting=core.tokens.run_phase(client,one,concurrency=1,count=count,phase=phase)
                    raw=directory/(phase+'.jsonl')
                    with raw.open('x') as output:
                        for row in rows:output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    write(directory/(phase+'-accounting.json'),accounting)
                    core.phase_check(rows,accounting,count,1)
                    require(not errors and process.poll() is None,'diagnostic contaminated/server exited')
                    responses.append({'phase':phase,'raw':evidence(raw),'accounting':evidence(directory/(phase+'-accounting.json'))})
                write(directory/'driver-maps-after.json',core.compute_maps(context['run_plan'],env,context['binding'],session,runtime,process))
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=35)
            if monitor.is_alive():errors.append('diagnostic GPU monitor did not stop')
        if process is not None:
            try:cleanup=cleanup_owned(core,process,log)
            except BaseException as error:
                failure={'primary':failure,'cleanup':{'type':type(error).__name__,'message':str(error)},
                         'owned_cleanup_verified':False}
        write(directory/'process-exit.json',{'cleanup':cleanup,'failure':failure})
        write(directory/'gpu-monitor.json',{'samples':samples,'errors':errors})
    require(failure is None and not errors and cleanup is not None and len(responses)==3,'diagnostic process incomplete: '+str(failure or errors))
    report,validation=extract_and_summarize(log,directory,bundle,mode)
    result={'lane':name,'mode':mode,'completed':True,'responses':responses,'cleanup':cleanup,'native':validation,
            'binary':{'path':bundle['build']['binary'],'sha256':bundle['build']['binary_sha256']},
            'performance_claim_eligible':False}
    write(directory/'completion.json',result)
    return result,report


def compare(reports):
    def metric(label,phase):return reports[label]['whole_graph']['groups'][phase]['metrics']['cuda_graph_span_ns']['median']
    def ratio(a,b):return b/a if a and b is not None else None
    pairs=[]
    for order in ('ab','ba'):
        a,b=(reports['combined-'+order+'-'+name] for name in ('baseline','candidate'))
        left,right=(r['operator_groups'][0]['sum_across_selected_layers_ns']['median'] for r in (a,b))
        pairs.append({'order':order.upper(),'baseline_combined_median_ns':left,'candidate_combined_median_ns':right,
                      'candidate_over_baseline':ratio(left,right)})
    drift=[]
    for name in LANES:
        for phase in ('prefill','decode'):
            before,after=(metric('off-'+when+'-'+name,phase) for when in ('before','after'))
            selected=[metric('combined-'+order+'-'+name,phase) for order in ('ab','ba')]
            drift.append({'lane':name,'phase':phase,'off_before_median_ns':before,'off_after_median_ns':after,
                          'off_after_over_before':ratio(before,after),'selected_median_ns':selected,
                          'selected_over_off_before':[ratio(before,x) for x in selected],
                          'selected_over_off_after':[ratio(after,x) for x in selected]})
    return {'paired_combined_intervals':pairs,'whole_graph_drift_and_perturbation':drift,
            'scope':'Combined region is timed directly, then layers are summed per replay. Event perturbation remains; two pairs are diagnostic evidence only.',
            'performance_claim_eligible':False,'candidate_acceptance':None}


def measure(context,prepared,output):
    core,session=context['core'],context['session'];completed=[];reports={};failure=None;restored=None;stopped=False
    try:
        unchanged(context,prepared)
        require(session.validate_runtime()==prepared['runtime'] and session.check(prepared['runtime'])==prepared['sessions'],'pre-stop session/runtime changed')
        for name in LANES:core.shared.check_port(context['parent']['lanes'][name]['port'])
        stopped=True
        core.session_action(context['run_plan'],'stop',output)
        require(session.bound_runtime()==prepared['runtime'],'stopped runtime changed')
        with core.Watchdog(context['run_plan'],output) as watchdog:
            for label,name,mode in SCHEDULE:
                unchanged(context,prepared);watchdog.check()
                result,report=run_case(context,prepared,output/label,name,mode,watchdog)
                unchanged(context,prepared);watchdog.check()
                completed.append({'label':label,'completion':evidence(output/label/'completion.json')})
                reports[label]=report
        unchanged(context,prepared)
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
    finally:
        if stopped and session.SNAPSHOT.exists():
            try:
                core.session_action(context['run_plan'],'restore',output)
                restored=core.verify_restoration(session,prepared)
            except BaseException as error:
                failure={'primary':failure,'restore':{'type':type(error).__name__,'message':str(error)},'independent_watchdog_remains_responsible':True}
        write(output/'finalization.json',{'completed_processes':completed,'failure':failure,'restoration':restored,'performance_claim_eligible':False})
    require(failure is None and restored is not None and len(completed)==8,'diagnostic incomplete; see finalization.json')
    unchanged(context,prepared)
    comparison=compare(reports)
    write(output/'comparison.json',comparison)
    result={'schema':SCHEMA,'completed':True,'preparation':evidence(output/'preparation.json'),
            'finalization':evidence(output/'finalization.json'),'processes':completed,'comparison':evidence(output/'comparison.json'),
            'restoration':restored,'workload':WORKLOAD,'performance_claim_eligible':False,'candidate_acceptance':None}
    write(output/'completion.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--measure',action='store_true')
    args=parser.parse_args(argv)
    output=args.output.resolve()
    require(not output.exists(),'output exists; evidence will not be overwritten')
    context,prepared=prepare(args.root)
    output.mkdir(mode=0o700)
    write(output/'preparation.json',prepared)
    if args.measure:
        with context['core'].termination_handlers():measure(context,prepared,output)
    else:print(json.dumps({'prepared':True,'diagnostic_started':False,'preparation':evidence(output/'preparation.json')}))


if __name__=='__main__':main()
