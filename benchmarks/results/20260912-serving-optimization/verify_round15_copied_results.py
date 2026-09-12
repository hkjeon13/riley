"""Reparse copied Round15 native logs and strict HTTP responses, without GPU work."""
from pathlib import Path
import hashlib, importlib.util, json, sys

HERE = Path(__file__).resolve().parent
RAW = HERE/'raw'
CAMPAIGN = RAW/'paired-rope-attention-round15'
SCRIPTS = HERE.parent.parent/'scripts'
sys.path.insert(0, str(SCRIPTS))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(item):
    path = RAW/Path(item['path']).relative_to('/tmp/riley-opt-260912')
    assert sha(path)==item['sha256'],str(path)
    return path

runner=load('paired_diagnostic',HERE/'run_paired_rope_attention_profile.py')
assert sha(Path(runner.__file__))=='c2260bfed0b5a904bd1972b52911c8a760742f006ac3eab0d49bf82e32f5c90c'
import analyze_serving_token_optimization_v2 as analyzer
import serving_token_client_v2 as tokens
core,_=analyzer.private_core()
parent=read(RAW/'batch7-http-plan.json')
base=HERE.parent/'20260911-g04-vllm-profile'
binding=read(base/'native-binding.json');request=read(base/'request.json')
reference=tokens.TokenReference('g04-smol',tuple(binding['input_token_ids']),tuple(binding['generated_token_ids']),parent['http_lanes']['riley']['expected_output_text'])
completion=read(CAMPAIGN/'completion.json');final=read(ref(completion['finalization']));prep=read(ref(completion['preparation']))
assert completion['completed'] and not completion['performance_claim_eligible'] and completion['candidate_acceptance'] is None
assert final['failure'] is None and len(final['completed_processes'])==8 and completion['processes']==final['completed_processes']
assert [p['label'] for p in completion['processes']]==[x[0] for x in runner.SCHEDULE]
restored=read(ref(completion['restoration']));assert completion['restoration']==final['restoration']
assert restored['alive_and_listening'] and restored['commands_and_gui_environment_match'] and restored['all_relaunched_processes_have_pinned_vendor_maps'] and len(restored['processes'])==3
assert read(RAW/'blender-round15/runtime.json')==prep['runtime']
summation_deltas=[]
def equal_report(actual, saved, path=''):
    if isinstance(actual,dict) and isinstance(saved,dict):
        assert set(actual)==set(saved),path
        for key in actual:equal_report(actual[key],saved[key],path+'.'+key)
    elif isinstance(actual,list) and isinstance(saved,list):
        assert len(actual)==len(saved),path
        for i,(a,b) in enumerate(zip(actual,saved)):equal_report(a,b,path+f'[{i}]')
    elif actual!=saved:
        # Python3.13 uses compensated float sum; this analyzer host is3.10.
        # Only the unused derived sum may differ below one millionth ns.
        # Raw observations, counts, every percentile and comparison stay exact.
        assert path.endswith('.sum') and type(actual) is float and type(saved) is float and abs(actual-saved)<=1e-6,path
        summation_deltas.append({'path':path,'local_sum_ns':actual,'saved_sum_ns':saved,'absolute_difference_ns':abs(actual-saved)})
reports={};proofs=[];requests=0
for (label,lane,mode),item in zip(runner.SCHEDULE,completion['processes']):
    directory=CAMPAIGN/label;process=read(ref(item['completion']));validation=read(directory/'native-validation.json')
    for key in ('raw_log','retained_log','summary'):ref(validation[key])
    records=runner.native_records(directory/'server.log')
    full=runner.validate_native(records,mode)
    retained=[r for r in records if 'replay_id' not in r or r['replay_id']>=321]
    assert runner.validate_native(retained,mode,True)==validation['retained'] and full==validation['full']
    assert ''.join(json.dumps(r,allow_nan=False)+'\n' for r in retained)==(directory/'retained-native.log').read_text()
    filename='profile_decode_rope_attention_baseline.py' if lane=='baseline' else 'profile_decode_operators_batch8.py'
    tool=load('profiler_'+label.replace('-','_'),SCRIPTS/filename)
    report=tool.summarize(directory/'retained-native.log')
    equal_report(report,read(directory/'native-summary.json'),label)
    for capture in report['captures']:
        assert capture['operator']==mode and capture['tool_sha256']==sha(Path(tool.__file__))
    reports[label]=report
    cleaned=read(directory/'process-exit.json');assert cleaned['failure'] is None
    assert cleaned['cleanup']['cleanup_verified'] and not cleaned['cleanup']['remaining_owned_pids']
    assert cleaned['cleanup']==process['cleanup']
    for phase,streaming,count in [('warmup-nonstream',False,5),('warmup-stream',True,5),('retained',True,6)]:
        path=directory/(phase+'.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows)==count
        for row in rows:core.reparse_row(row,reference,request,streaming,phase,120)
        requests+=len(rows)
    proofs.append({'label':label,'server_log_sha256':sha(directory/'server.log'),'native_validation_sha256':sha(directory/'native-validation.json'),'completion_sha256':sha(directory/'completion.json'),'native_replays':full['replays'],'strict_http_responses':16})
assert requests==128
comparison=runner.compare(reports);assert comparison==read(ref(completion['comparison']))
result={'completed':True,'scope':'offline copied-log reparse; no fresh GPU or live-process claim','raw_native_replays':4096,'retained_native_replays':1536,'strict_http_responses':requests,'python_float_sum_differences':summation_deltas,'percentiles_and_comparison_exact':True,'processes':proofs,'comparison':comparison,'completion_sha256':sha(CAMPAIGN/'completion.json'),'restoration_sha256':completion['restoration']['sha256'],'performance_acceptance':False}
with (HERE/'round15-copied-results-verification.json').open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
print(json.dumps({'completed':True,'raw_native_replays':4096,'strict_http_responses':requests}))
