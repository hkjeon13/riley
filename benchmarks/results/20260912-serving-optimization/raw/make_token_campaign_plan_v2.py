"""Prepare a new concrete three-way token measurement manifest; no GPU or session mutation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT=Path('/tmp/riley-opt-260912')
BASE=Path('/tmp/riley-g04-vllm-profile-260911')
MODEL=Path('/data/riley-benchmark/20260827T051948Z-d7ad713a/model')
VENV=Path('/data/riley-vllm-interim.CfrT9T/venv')
JSON_SEEN=set()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def evidence(path):
    p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':sha(p)}


def replace_option(argv,key,value):
    result=list(argv);index=result.index(key);result[index+1]=value;return result


def nested_pins(value,pins):
    if isinstance(value,dict):
        if value.get('schema')=='riley.http-token-local-source.v1':
            local_receipt=ROOT/'http-token-source-after-v2.json'
            assert sha(local_receipt)=='d8a03a06c3bf7f930d044cf082574fb9443ed8a3023b8df7dce495a525610f53'
            assert value==read(local_receipt)
            pins[str(local_receipt)]=sha(local_receipt)
            # Local historical test paths are provenance metadata. Exact source
            # identity and fresh GPU/HTTP qualification are validated separately.
            return
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            p=Path(value['path']);assert p.is_absolute() and sha(p)==value['sha256'];pins[str(p)]=value['sha256']
            if p.suffix=='.json' and str(p) not in JSON_SEEN:
                JSON_SEEN.add(str(p));nested_pins(read(p),pins)
        for key,item in value.items():
            if isinstance(key,str) and key.startswith('/') and isinstance(item,str) and len(item)==64:
                assert sha(key)==item;pins[key]=item
                if Path(key).suffix=='.json' and key not in JSON_SEEN:
                    JSON_SEEN.add(key);nested_pins(read(key),pins)
            else:nested_pins(item,pins)
    elif isinstance(value,list):
        for item in value:nested_pins(item,pins)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-proof',type=Path,required=True)
    parser.add_argument('--baseline-validator',type=Path,required=True)
    parser.add_argument('--candidate-proof',type=Path)
    parser.add_argument('--candidate-validator',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=5)
    parser.add_argument('--retained',type=int,default=1000)
    args=parser.parse_args()
    assert bool(args.candidate_proof)==bool(args.candidate_validator)
    assert not args.output.exists()
    parent=read(ROOT/'batch7-http-plan.json')
    pins=dict(parent['immutable_files'])
    assert all(sha(path)==digest for path,digest in pins.items())
    # Freeze every local imported helper, including the static source review.
    for p in sorted(ROOT.glob('*.py')):pins[str(p)]=sha(p)
    for name in ['TOKEN_BENCHMARK_CONTRACT.md','HTTP_TOKEN_PROOF_SCOPE.md']:
        pins[str(ROOT/name)]=sha(ROOT/name)
    reference={key:evidence(path) for key,path in {
        'parent_c1_plan':ROOT/'batch7-http-plan.json','request':BASE/'request.json',
        'binding':BASE/'native-binding.json','verification':BASE/'verification.json'}.items()}
    for ref in reference.values():pins[ref['path']]=ref['sha256']
    proofs=[('baseline',args.baseline_proof,args.baseline_validator,'http-token',ROOT/'http-token-gpu-tests.json',19431)]
    if args.candidate_proof:proofs.append(('candidate',args.candidate_proof,args.candidate_validator,'batch8',ROOT/'batch8-model-tests.json',19432))
    lanes={}
    for name,proof_path,validator_path,prefix,model_path,port in proofs:
        proof=read(proof_path);assert proof['completed'] and proof['gpu_tests_executed']
        model_proof=read(model_path);build=read(ROOT/(prefix+'-build.json'))
        assert proof['source_commit']==model_proof['source_commit']==build['source_commit']
        preparation=read(proof['per_sampler'][1]['preparation']['path'])
        argv=replace_option(preparation['argv'],'--bind','127.0.0.1:{port}')
        argv=replace_option(argv,'--max-waiting-requests','64')
        deps=[evidence(p) for p in sorted(ROOT.glob('*.py'))]
        lane={'kind':'riley','argv':argv,'env':{'CUDA_VISIBLE_DEVICES':'0',
              'LD_LIBRARY_PATH':str(ROOT/'driver580173-gui-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':'+str(ROOT/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib'},'cwd':build['source_root'],
              'port':port,'model_path':str(MODEL),'model_id':'g04-smol','model_files':proof['model_files'],
              'build':evidence(ROOT/(prefix+'-build.json')),'model_proof':evidence(model_path),
              'raw_token_proof':evidence(proof_path),'raw_validator':evidence(validator_path),'validator_dependencies':deps}
        nested_pins(lane,pins);nested_pins(proof,pins);nested_pins(model_proof,pins);nested_pins(build,pins)
        if prefix=='batch8':
            lane['model_validator']=evidence(ROOT/'qualify_batch8.py')
            lane['model_qualification']=evidence(ROOT/'batch8-qualification.json')
            for p in [ROOT/'batch8-qualification.json',ROOT/'batch8-fusion-probe/receipt.json',ROOT/'batch8-source-overlay.json',ROOT/'http-token-source-after-v2.json']:
                pins[str(p)]=sha(p);nested_pins(read(p),pins)
        lanes[name]=lane
    original=parent['http_lanes']['vllm'];vargv=original['argv']
    for key,value in [('--port','{port}'),('--max-num-seqs','{concurrency}'),('--max-num-batched-tokens','{vllm_budget}')]:vargv=replace_option(vargv,key,value)
    vmodel=Path(vargv[2])
    settings=[{'id':f'c{c}-b{c*128}','offered_concurrency':c,'vllm_token_budget':128*c} for c in (1,2)]
    diagnostics={setting['id']:evidence(ROOT/f"vllm-output-matrix-diagnostic/c{setting['offered_concurrency']}-vllm-budget{setting['vllm_token_budget']}/completion.json") for setting in settings}
    for ref in diagnostics.values():pins[ref['path']]=ref['sha256'];nested_pins(read(ref['path']),pins)
    lanes['vllm']={'kind':'vllm','argv':vargv,'env':dict(original['env'],CUDA_VISIBLE_DEVICES='0'),
                  'cwd':str(ROOT),'port':19433,'model_path':str(vmodel),'model_id':'g04-smol',
                  'model_files':{str(vmodel/name):sha(vmodel/name) for name in ('model.safetensors','tokenizer.json')},
                  'output_diagnostics':diagnostics}
    # Pin the entire installed vLLM implementation and model JSON metadata.
    distribution=importlib.metadata.distribution('vllm')
    assert distribution.version=='0.27.1'
    package_root=Path(distribution.locate_file('vllm')).resolve(strict=True)
    runtime_files={str(p):sha(p) for p in package_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    runtime_files[str(VENV/'bin/python')]=sha(VENV/'bin/python')
    lanes['vllm'].update(package_root=str(package_root),runtime_files=runtime_files)
    pins.update(runtime_files)
    for model_dir in (MODEL,vmodel):
        for p in model_dir.rglob('*.json'):pins[str(p)]=sha(p)
    selected=('METADATA','RECORD','WHEEL','vllm/__init__.py','vllm/engine/arg_utils.py',
              'vllm/v1/worker/gpu_model_runner.py','vllm/config/compilation.py')
    for member in distribution.files:
        name=str(member)
        if any(name.endswith(suffix) for suffix in selected) or (name.startswith('vllm/') and name.endswith('.so')):
            p=Path(distribution.locate_file(member)).absolute();pins[str(p)]=sha(p)
    smi=ROOT/'driver580173-runtime-20260901/extracted/usr/bin/nvidia-smi'
    session={'helper':evidence(ROOT/'remote_session_round13_v2.py'),'python':evidence('/usr/bin/python3'),
             'root':str(ROOT/'blender-round13')}
    nested_pins(session,pins);pins[str(smi)]=sha(smi)
    for p in [VENV/'bin/python',Path(__file__).resolve()]:pins[str(p)]=sha(p)
    comparisons=[{'id':'baseline-vllm','left':'vllm','right':'baseline'}]
    if args.candidate_proof:comparisons.extend([{'id':'candidate-baseline','left':'baseline','right':'candidate'},
                                               {'id':'candidate-vllm','left':'vllm','right':'candidate'}])
    plan={'schema_version':'riley.serving-token-plan.v1','immutable_files':pins,
          'contract':evidence(ROOT/'TOKEN_BENCHMARK_CONTRACT.md'),'reference':reference,
          'workload':{'id':'g04-token-c1-c2-batch8-initial-v1','purpose':'initial-measurement' if args.pairs==5 and args.retained>=1000 else 'screening',
                      'pairs':args.pairs,'retained_requests_per_process':args.retained,
                      'warmups_per_worker_per_transport':5,'arrival_policy':'closed-loop-refill'},
          'settings':settings,'comparisons':comparisons,
          'base_environment':{'HOME':'/home/psyche','PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'},'session':session,'nvidia_smi':evidence(smi),
          'startup_timeout_seconds':900,'request_timeout_seconds':120,'cooldown_timeout_seconds':900,'lanes':lanes}
    for lane in lanes.values():nested_pins(lane,pins)
    with args.output.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps({'plan':evidence(args.output),'settings':settings,'comparisons':plan['comparisons'],
                      'processes':len(settings)*len(plan['comparisons'])*args.pairs*2,'retained_requests_per_process':args.retained,
                      'measurement_started':False}))


if __name__=='__main__':main()
