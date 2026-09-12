import copy,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912');BASE=Path('/tmp/riley-g04-vllm-profile-260911')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 with p.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
stage=json.loads((ROOT/'stage-correctness.json').read_text());http=json.loads((ROOT/'candidate-http-correctness/http-validation.json').read_text());assert stage['passed'] and stage['vllm_reference_tokens_exact'] and all(x['post_disconnect_reuse_exact'] for x in http['results'])
source=ROOT/'candidate-source';commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
assert commit==stage['candidate_source_commit'];binaries={str(ROOT/'candidate-target/release'/x):sha(ROOT/'candidate-target/release'/x) for x in ['riley','riley-profile']};assert binaries==stage['candidate_binaries'];assert http['binary_sha256']==binaries[str(ROOT/'candidate-target/release/riley')]
proof=ROOT/'candidate-qualification.json';write(proof,{'schema_version':'riley.stage-split-qualification.v1','passed':True,'source_commit':commit,'implementation_id':'g05-stage-split-v1','numerical_profile':'vllm-smol-p128-v1','scope':'SmolLM2-135M c1 P128 O32; Hello exact vLLM reference, diverse prompts old-candidate byte equivalence only','internal_stage_tests':{'path':str(ROOT/'stage-correctness.json'),'sha256':sha(ROOT/'stage-correctness.json')},'release_http':{'path':str(ROOT/'candidate-http-correctness/http-validation.json'),'sha256':sha(ROOT/'candidate-http-correctness/http-validation.json')},'binaries':binaries,'canonical_e0_claim':False,'performance_measured':False})
binding=json.loads((BASE/'native-binding.json').read_text());binding['source'].update(git_commit=commit,git_dirty=False,implementation_id='g05-stage-split-v1',correctness_gate_id='g05-stage-split-v1',correctness_report_sha256=sha(proof),executable_sha256=binaries[str(ROOT/'candidate-target/release/riley-profile')]);binding_path=ROOT/'candidate-binding.json';write(binding_path,binding)
plan=json.loads((BASE/'measurement-plan.json').read_text());plan.update(source_root=str(source),source_commit=commit,measurement_started=False,comparison_status='Stage split qualified; performance pending',qualification_blockers=[],measurement_environment_blockers=[])
plan['http_lanes']['riley']['argv'][0]=str(ROOT/'candidate-target/release/riley')
argv=plan['engine_lanes']['riley']['argv'];argv[0]=str(ROOT/'candidate-target/release/riley-profile')
for key,value in {'--git-commit':commit,'--implementation-id':'g05-stage-split-v1','--correctness-gate-id':'g05-stage-split-v1','--correctness-report-sha256':sha(proof),'--executable-sha256':binding['source']['executable_sha256']}.items():argv[argv.index(key)+1]=value
plan['prepare_only_argv']=[]
immutable={p:h for p,h in plan['immutable_files'].items() if '/data/' in p}
for p in [BASE/'request.json',BASE/'native-binding.json',BASE/'verification.json',BASE/'prompts.jsonl',proof,binding_path,ROOT/'stage-correctness.json',ROOT/'candidate-http-correctness/http-validation.json']:
 immutable[str(p)]=sha(p)
immutable.update(binaries);plan['immutable_files']=immutable
write(ROOT/'candidate-plan.json',plan)
print(json.dumps({'candidate_source_commit':commit,'plan':str(ROOT/'candidate-plan.json'),'correctness_report_sha256':sha(proof)}))
