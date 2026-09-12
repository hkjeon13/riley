"""Pin the verified candidate and prepare commands; never execute performance trials."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from tokenizers import Tokenizer
r=Path('/tmp/riley-g04-vllm-profile-260911');old=Path('/tmp/riley-g04-followup-260911');diag=Path('/tmp/riley-g04-native-profile-260911')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(r/name).write_text(json.dumps(obj,indent=2)+'\n')
manifest=json.loads((r/'source-manifest.json').read_text());candidate=json.loads((old/'candidate.json').read_text());candidate.update(source_root=manifest['source_root'],source_commit=manifest['source_commit'],source_clean=True,numerical_profile='vllm-smol-p128-v1')
for name in candidate['binaries']:
 p=Path('/tmp/riley-g04-vllm-profile-target/release')/name;candidate['binaries'][name]={'path':str(p),'sha256':sha(p)}
save('candidate.json',candidate)
for name in ['environment.json','dependencies.lock','request.json','measure.py','vllm_lane.py','build_plan.py']:(r/name).write_bytes((old/name).read_bytes())
env=json.loads((r/'environment.json').read_text());tokens=json.loads(Path('/tmp/riley-g04-serving-trace-260911/invariance.json').read_text())['before'];text=Tokenizer.from_file(env['riley_checkpoint']+'/tokenizer.json').decode(tokens,skip_special_tokens=True)
save('parity.json',{'numerical_profile':'vllm-smol-p128-v1','riley_vs_vllm_exact_tokens':True,'riley_graph_equals_old_riley_eager_all_logits':False,'first_different_output_index_zero_based':None,'riley_output_tokens':tokens,'vllm_output_tokens':tokens,'riley_output_text':text,'vllm_output_text':text,'performance_trials':0})
for name in ['clean-cpu-final.log','clean-gpu-test.log','clean-cli-test.log','clean-profile-cli-test.log','clean-profile-checker-test.log','clean-release-final.log','native-all-final-validation.json','prefill-gemm-shapes.json','gemv-all-shapes.json']:(r/name).write_bytes((diag/name).read_bytes())
native=json.loads((r/'native-all-final-validation.json').read_text());assert native['unequal']==0 and native['tokens_exact'] and not native['external_kv_input']
http=json.loads((r/'http-validation.json').read_text());assert http['binary_sha256']==candidate['binaries']['riley']['sha256'];assert len(http['results'])==2
assert '2 passed; 0 failed' in (r/'clean-gpu-test.log').read_text();assert 'vllm_tokens_exact=true' in (r/'clean-gpu-test.log').read_text()
save('verification.json',{'status':'correctness-qualified','scope':'SmolLM2-135M c1/p128/o32 Hello repeated 128 times; SM89/CUDA13.0/cuBLASLt13.1.1; vLLM0.27.1 compiled FlashAttention2','source_commit':candidate['source_commit'],'binaries':candidate['binaries'],'native_prefill_and_decode':True,'external_kv_or_context_input':False,'tensor_comparisons':19080,'unequal_tensor_elements':0,'generated_tokens_exact':32,'new_profile_replays':491,'requests':4,'cancelled_prefill_tokens':23,'cancelled_generated_tokens':23,'existing_profile_regression_passed':True,'http_cpu_gpu_streaming_disconnect_reuse_passed':True,'tests':{'cuda_cpu':84,'runtime_cpu':260,'server_cli':22,'profile_cli':10,'profile_checkers':25,'cuda_owned_profiles':2},'old_riley_eager_bitwise_equivalence_claim':False,'performance_trials':0,'evidence_sha256':{name:sha(r/name) for name in ['clean-cpu-final.log','clean-gpu-test.log','clean-cli-test.log','clean-profile-cli-test.log','clean-profile-checker-test.log','native-all-final-validation.json','http-validation.json']}})
subprocess.run([sys.executable,str(r/'build_plan.py')],check=True)
plan=json.loads((r/'measurement-plan.json').read_text());(r/'build_plan.py').unlink();argv=plan['engine_lanes']['riley']['argv']
for flag,value in [('--runtime-flag-name','graph_numerics'),('--runtime-flag-value','vllm-smol-p128-v1'),('--semantic-class','VLLM_REFERENCE'),('--correctness-gate-id','g04-vllm-smol-p128-v1'),('--implementation-id','g04-vllm-smol-p128-v1')]:argv[argv.index(flag)+1]=value
plan['http_lanes']['riley']['argv']+=['--graph-numerics','vllm-smol-p128-v1']
plan['qualification_blockers']=[];plan['comparison_status']='correctness qualified for diagnostic SmolLM2 c1/p128/o32; performance unmeasured; exclusive GPU preflight required';plan['numerical_profile']='vllm-smol-p128-v1';plan['riley_result_schema']='riley.vllm-profile-run.v1'
# Prepare-only retains the same model, source and runtime bindings as the eventual lane.
args=[x.format(index='1',output=str(r),started_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')) for x in argv]
args[args.index('--output')+1]=str(r/'native-preparation.json');args+=['--prepare-only','true']
runenv=os.environ.copy();runenv.update(plan['engine_lanes']['riley']['env'])
with (r/'native-preparation.log').open('w') as log:subprocess.run(args,env=runenv,cwd=manifest['source_root'],stdout=log,stderr=log,check=True)
prepared=json.loads((r/'native-preparation.json').read_text());assert prepared['prepared'] and prepared['performance_trials']==0 and prepared['cleanup_allocations_zero']
binding={key:prepared[key] for key in ['source','environment','workload']};binding.update(input_token_ids=[19556]*128,generated_token_ids=tokens);save('native-binding.json',binding)
# Live preflight stays authoritative; unchanged external GPU work is never stopped.
preenv=os.environ.copy();preenv['RILEY_PREFLIGHT_OUTPUT_ROOT']=str(r/'preflight');preenv['RILEY_PREFLIGHT_ENVIRONMENT_ID']=plan['preflight_environment_id']
with (r/'preflight.stdout').open('w') as out,(r/'preflight.stderr').open('w') as err:status=subprocess.run(['bash',str(Path(manifest['source_root'])/'benchmarks/scripts/preflight.sh')],env=preenv,cwd=manifest['source_root'],stdout=out,stderr=err).returncode
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
save('live-preflight.json',{'exit_code':status,'stderr':(r/'preflight.stderr').read_text(),'gpu_processes':processes,'performance_trials':0})
plan['measurement_environment_blockers']=[] if status==0 and len(processes.strip().splitlines())<=1 else ['exclusive GPU preflight not satisfied; see live-preflight.json']
plan['live_preflight_last_failure']=(r/'preflight.stderr').read_text().strip()
# Validate the distinct result after each Riley measured engine-only run.
p=r/'measure.py';s=p.read_text();needle="                    subprocess.run(argv, cwd=plan['source_root'], env=env, stdout=out, stderr=err, check=True)";assert needle in s
s=s.replace(needle,needle+"\n                if role == 'riley':\n                    with (directory/'reference-check.json').open('x') as checked:\n                        subprocess.run([plan['reference_checker_python'],str(Path(plan['source_root'])/'benchmarks/scripts/check_vllm_profile_run.py'),str(directory/'native-profile.json'),'--binding',str(ROOT/'native-binding.json')],stdout=checked,check=True)")
s=s.replace("'qualification_blockers':plan['qualification_blockers'], 'next_gate'","'qualification_blockers':plan['qualification_blockers'], 'measurement_environment_blockers':plan['measurement_environment_blockers'], 'next_gate'");p.write_text(s)
plan['reference_checker_python']=env['executable'];plan['prepare_only_argv']=args
for name in ['measure.py','native-binding.json','native-preparation.json','verification.json','candidate.json','source-manifest.json','live-preflight.json']:plan['immutable_files'][str(r/name)]=sha(r/name)
save('measurement-plan.json',plan)
subprocess.run([sys.executable,str(r/'measure.py')],check=True)
print(json.dumps({'correctness_qualified':True,'native_prepare_only_passed':True,'measurement_environment_blockers':plan['measurement_environment_blockers'],'performance_trials':0}))
