import pathlib,json,hashlib,subprocess,time,sys,tarfile
b=pathlib.Path('/data/riley-serving-261007');o=b/'kernel-batch11-preparation-attempt01';code=b/'kernel-batch11-controller'
sys.path.insert(0,str(code))
from serving_token_client_v2 import TokenResponseParser,TokenReference
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
failure=None;executed=False
try:
 p=pathlib.Path('/proc/2304082')
 while p.exists():
  assert str(o/'server-after-native.py').encode() in (p/'cmdline').read_bytes().split(b'\0'),'server queue PID changed'
  time.sleep(30)
 assert json.loads((o/'server-queue-completion.json').read_text())['failure'] is None
 n=b/'kernel-batch11-primitive-validation-attempt01';s=b/'kernel-batch11-server-validation-attempt01';h=b/'kernel-batch11-http-screen-attempt01'
 assert json.loads((s/'completion.json').read_text())['failure'] is None
 terminal=json.loads((h/'completion.json').read_text())
 assert terminal['failure'] is None and terminal['requests']==60 and len(terminal['records'])==8
 preparation=json.loads((o/'matched96-preparation.json').read_text())
 for path,digest in preparation['immutable_controller_files'].items():assert sha(pathlib.Path(path))==digest
 fixtures=json.loads((code/'fixtures.json').read_text());model=b.parent/'riley-serving-260913-recovery/runtime-assets-20260915/model'
 for path,digest in preparation['screen_fixture_and_model_pins'].items():assert sha(pathlib.Path(path))==digest
 manifest=json.loads((b/'kernel-batch11-source-receipt-attempt01.json').read_text());source=b/'kernel-batch11-source-attempt01'
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==manifest['source_commit']
 for rel,digest in manifest['files'].items():assert sha(source/rel)==digest
 native=json.loads((o/'independent-native-proof.json').read_text())
 receipt=json.loads((s/'binary-receipt.json').read_text());binary=b/'kernel-batch11-target-attempt01/release/riley'
 assert receipt['source_commit']==manifest['source_commit'] and sha(binary)==receipt['sha256']
 cases=[];count=0
 for capacity in [1,8,16,32]:
  for workload in ['fixed','natural']:
   label=f'c{capacity}-{workload}'
   launch=json.loads((h/(label+'-launch.json')).read_text())
   prior=json.loads((b/'kernel-batch10-http-screen-attempt01'/(label+'-launch.json')).read_text())
   argv=launch['argv'];a=list(argv);z=list(prior['argv']);a[0]=z[0]='candidate';a[a.index('--bind')+1]=z[z.index('--bind')+1]='port'
   assert a==z and launch['env']==prior['env'] and launch['binary_sha256']==receipt['sha256']
   rows=json.loads((h/(label+'-rows.json')).read_text());data=fixtures[workload]
   assert len(rows)==len(data['corpus'])
   for row,c,response in zip(rows,data['corpus'],data['responses']):
    choice=response['choices'][0];ref=TokenReference('g04-smol',tuple(choice.get('prompt_token_ids',c.get('prompt_token_ids',[]))),tuple(choice['token_ids']),choice['text'],choice['finish_reason'])
    parser=TokenResponseParser(ref,streaming=True,started_ns=row['started_ns'],mode='observe')
    for frame in row['frames']:parser.feed_sse(frame['data'].encode(),frame['arrived_ns'])
    replay=parser.finish()
    for key,value in replay.items():assert row[key]==value,'raw replay field mismatch '+label+' '+key
    assert replay['protocol_valid'] and replay['reference_match']
    assert row['status']=='success' and row['http_status']==200 and row['transport_complete'] and row['owned_connection_closed'] and not row['cleanup_errors']
    assert row['request']=={'model':'g04-smol','prompt':c['prompt'],'max_tokens':c['max_tokens'],'temperature':0}
    count+=1
   exit=json.loads((h/(label+'-exit.json')).read_text());assert not exit['gpu_compute_after']
   cases.append({'case':label,'replayed':len(rows)})
 assert count==60 and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 proof={'source_commit':manifest['source_commit'],'source_files_bound_to_true_Git_blobs':len(manifest['files']),'native_exact_cases':3161,'fusion_exact_cases':272,'prefill_gate_exact_cases':120,'decode_value_exact_cases':360,'mapped_value_exact_cases':1593,'native_binary_sha256':native['native_binary_sha256'],'HTTP_raw_requests_replayed':count,'HTTP_cases':cases,'server_binary_sha256':receipt['sha256'],'serving_performance':'미실행','adopted':False,'goal_achieved':False,'time_ns':time.time_ns()}
 proof_path=b/'kernel-batch11-native-http-independent-verification.json';assert not proof_path.exists();proof_path.write_text(json.dumps(proof,indent=2));(code/proof_path.name).write_bytes(proof_path.read_bytes())
 roots=[n,s,h];before={str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file()};archive=o/'native-http-evidence.tar.gz'
 with archive.open('xb') as f:
  with tarfile.open(fileobj=f,mode='w:gz') as t:
   for root in roots:t.add(root,arcname=root.name)
 assert before=={str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file()}
 (o/'native-http-evidence-receipt.json').write_text(json.dumps({'archive_sha256':sha(archive),'files_before':before,'files_after':before,'originals_preserved':True},indent=2))
 plan=json.loads((code/'batch-plan.json').read_text());plan['candidate_binary_sha256']=receipt['sha256'];plan['independent_correctness_sha256']=sha(proof_path);plan['execution_state']='predeclared before serving launch; independent native3161 and HTTP60 passed'
 phase=b/'kernel-batch10-profile32-independent-analysis-attempt01/kernel-batch10-cpu-capture-all8-profile-phase-analysis-attempt01/summary.json'
 assert json.loads(phase.read_text())['profiles_analyzed']==32
 plan['measured_motivation']={'profiles':str(phase),'sha256':sha(phase),'scope':'all32 preserved/verified profiles; restoration of source-bound V52 AV loads and warp synchronization','limits':'diagnostic only; no serving performance inferred'}
 plan['predeclared_time_ns']=time.time_ns();(code/'batch-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 assert sha(binary)==receipt['sha256']
 argv=[str(b/'vllm0271-venv/bin/python'),str(code/'run.py'),'--attempt','1','--mode','quiet']
 executed=True
 with (o/'matched96-controller.log').open('x') as log:
  process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (o/'matched96-dispatch.json').write_text(json.dumps({'pid':process.pid,'argv':argv,'plan_sha256':sha(code/'batch-plan.json'),'controller_sha256':sha(code/'run.py'),'independent_correctness_sha256':sha(proof_path),'time_ns':time.time_ns()},indent=2))
  assert process.wait()==0,'matched96 serving failed; preserve all samples'
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(o/'matched96-queue-completion.json').write_text(json.dumps({'failure':failure,'executed':executed,'adopted':False,'goal_achieved':False},indent=2))
