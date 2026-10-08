import pathlib,json,hashlib,struct,heapq,subprocess,tarfile,time
r=pathlib.Path('/data/riley-serving-261007')
p=r/'qwen-http-logit-trace-preparation-attempt01'
o=r/'qwen-http-logit-trace-validation-attempt01'
a=r/'qwen-http-logit-trace-independent-analysis-attempt01'
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def values(raw):
 assert len(raw)==151936*2
 bits=struct.unpack('<151936H',raw)
 assert all((x & 0x7f80)!=0x7f80 for x in bits),'nonfinite raw BF16'
 return [struct.unpack('<f',struct.pack('<I',x<<16))[0] for x in bits]
failure=None
try:
 terminal=json.loads((p/'completion.json').read_text())
 assert terminal['failure'] is None and terminal['executed']
 assert not pathlib.Path('/proc/774205').exists(),'trace controller still exists'
 plan=json.loads((p/'plan.json').read_text())
 manifest=json.loads(pathlib.Path(plan['source_manifest']).read_text())
 source=pathlib.Path(plan['source'])
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==plan['source_commit']
 for rel,digest in manifest['files_SHA256'].items():assert sha(source/rel)==digest
 parent=json.loads((r/'qwen-free-running-source-receipt-attempt01.json').read_text())
 for rel,digest in parent['files'].items():
  if rel!='crates/riley-server/Cargo.toml':assert manifest['files_SHA256'][rel]==digest
 for path,digest in plan['immutable_pins'].items():assert sha(path)==digest
 launch=json.loads((o/'test-launch.json').read_text())
 process=json.loads((o/'test-process.json').read_text())
 assert process['exit']==0 and process['binary_before']==process['binary_after']==sha(launch['argv'][0])
 assert not process['GPU_compute_after']
 assert launch['env']==plan['run_env']
 assert launch['argv'][1:]==[plan['test_filter'],*plan['test_args']]
 result=json.loads((o/'result.json').read_text())
 rows=result['trace']['rows'];own=result['own_token_ids']
 http=json.loads(pathlib.Path(plan['run_env']['RILEY_QWEN_HTTP_OBSERVATION']).read_text())
 hf=json.loads(pathlib.Path(plan['run_env']['RILEY_QWEN_HF_TEACHER_FORCED_ORACLE']).read_text())
 teacher=hf['generation']['teacher_token_ids']
 assert len(rows)==len(own)==len(http['token_ids'])==len(teacher)==128
 assert own==http['token_ids'] and http['prompt_token_ids']==[3409]*2048
 assert result['HTTP_observation_sha256']==manifest['HTTP_observation_SHA256']
 assert len(list((o/'raw').glob('*.bf16le')))==128
 hfpath=pathlib.Path(plan['run_env']['RILEY_QWEN_HF_CACHE_ON_SIDECAR'])
 with hfpath.open('rb') as f:
  header_size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(header_size))
  tensor=header['teacher_forced/logits']
  assert tensor['dtype']=='BF16' and tensor['shape']==[128,151936]
  start,end=tensor['data_offsets'];assert end-start==128*151936*2
  f.seek(8+header_size+start);hfraw=f.read(end-start)
 report=[]
 for step,row in enumerate(rows):
  assert row['step']==step and row['target_logical_length']==2048+step
  name=f'strict-fixed-maximum-step-{step:03}.bf16le'
  assert row['raw_sidecar']==name
  raw=(o/'raw'/name).read_bytes()
  digest=hashlib.sha256(raw).hexdigest();assert digest==row['row_bf16_le_sha256']
  assert hashlib.sha256(raw[:151665*2]).hexdigest()==row['addressable_bf16_le_sha256']
  vals=values(raw)
  selected=min(range(151665),key=lambda i:(-vals[i],i))
  rawbest=min(range(151936),key=lambda i:(-vals[i],i))
  top=heapq.nsmallest(32,range(151665),key=lambda i:(-vals[i],i))
  assert selected==row['selected_token_id']==own[step]
  assert rawbest==row['raw_argmax_token_id']
  assert top==row['top_token_ids'] and [vals[i] for i in top]==row['top_values_bf16_as_f32']
  same=own[:step]==teacher[:step]
  assert same==row['HF_comparison_same_input_prefix']
  reference=hfraw[step*303872:(step+1)*303872]
  refvals=values(reference)
  hfselected=min(range(151665),key=lambda i:(-refvals[i],i))
  assert hfselected==teacher[step]
  equal=raw==reference
  assert equal==row['hf_cache_on']['raw_hash_matches']
  differences=None
  if same:
   differences={'unequal_BF16_elements':sum(x!=y for x,y in zip(struct.unpack('<151936H',raw),struct.unpack('<151936H',reference))),'max_absolute_logit_error':max(abs(x-y) for x,y in zip(vals,refvals))}
  report.append({'step':step,'raw_sha256':digest,'own_selected':selected,'HF_selected':hfselected,'same_HF_input_prefix':same,'raw_exact':equal,'selected_exact':selected==hfselected,'same_input_numerical_difference':differences})
 before={str(x.relative_to(o)):sha(x) for x in o.rglob('*') if x.is_file()}
 archive=a/'evidence.tar.gz'
 with archive.open('xb') as output:
  with tarfile.open(fileobj=output,mode='w:gz') as t:t.add(o,arcname=o.name)
 assert before=={str(x.relative_to(o)):sha(x) for x in o.rglob('*') if x.is_file()}
 proof={'source_commit':plan['source_commit'],'runtime_engine_unchanged':True,'raw_rows_verified':128,'finite_elements_verified':128*151936,'own_chosen128_HTTP_token_parity':True,'HF_correctness_pass':own==teacher,'first_selected_mismatch':next((i for i,(x,y) in enumerate(zip(own,teacher)) if x!=y),None),'same_input_rows':[x for x in report if x['same_HF_input_prefix']],'all_rows':report,'binary_sha256':process['binary_before'],'archive_sha256':sha(archive),'files_sha256':before,'originals_preserved':True,'serving_performance':'미실행','cancellation_and_internal_KV_accounting':'미실행','adopted':False,'goal_achieved':False,'time_ns':time.time_ns()}
 (a/'proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in proof.items() if k not in ['all_rows','files_sha256']},ensure_ascii=False))
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:(a/'completion.json').write_text(json.dumps({'failure':failure,'adopted':False,'goal_achieved':False},indent=2))
