import json,hashlib,struct,tarfile,time,math
from pathlib import Path
from verify_qwen_full128_native_raw_attempt03 import verify
R=Path(__file__).resolve().parent;C=R/'qwen-step109-layer3-localization-collection-attempt01';O=R/'qwen-step109-layer3-independent-localization-attempt01';O.mkdir();failure=None
sha=lambda raw:hashlib.sha256(raw).hexdigest()
names=['embedding.last', 'layer0.output.last', 'layer1.output.last', 'layer2.output.last', 'layer3.input_norm.last', 'layer3.q_proj.last', 'layer3.k_proj.last', 'layer3.v_proj.last', 'layer3.q_rope.last', 'layer3.k_rope.last', 'layer3.attention_context.last', 'layer3.after_attention_residual.last', 'layer3.post_attention_norm.last', 'layer3.gate_proj.last', 'layer3.up_proj.last', 'layer3.gated.last', 'layer3.down_proj.last', 'layer3.output.last', 'layer4.output.last', 'layer5.output.last', 'layer6.output.last', 'layer7.output.last', 'layer8.output.last', 'layer9.output.last', 'layer10.output.last', 'layer11.output.last', 'layer12.output.last', 'layer13.output.last', 'layer14.output.last', 'layer15.output.last', 'layer16.output.last', 'layer17.output.last', 'layer18.output.last', 'layer19.output.last', 'layer20.output.last', 'layer21.output.last', 'layer22.output.last', 'layer23.output.last', 'layer24.output.last', 'layer25.output.last', 'layer26.output.last', 'layer27.output.last', 'layer28.output.last', 'layer29.output.last', 'layer30.output.last', 'layer31.output.last', 'layer32.output.last', 'layer33.output.last', 'layer34.output.last', 'layer35.output.last', 'final_norm.output.last', 'last_logits']
try:
 deadline=time.monotonic()+21600
 while not (C/'completion.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('paired collection still incomplete; no localization inferred')
  time.sleep(20)
 assert json.loads((C/'completion.json').read_bytes())['failure'] is None
 collection=json.loads((C/'completion.json').read_bytes())
 before=json.loads((C/'source-manifest-before.json').read_bytes());after=json.loads((C/'source-manifest-after.json').read_bytes());assert before==after
 assert len(collection['records'])==2
 for record in collection['records']:
  archive=C/Path(record['archive']).name
  assert sha(archive.read_bytes())==record['archive_sha256']
  expected_files=before[record['scope']]['files'];observed_files={}
  with tarfile.open(archive,'r:gz') as tar:
   for member in tar:
    assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    if member.isdir():continue
    assert member.isfile() and member.name.startswith(record['scope']+'/')
    relative=member.name[len(record['scope'])+1:];assert relative not in observed_files
    raw=tar.extractfile(member).read();observed_files[relative]={'sha256':sha(raw),'bytes':len(raw)}
    target=O/member.name;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(raw)
  assert observed_files==expected_files
 hf=O/'qwen-step109-layer3-hf-stage-validation-attempt01';native=O/'qwen-full128-validation-attempt03'
 h=json.loads((hf/'result.json').read_bytes());n=json.loads((native/'qwen3b-p2048-cache-on-full128-native-result.json').read_bytes());receipt=json.loads((R/'qwen-full128-source-receipt-attempt03.json').read_bytes());full=verify(native,R/'qwen-full128-independent-collection-attempt01/teacher-cache-on-logits.safetensors',receipt)
 assert h['selected_layer_index']==n['selected_layer_index']==3
 assert h['source110_logits_all_exact'] is True and h['teacher_sidecar_sha256']=='d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
 assert h['overlay_script_sha256']==sha((R/'qwen_step109_layer3_hf_stage_probe_attempt01.py').read_bytes())
 original=json.loads((R/'verified-stages-attempt02/hf-original/qwen3b-p2048-cache-on-layer-stage.json').read_bytes())
 for k in ['python_version','python_executable_sha256','torch_version','transformers_version','safetensors_version','transformers_qwen2_source']:assert h['producer'][k]==original['producer'][k]
 hr=(hf/'selected-stages.bf16').read_bytes();nr=(native/Path(n['selected_stage_raw_sidecar']['path']).name).read_bytes();assert sha(hr)==h['raw_sha256'] and sha(nr)==n['selected_stage_raw_sidecar']['sha256'] and len(nr)==n['selected_stage_raw_sidecar']['bytes']
 expected={f'decode_step_{i}.{name}' for i in [108,109] for name in names};assert set(h['selected_stages'])==set(n['selected_stages'])==expected
 for metadata,raw in [(h['selected_stages'],hr),(n['selected_stages'],nr)]:
  intervals=sorted(tuple(item['raw_data_offsets']) for item in metadata.values());position=0
  for a,b in intervals:
   assert a==position and a<b<=len(raw);position=b
  assert position==len(raw),'raw gaps or trailing bytes'
 results=[]
 for step in [108,109]:
  rows=[]
  for name in names:
   key=f'decode_step_{step}.{name}';hs=h['selected_stages'][key];ns=n['selected_stages'][key];a,b=hs['raw_data_offsets'];x,y=ns['raw_data_offsets'];hb=hr[a:b];nb=nr[x:y];assert hs['dtype']=='BF16' and len(hb)==2*math.prod(hs['shape']);assert sha(hb)==hs['sha256'] and sha(nb)==ns['sha256'] and len(nb)==len(hb)==ns['bytes'];unequal=sum(x!=y for x,y in zip(struct.iter_unpack('<H',hb),struct.iter_unpack('<H',nb)));rows.append({'stage':key,'bf16_exact':hb==nb,'unequal_elements':unequal})
  results.append({'step':step,'exact_stages':sum(r['bf16_exact'] for r in rows),'first_non_exact_boundary':next((r['stage'] for r in rows if not r['bf16_exact']),None),'rows':rows})
 report={'full128_replay':{k:v for k,v in full.items() if k!='rows'},'selected104_raw_stages_verified':104,'HF_source110_logits_exact':True,'steps':results,'scope':'first observed captured boundary in forward order; prior KV and uncaptured attention scores/probabilities remain unverified; no internal causal operation inferred','serving_performance':'미실행','goal_achieved':False};(O/'independent-localization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='steps'}));print([(x['step'],x['exact_stages'],x['first_non_exact_boundary']) for x in results])
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'goal_achieved':False},indent=2)+'\n')
