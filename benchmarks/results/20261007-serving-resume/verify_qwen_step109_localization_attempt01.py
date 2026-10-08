import json,hashlib,struct,tarfile,time
from pathlib import Path
from verify_qwen_full128_native_raw_attempt02 import verify
R=Path(__file__).resolve().parent;C=R/'qwen-step109-localization-collection-attempt01';O=R/'qwen-step109-independent-localization-attempt01';O.mkdir();failure=None
sha=lambda raw:hashlib.sha256(raw).hexdigest()
names=['embedding.last','layer0.input_norm.last','layer0.q_proj.last','layer0.k_proj.last','layer0.v_proj.last','layer0.q_rope.last','layer0.k_rope.last','layer0.attention_context.last','layer0.after_attention_residual.last','layer0.post_attention_norm.last','layer0.gate_proj.last','layer0.up_proj.last','layer0.gated.last','layer0.down_proj.last','layer0.output.last',*[f'layer{i}.output.last' for i in range(1,36)],'final_norm.output.last','last_logits']
try:
 deadline=time.monotonic()+21600
 while not (C/'completion.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('paired collection still incomplete; no localization inferred')
  time.sleep(20)
 assert json.loads((C/'completion.json').read_bytes())['failure'] is None
 for archive in C.glob('*.tar.gz'):
  with tarfile.open(archive,'r:gz') as tar:
   for member in tar:
    assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    if not member.isfile():continue
    target=O/member.name;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(tar.extractfile(member).read())
 hf=O/'qwen-step109-hf-stage-validation-attempt01';native=O/'qwen-full128-validation-attempt02'
 h=json.loads((hf/'result.json').read_bytes());n=json.loads((native/'qwen3b-p2048-cache-on-full128-native-result.json').read_bytes());receipt=json.loads((R/'qwen-full128-source-receipt-attempt02.json').read_bytes());full=verify(native,R/'qwen-full128-independent-collection-attempt01/teacher-cache-on-logits.safetensors',receipt)
 assert h['source110_logits_all_exact'] is True and h['teacher_sidecar_sha256']=='d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
 assert h['overlay_script_sha256']==sha((R/'qwen_step109_hf_stage_probe_attempt01.py').read_bytes())
 original=json.loads((R/'verified-stages-attempt02/hf-original/qwen3b-p2048-cache-on-layer-stage.json').read_bytes())
 for k in ['python_version','python_executable_sha256','torch_version','transformers_version','safetensors_version','transformers_qwen2_source']:assert h['producer'][k]==original['producer'][k]
 hr=(hf/'selected-stages.bf16').read_bytes();nr=(native/Path(n['selected_stage_raw_sidecar']['path']).name).read_bytes();assert sha(hr)==h['raw_sha256'] and sha(nr)==n['selected_stage_raw_sidecar']['sha256'] and len(nr)==n['selected_stage_raw_sidecar']['bytes']
 expected={f'decode_step_{i}.{name}' for i in [108,109] for name in names};assert set(h['selected_stages'])==set(n['selected_stages'])==expected
 results=[]
 for step in [108,109]:
  rows=[]
  for name in names:
   key=f'decode_step_{step}.{name}';hs=h['selected_stages'][key];ns=n['selected_stages'][key];a,b=hs['raw_data_offsets'];x,y=ns['raw_data_offsets'];hb=hr[a:b];nb=nr[x:y];assert sha(hb)==hs['sha256'] and sha(nb)==ns['sha256'] and len(nb)==len(hb)==ns['bytes'];unequal=sum(x!=y for x,y in zip(struct.iter_unpack('<H',hb),struct.iter_unpack('<H',nb)));rows.append({'stage':key,'bf16_exact':hb==nb,'unequal_elements':unequal})
  results.append({'step':step,'exact_stages':sum(r['bf16_exact'] for r in rows),'first_non_exact_boundary':next((r['stage'] for r in rows if not r['bf16_exact']),None),'rows':rows})
 report={'full128_replay':{k:v for k,v in full.items() if k!='rows'},'selected104_raw_stages_verified':104,'HF_source110_logits_exact':True,'steps':results,'scope':'first observed causal boundary only; finer internal operation remains unmeasured where not captured','serving_performance':'미실행','goal_achieved':False};(O/'independent-localization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='steps'}));print([(x['step'],x['exact_stages'],x['first_non_exact_boundary']) for x in results])
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'goal_achieved':False},indent=2)+'\n')
