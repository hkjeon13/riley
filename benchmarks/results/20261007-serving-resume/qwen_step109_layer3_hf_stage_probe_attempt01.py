"""Observer-only extension of the immutable HF eager capture to row109."""
import hashlib,json,sys,time
from pathlib import Path
from riley_reference import qwen3b_cache_on_layer_stage_trace as m
R=Path('/repo');O=Path('/output');B=Path('/evidence');failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 original=json.loads((B/'qwen3b-p2048-cache-on-layer-stage.json').read_bytes())
 for source in original['provenance']['source_repository']['sources'].values():
  if source['path'].startswith('tools/'):assert sha(R/source['path'])==source['sha256'],'HF source drift '+source['path']
 m.TEACHER_DECODE_TOKEN_COUNT=109
 m.TRACE_STEPS=(m.TRACE_STEPS[0],)+tuple(m.TraceStep(name=f'decode_step_{i}',source_logit_row=i,input_token_count=1,attention_mask_token_count=2048+i,position_start=2047+i,position_end=2047+i,cache_length_before=2047+i,cache_length_after=2048+i,teacher_decode_token_index=i-1) for i in range(1,110))
 m.TRACE_TENSORS=tuple(f'{step.name}.{name}' for step in m.TRACE_STEPS for name in m._BASE_TRACE_TENSORS)
 # Derive the observer from the exact source method; no HF arithmetic call changes.
 import inspect,textwrap
 original_capture=textwrap.dedent(inspect.getsource(m.HuggingFaceQwen3BCacheOnLayerStageTraceBackend.capture))
 capture_source=original_capture
 replacements=[('layer0 = self._layers[0]','layer0 = self._layers[3]'),('if rope_calls == 0:','if rope_calls == 3:'),('for index in range(1, MODEL_LAYER_COUNT):','for index in range(MODEL_LAYER_COUNT):\n        if index == 3: continue')]
 for old,new in replacements:
  assert capture_source.count(old)==1,'observer source context drift'
  capture_source=capture_source.replace(old,new,1)
 capture_source=capture_source.replace('"layer0.','"layer3.')
 original_base=m._BASE_TRACE_TENSORS
 base_shapes=m.cache_free._expected_shapes()
 selected_names=('embedding.last',)+tuple(n.replace('layer0.','layer3.') for n in original_base if n.startswith('layer0.'))+tuple(f'layer{i}.output.last' for i in range(36) if i!=3)+('final_norm.output.last','last_logits')
 assert len(selected_names)==len(set(selected_names))==52
 selected_shapes={n:base_shapes[n.replace('layer3.','layer0.') if n.startswith('layer3.') else n] for n in selected_names}
 m._BASE_TRACE_TENSORS=selected_names
 m.TRACE_TENSORS=tuple(f'{step.name}.{name}' for step in m.TRACE_STEPS for name in selected_names)
 def validate_selected(tensors,torch):
  assert set(tensors)==set(selected_names),'selected capture names differ'
  identities=set()
  for name in selected_names:
   tensor=tensors[name]
   assert tuple(tensor.shape)==selected_shapes[name] and tensor.dtype==torch.bfloat16 and bool(torch.isfinite(tensor).all().item()),'selected tensor contract differs '+name
   assert id(tensor) not in identities,'aliased capture'
   identities.add(id(tensor))
 namespace=dict(vars(m));namespace['_validate_step_tensors']=validate_selected
 exec(compile(capture_source,'frozen-HF-capture:layer3-observer','exec'),namespace)
 class SelectedBackend(m.HuggingFaceQwen3BCacheOnLayerStageTraceBackend):
  capture=namespace['capture']
 workload=m.oracle.load_workload(Path('/inputs/qwen3b-c8-p2048-o128.json'))
 checkpoint=m.oracle.inspect_checkpoint(Path('/checkpoint'))
 teacher=m._load_teacher_stream(teacher_manifest_path=Path('/teacher/teacher-forced-oracle.json'),teacher_cache_on_sidecar_path=Path('/teacher/cache-on-logits.safetensors'))
 source_rows=m._load_source_logit_rows(teacher_cache_on_sidecar_path=Path('/teacher/cache-on-logits.safetensors'))
 backend=SelectedBackend.load(checkpoint=checkpoint,device='cuda:0')
 producer=backend.producer_metadata
 for key in ['python_version','python_executable_sha256','torch_version','transformers_version','safetensors_version','transformers_qwen2_source']:
  assert producer[key]==original['producer'][key],'producer differs '+key
 (O/'producer.json').write_text(json.dumps(producer,indent=2)+'\n')
 captured=backend.capture(prompt_token_ids=workload.prompt_token_ids,teacher_decode_token_ids=teacher.token_ids[:109],expected_source_logits=source_rows)
 tensors=dict(captured.tensors);assert set(tensors)==set(m.TRACE_TENSORS)
 # Capture itself rejects any of the110 source-logit mismatches before return.
 raw=bytearray();metadata={}
 for step in [108,109]:
  for name in m._BASE_TRACE_TENSORS:
   key=f'decode_step_{step}.{name}';tensor=tensors[key];data=m._canonical_bf16_le_bytes(tensor,backend._torch);a=len(raw);raw.extend(data);metadata[key]={'shape':list(tensor.shape),'dtype':'BF16','raw_data_offsets':[a,len(raw)],'sha256':hashlib.sha256(data).hexdigest()}
 with (O/'selected-stages.bf16').open('xb') as f:f.write(raw)
 report={'schema':'riley.qwen-hf-step108-109-layer3-observer.v1','selected_layer_index':3,'original_capture_method_sha256':hashlib.sha256(original_capture.encode()).hexdigest(),'observer_capture_method_sha256':hashlib.sha256(capture_source.encode()).hexdigest(),'producer':producer,'overlay_script_sha256':sha(Path(__file__)),'original_HF_manifest_sha256':sha(B/'qwen3b-p2048-cache-on-layer-stage.json'),'teacher_manifest_sha256':sha('/teacher/teacher-forced-oracle.json'),'teacher_sidecar_sha256':sha('/teacher/cache-on-logits.safetensors'),'source110_logits_all_exact':True,'selected_stages':metadata,'raw_sha256':hashlib.sha256(raw).hexdigest(),'diagnostic_only':True,'serving_performance':'미실행','goal_achieved':False}
 (O/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 backend.close()
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','goal_achieved':False},indent=2)+'\n')
