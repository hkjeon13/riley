"""Exercise the immutable HF final validator's global lookup with metadata only."""
import ast,json,sys,hashlib,time
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent;W=Path('/Users/psyche/.codex/worktrees/qwen-cache-full128/riley');sys.path.insert(0,str(W/'tools/python/reference'))
from riley_reference import qwen3b_cache_on_layer_stage_trace as m
old=m._BASE_TRACE_TENSORS;names=('embedding.last',)+tuple(n.replace('layer0.','layer3.') for n in old if n.startswith('layer0.'))+tuple(f'layer{i}.output.last' for i in range(36) if i!=3)+('final_norm.output.last','last_logits');assert len(set(names))==52
base=m.cache_free._expected_shapes();shapes={n:base[n.replace('layer3.','layer0.') if n.startswith('layer3.') else n] for n in names}
m.TRACE_STEPS=tuple(SimpleNamespace(name=f'decode_step_{i}') for i in range(110));m.TRACE_TENSORS=tuple(f'{step.name}.{n}' for step in m.TRACE_STEPS for n in names)
class Tensor:
 def __init__(self,shape):self.shape=shape;self.dtype='BF16';self.finite=True
class Finite:
 def __init__(self,value):self.value=value
 def all(self):return self
 def item(self):return self.value
torch=SimpleNamespace(bfloat16='BF16',isfinite=lambda tensor:Finite(tensor.finite))
def tensors():return {f'{step.name}.{n}':Tensor(shapes[n]) for step in m.TRACE_STEPS for n in names}
raw=tensors()
try:m._validate_tensors(raw,torch)
except m.Qwen3BCacheOnLayerStageTraceError as e:assert str(e)=='trace tensor names differ'
else:raise AssertionError('original final validator did not reproduce actual failure')
overlay=R/'qwen_step109_layer3_hf_stage_probe_attempt02.py';tree=ast.parse(overlay.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='validate_selected');env={'selected_names':names,'selected_shapes':shapes};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(overlay),'exec'),env)
m._validate_step_tensors=env['validate_selected'];m._validate_tensors(raw,torch)
negative=[]
for name in ['names','shape','dtype','finite','alias']:
 bad=tensors();key='decode_step_109.embedding.last'
 if name=='names':bad['unexpected']=bad.pop(key)
 elif name=='shape':bad[key].shape=(1,)
 elif name=='dtype':bad[key].dtype='FP32'
 elif name=='finite':bad[key].finite=False
 elif name=='alias':bad['decode_step_109.layer0.output.last']=bad[key]
 try:m._validate_tensors(bad,torch)
 except (AssertionError,m.Qwen3BCacheOnLayerStageTraceError):negative.append(name)
 else:raise AssertionError('required guard weakened: '+name)
report={'created_ns':time.time_ns(),'original_final_validator_failure_reproduced':True,'repaired_global_binding_validates_all_steps':110,'selected_tensors_per_step':52,'negative_guards_rejected':negative,'observer_source_sha256':hashlib.sha256(overlay.read_bytes()).hexdigest(),'immutable_HF_validator_source_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'scope':'metadata-only validation of original source function global lookup and exact observer validator; no numeric tensors, GPU inference or HF logits claim','serving_performance':'미실행','goal_achieved':False}
(R/'qwen-layer3-observer-validation-binding-preflight-attempt02.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
