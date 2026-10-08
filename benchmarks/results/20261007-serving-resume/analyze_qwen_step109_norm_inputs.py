"""Describe exact norm inputs and source-reduction arithmetic; no CUDA rsqrt emulation."""
import json,struct,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'qwen-step109-layer3-independent-localization-attempt02';hf=O/'qwen-step109-layer3-hf-stage-validation-attempt02';native=O/'qwen-full128-validation-attempt04';h=json.loads((hf/'result.json').read_bytes());n=json.loads((native/'qwen3b-p2048-cache-on-full128-native-result.json').read_bytes());hr=(hf/'selected-stages.bf16').read_bytes();nr=(native/Path(n['selected_stage_raw_sidecar']['path']).name).read_bytes();sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha(hr)==h['raw_sha256'] and sha(nr)==n['selected_stage_raw_sidecar']['sha256']
def tensor(meta,raw,key):
 item=meta['selected_stages'][key];a,b=item['raw_data_offsets'];result=raw[a:b];assert sha(result)==item['sha256'];return result
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def floats(raw):return [struct.unpack('<f',struct.pack('<I',word[0]<<16))[0] for word in struct.iter_unpack('<H',raw)]
rows=[]
for step in [108,109]:
 key=f'decode_step_{step}.layer3.after_attention_residual.last';a=tensor(h,hr,key);b=tensor(n,nr,key);assert a==b and len(a)==4096
 values=floats(a);assert all(math.isfinite(x) for x in values)
 lanes=[[0.0]*4 for _ in range(32)]
 for lane in range(32):
  for word in range(lane,512,32):
   for i in range(4):lanes[lane][i]=f32(lanes[lane][i]+f32(values[word*4+i]*values[word*4+i]))
 partial=[]
 for registers in lanes:
  result=registers[0]
  for other in registers[1:]:result=f32(result+other)
  partial.append(result)
 for offset in [16,8,4,2,1]:
  previous=partial[:];partial=[f32(previous[i]+previous[i+offset]) if i+offset<32 else previous[i] for i in range(32)]
 variance=f32(partial[0]/2048);high_precision=math.fsum(x*x for x in values)/2048;rounded=f32(high_precision)
 norm_key=f'decode_step_{step}.layer3.post_attention_norm.last';x=tensor(h,hr,norm_key);y=tensor(n,nr,norm_key);hx=floats(x);ny=floats(y);differing=[{'index':i,'HF':u,'native':v,'absolute_difference':abs(u-v)} for i,(u,v) in enumerate(zip(hx,ny)) if u!=v]
 rows.append({'step':step,'input_exact_BF16':True,'input_sha256':sha(a),'source_warp32_four_register_variance_F32':variance,'source_warp32_variance_bits':hex(bits(variance)),'FP64_fsum_variance_reference':high_precision,'rounded_FP64_reference_F32_bits':hex(bits(rounded)),'variance_F32_ULP_difference_from_rounded_FP64_reference':bits(variance)-bits(rounded),'all_nonzero_squared_inputs_FP32_normal':all(f32(v*v)==0 or abs(f32(v*v))>=2**-126 for v in values),'norm_output_unequal_elements':len(differing),'norm_output_differences':differing})
p=Path('/Users/psyche/.codex/worktrees/qwen-cache-full128/riley/kernels/src/primitives.cu');receipt=json.loads((R/'qwen-full128-source-receipt-attempt03.json').read_bytes());assert sha(p.read_bytes())==receipt['files']['kernels/src/primitives.cu']
report={'source_commit':receipt['source_commit'],'native_primitive_source_sha256':sha(p.read_bytes()),'rows':rows,'scope':'CPU description of pinned native sum reduction and exact captured input/output; FP64 reference is diagnostic only, not a replacement numerical policy','unmeasured':['actual HF variance FP32 bits','actual native/HF rsqrt FP32 bits','standalone RMSNorm replay with freshly verified checkpoint weight'],'cause':'RMSNorm boundary isolated; reduction/rsqrt/rounding cause remains unproven until actual same-input isolated operator replay','arithmetic_or_tolerance_change':False,'serving_performance':'미실행','goal_achieved':False}
(R/'qwen-step109-layer3-norm-input-analysis-attempt01.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v for k,v in row.items() if k!='norm_output_differences'} for row in rows]))
