"""Pinned HF isolated RMSNorm replay; shape counterfactuals are diagnostics only."""
import hashlib,json,struct,inspect,sys
from pathlib import Path
import torch,safetensors,transformers
from safetensors import safe_open
from transformers.models.qwen2 import modeling_qwen2 as qm
I=Path('/inputs');O=Path('/output');M=Path('/checkpoint');failure=None
sha=lambda b:hashlib.sha256(b).hexdigest();file_sha=lambda p:sha(Path(p).read_bytes())
def raw(t):return t.detach().contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False).tobytes()
def fpbits(t):return hex(struct.unpack('<I',struct.pack('<f',t.item()))[0])
try:
 manifest=json.loads((I/'manifest.json').read_bytes());producer={'python_version':sys.version.split()[0],'python_executable_sha256':file_sha(sys.executable),'torch_version':torch.__version__,'transformers_version':transformers.__version__,'safetensors_version':safetensors.__version__}
 for k,v in producer.items():assert v==manifest['HF_producer'][k],'producer drift '+k
 expected=manifest['HF_producer']['transformers_qwen2_source'];assert file_sha(qm.__file__)==expected['sha256'],'Qwen2 source drift'
 config=json.loads((M/'config.json').read_bytes());assert config['hidden_size']==2048 and struct.unpack('<I',struct.pack('<f',config['rms_norm_eps']))[0]==0x358637bd
 name=manifest['tensor_name'];index=M/'model.safetensors.index.json'
 if index.exists():weight_file=M/json.loads(index.read_bytes())['weight_map'][name]
 else:weight_file=M/'model.safetensors'
 with safe_open(str(weight_file),framework='pt',device='cpu') as source:weight=source.get_tensor(name)
 assert tuple(weight.shape)==(2048,) and weight.dtype==torch.bfloat16
 weight_raw=raw(weight);(O/'norm-weight.bf16').write_bytes(weight_raw)
 norm=qm.Qwen2RMSNorm(2048,eps=config['rms_norm_eps']).to(device='cuda:0',dtype=torch.bfloat16).eval()
 with torch.no_grad():norm.weight.copy_(weight.to('cuda:0'))
 rows=[]
 for item in manifest['rows']:
  files={x['role']:x for x in item['files']}
  for f in files.values():assert file_sha(I/f['name'])==f['sha256'] and (I/f['name']).stat().st_size==f['bytes']
  source_bytes=(I/files['input']['name']).read_bytes();x=torch.frombuffer(bytearray(source_bytes),dtype=torch.bfloat16).clone().reshape(1,1,2048).to('cuda:0')
  expected_hf=(I/files['HF_output']['name']).read_bytes();expected_native=(I/files['native_output']['name']).read_bytes();shapes=[]
  for count in [1,16,2048]:
   input=x.expand(1,count,2048).clone()
   with torch.no_grad():
    result=norm(input);xf=input.float();variance=xf.pow(2).mean(-1,keepdim=True);variance_eps=variance+config['rms_norm_eps'];inverse=torch.rsqrt(variance_eps);normalized=(xf*inverse).to(torch.bfloat16);manual=norm.weight*normalized
   assert raw(manual)==raw(result),'scalar observation did not reproduce original Qwen2 class output'
   output=raw(result[0,-1,:]);all_equal=bool(torch.all(result==result[:,0:1,:]).item());assert all_equal,'identical rows do not reproduce identical output'
   if count==1:assert output==expected_hf,'isolated singleton HF replay differs from captured original output'
   path=f'step{item["step"]}-rows{count}-HF-output.bf16';(O/path).write_bytes(output)
   shapes.append({'rows':count,'shape':list(input.shape),'variance_FP32_bits':fpbits(variance[0,-1,0]),'variance_plus_epsilon_FP32_bits':fpbits(variance_eps[0,-1,0]),'inverse_RMS_FP32_bits':fpbits(inverse[0,-1,0]),'output_sha256':sha(output),'captured_HF_exact':output==expected_hf,'captured_native_exact':output==expected_native,'output_file':path,'all_replicated_rows_equal':all_equal})
  rows.append({'step':item['step'],'shapes':shapes})
 report={'producer':producer,'Qwen2_source_sha256':file_sha(qm.__file__),'Qwen2_norm_forward_source_sha256':sha(inspect.getsource(qm.Qwen2RMSNorm.forward).encode()),'input_manifest_sha256':file_sha(I/'manifest.json'),'weight_tensor_name':name,'weight_sha256':sha(weight_raw),'weight_file':weight_file.name,'weight_file_sha256':file_sha(weight_file),'checkpoint_revision':manifest['checkpoint_revision'],'rows':rows,'arithmetic_policy':'actual pinned Qwen2RMSNorm class and FP32 Torch mean/rsqrt; BF16 normalization and weighted output boundaries unchanged','serving_performance':'미실행','goal_achieved':False};(O/'result.json').write_text(json.dumps(report,indent=2)+'\n')
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','goal_achieved':False},indent=2)+'\n')
