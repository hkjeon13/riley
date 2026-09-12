import torch,json
from pathlib import Path
from safetensors.torch import load_file
r=Path('/tmp/riley-g04-native-profile-260911');w=load_file('/data/riley-vllm-interim.CfrT9T/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors',device='cuda')
p='model.layers.0.'
weights={'qkv':torch.cat([w[p+'self_attn.'+n+'_proj.weight'] for n in ['q','k','v']],0),'o':w[p+'self_attn.o_proj.weight'],'gate_up':torch.cat([w[p+'mlp.'+n+'_proj.weight'] for n in ['gate','up']],0),'down':w[p+'mlp.down_proj.weight']}
records=[]
for name,weight in weights.items():
 x=torch.zeros((128,weight.shape[1]),device='cuda',dtype=torch.bfloat16)
 for _ in range(3): torch.nn.functional.linear(x,weight)
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof: y=torch.nn.functional.linear(x,weight)
 torch.cuda.synchronize();trace=r/'temporary-shape-trace.json';prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents'];trace.unlink()
 kernels=[{'name':e['name'],'grid':e['args']['grid'],'block':e['args']['block']} for e in events if e.get('cat')=='kernel']
 row={'point':name,'n':weight.shape[0],'k':weight.shape[1],'kernels':kernels};records.append(row);print(json.dumps(row),flush=True)
(r/'prefill-gemm-shapes.json').write_text(json.dumps({'records':records,'performance_trials':0},indent=2)+'\n')
