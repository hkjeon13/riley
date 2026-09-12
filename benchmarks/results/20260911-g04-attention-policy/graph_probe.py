import torch,ctypes,json
from pathlib import Path
r=Path('/tmp/riley-g04-serving-trace-260911');out=Path('/tmp/riley-g04-attention-policy-260911')
lib=ctypes.CDLL(str(out/'mma_attention_graph.so'))
lib.replay_dynamic.argtypes=[ctypes.c_void_p]*6;lib.replay_dynamic.restype=ctypes.c_int
q=torch.zeros((1,9,64),device='cuda',dtype=torch.bfloat16)
k=torch.zeros((160,3,64),device='cuda',dtype=torch.bfloat16);v=torch.zeros_like(k);result=torch.empty_like(q)
length=torch.tensor([1],device='cuda',dtype=torch.int32)
def launch():
 assert lib.replay_dynamic(q.data_ptr(),k.data_ptr(),v.data_ptr(),result.data_ptr(),length.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
launch();torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph): launch()
def read(l,c,name,heads):
 return torch.frombuffer(bytearray((r/f'layer{l}-call{c}-{name}.bf16').read_bytes()),dtype=torch.bfloat16).reshape(-1,heads,64).cuda()
records=[]
with torch.inference_mode():
 for layer in range(30):
  k[:128].copy_(read(layer,0,'k',3));v[:128].copy_(read(layer,0,'v',3))
  for call in range(1,32):
   q.copy_(read(layer,call,'q',9));k[127+call:128+call].copy_(read(layer,call,'k',3));v[127+call:128+call].copy_(read(layer,call,'v',3))
   length.fill_(128+call);target=read(layer,call,'context',9)
   before=torch.cuda.memory_allocated();graph.replay();torch.cuda.synchronize();after=torch.cuda.memory_allocated()
   assert before==after
   unequal=int((result!=target).sum().item());assert unequal==0,(layer,call,unequal)
   records.append({'layer':layer,'call':call,'unequal':unequal,'replay_allocated_bytes_delta':after-before})
 for invalid in [0,161]:
  length.fill_(invalid);graph.replay();torch.cuda.synchronize()
  assert torch.isnan(result).all()
 length.fill_(159);graph.replay();torch.cuda.synchronize();assert torch.equal(result,target)
(out/'graph-replay.json').write_text(json.dumps({'captures':1,'replays':len(records),'records':records,
    'all_exact':True,'invalid_lengths_rejected':[0,161],'valid_replay_recovers':True,'performance_trials':0,'scope':'standalone CUDA attention graph with dynamic device length; not full Riley decode graph'},indent=2)+'\n')
print('GRAPH attention captures=1 replays=930 all_exact=true replay_allocations=0 performance_trials=0')
