import torch,ctypes,json,os
from pathlib import Path
r=Path('/tmp/riley-g04-serving-trace-260911');out=Path('/tmp/riley-g04-attention-policy-260911')
lib=ctypes.CDLL(str(out/os.environ.get('RILEY_MMA_LIBRARY','mma_attention.so')))
lib.replay.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p];lib.replay.restype=ctypes.c_int
def native(q,k,v):
 q=q.contiguous();k=k.contiguous();v=v.contiguous();result=torch.empty_like(q)
 rc=lib.replay(q.data_ptr(),k.data_ptr(),v.data_ptr(),result.data_ptr(),q.shape[0],k.shape[0],torch.cuda.current_stream().cuda_stream)
 assert rc==0,rc
 return result

def read(l,c,name,heads):
 return torch.frombuffer(bytearray((r/f'layer{l}-call{c}-{name}.bf16').read_bytes()),dtype=torch.bfloat16).reshape(-1,heads,64).cuda()
if __name__=='__main__':
 records=[]
 with torch.inference_mode():
  for l in range(30):
   k=None;v=None
   for c in range(32):
    kk=read(l,c,'k',3);vv=read(l,c,'v',3)
    k=kk if k is None else torch.cat([k,kk]);v=vv if v is None else torch.cat([v,vv])
    result=native(read(l,c,'q',9),k,v);target=read(l,c,'context',9)
    records.append({'layer':l,'call':c,'unequal':int((result!=target).sum().item()),'max_abs':float((result.float()-target.float()).abs().max().item())})
 out.joinpath(os.environ.get('RILEY_MMA_RESULT','mma-comparison.json')).write_text(json.dumps({'records':records,'total_unequal':sum(x['unequal'] for x in records),'performance_trials':0},indent=2)+'\n')
 print('mma unequal',sum(x['unequal'] for x in records),'exact cases',sum(x['unequal']==0 for x in records),flush=True)
