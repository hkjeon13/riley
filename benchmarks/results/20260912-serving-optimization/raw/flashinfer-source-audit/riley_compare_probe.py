import json, torch, flashinfer, traceback, ctypes, sys
print(json.dumps({"torch":torch.__version__,"flashinfer":flashinfer.__version__,"gpu":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)}),flush=True)
torch.manual_seed(571); torch.set_num_threads(4)
b,n,h,d,page=32,int(sys.argv[1]),3,64,16
pages=(n+page-1)//page
indptr=torch.arange(b+1,dtype=torch.int32)*pages
indices=torch.randperm(b*pages,device="cuda").to(torch.int32)
last=torch.full((b,),n%page or page,dtype=torch.int32)
k=torch.randn(b*pages,h,page,d,device="cuda",dtype=torch.bfloat16)
v=torch.randn_like(k)
q=torch.randn(b,h*3,d,device="cuda",dtype=torch.bfloat16)
references=[]
for row in range(b):
 ids=indices[row*pages:(row+1)*pages].long()
 kr=k[ids].permute(1,0,2,3).reshape(h,-1,d)[:,:n].repeat_interleave(3,0).float()
 vr=v[ids].permute(1,0,2,3).reshape(h,-1,d)[:,:n].repeat_interleave(3,0).float()
 scores=torch.einsum("hd,hnd->hn",q[row].float(),kr)/8
 references.append(torch.einsum("hn,hnd->hd",scores.softmax(-1),vr))
ref=torch.stack(references)
shape=torch.zeros((32,416),dtype=torch.int32,device="cuda");shape[:,1]=n-1
shape[:,32:32+pages]=indices.reshape(32,pages)
live=torch.tensor([32],dtype=torch.int32,device="cuda")
riley=torch.empty_like(q);scratch=torch.empty((32,9,4096),dtype=torch.float32,device="cuda")
bridge=ctypes.CDLL("/tmp/riley-opt-260912/flashinfer-compatibility/riley_attention.so")
bridge.riley_attention.argtypes=[ctypes.c_void_p]*8+[ctypes.c_uint];bridge.riley_attention.restype=ctypes.c_int
assert bridge.riley_attention(torch.cuda.current_stream().cuda_stream,q.data_ptr(),k.data_ptr(),v.data_ptr(),riley.data_ptr(),scratch.data_ptr(),shape.data_ptr(),live.data_ptr(),n)==0
torch.cuda.synchronize()
e=(riley.float()-ref).abs();print(json.dumps({"context":n,"riley_finite":bool(riley.isfinite().all()),"riley_max_abs_vs_fp32":float(e.max()),"riley_rmse_vs_fp32":float(e.square().mean().sqrt())}),flush=True)
for tc in [False,True]:
 try:
  work=torch.empty(64*1024*1024,dtype=torch.uint8,device="cuda")
  w=flashinfer.BatchDecodeWithPagedKVCacheWrapper(work,kv_layout="HND",use_cuda_graph=True,use_tensor_cores=tc,paged_kv_indptr_buffer=indptr.cuda(),paged_kv_indices_buffer=indices.clone(),paged_kv_last_page_len_buffer=last.cuda(),backend="fa2" if tc else "auto")
  w.plan(indptr,indices,last,9,3,64,16,pos_encoding_mode="NONE",q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16)
  out=w.run(q,(k,v));torch.cuda.synchronize()
  diff=(out.float()-riley.float()).abs(); print(json.dumps({"context":n,"tensor_cores":tc,"riley_exact":bool(torch.equal(out,riley)),"riley_different_elements":int((out!=riley).sum()),"elements":out.numel(),"max_abs_vs_riley":float(diff.max())}),flush=True)
  err=(out.float()-ref).abs(); rec={"tensor_cores":tc,"finite":bool(out.isfinite().all()),"max_abs_vs_fp32":float(err.max()),"rmse_vs_fp32":float(err.square().mean().sqrt()),"exact_vs_rounded_fp32":bool(torch.equal(out,ref.bfloat16()))}
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g): captured=w.run(q,(k,v))
  g.replay();torch.cuda.synchronize();rec["graph_exact_eager"]=bool(torch.equal(out,captured))
  q.mul_(0.5); expected=w.run(q,(k,v)).clone();g.replay();torch.cuda.synchronize();rec["updated_q_graph_exact_eager"]=bool(torch.equal(expected,captured));q.mul_(2)
  print(json.dumps(rec),flush=True)
 except Exception as e:
  print(json.dumps({"tensor_cores":tc,"error":str(e)}),flush=True);traceback.print_exc()
