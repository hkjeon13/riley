#!/usr/bin/env python3
"""Functional/numerical observations for the optional adapter; no quality gate."""
import argparse
import ctypes
import json
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--library',required=True)
parser.add_argument('--native-only',action='store_true',help='skip the Python FlashInfer oracle for isolated native memory checking')
args=parser.parse_args()
flashinfer=None
if not args.native_only:
    import flashinfer
torch.manual_seed(91303);torch.set_num_threads(4)
lib=ctypes.CDLL(args.library)
lib.riley_flashinfer_decode_workspace_bytes.restype=ctypes.c_uint64
lib.riley_flashinfer_decode_prepare.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
lib.riley_flashinfer_decode_prepare.restype=ctypes.c_int
lib.riley_flashinfer_decode_run.argtypes=[ctypes.c_void_p]*6+[ctypes.c_uint64]
lib.riley_flashinfer_decode_run.restype=ctypes.c_int
workspace_bytes=lib.riley_flashinfer_decode_workspace_bytes()
workspace=torch.empty(workspace_bytes,dtype=torch.uint8,device='cuda')
shape=torch.zeros((32,416),dtype=torch.int32,device='cuda')
active=torch.zeros(1,dtype=torch.int32,device='cuda')
status=torch.zeros(1,dtype=torch.int32,device='cuda')
q=torch.randn(32,9,64,dtype=torch.bfloat16,device='cuda')
k=torch.randn(4096,3,16,64,dtype=torch.bfloat16,device='cuda');v=torch.randn_like(k)
out=torch.empty_like(q)
print(json.dumps({'torch':torch.__version__,'flashinfer':getattr(flashinfer,'__version__',None), 'native_only':args.native_only,
                  'gpu':torch.cuda.get_device_name(0),'workspace_bytes':workspace_bytes}),flush=True)

def execute():
    stream=torch.cuda.current_stream().cuda_stream
    assert lib.riley_flashinfer_decode_prepare(stream,shape.data_ptr(),active.data_ptr(),workspace.data_ptr(),workspace_bytes,4096,4096,status.data_ptr())==0
    assert lib.riley_flashinfer_decode_run(stream,q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),workspace.data_ptr(),workspace_bytes)==0


def configure(rows,lengths):
    host=torch.zeros((32,416),dtype=torch.int32)
    count=sum((n+15)//16 for n in lengths)
    indices=torch.randperm(4096)[:count].to(torch.int32)
    offset=0;indptr=[0]
    for row,n in enumerate(lengths):
        pages=(n+15)//16;host[row,1]=n-1;host[row,32:32+pages]=indices[offset:offset+pages]
        offset+=pages;indptr.append(offset)
    shape.copy_(host);active.fill_(rows);status.zero_();out.fill_(77)
    return host,indices,torch.tensor(indptr,dtype=torch.int32)


def references(rows,lengths,host):
    output=[]
    for row,n in enumerate(lengths):
        ids=host[row,32:32+(n+15)//16].long().cuda()
        kr=k[ids].permute(1,0,2,3).reshape(3,-1,64)[:,:n].repeat_interleave(3,0).float()
        vr=v[ids].permute(1,0,2,3).reshape(3,-1,64)[:,:n].repeat_interleave(3,0).float()
        scores=torch.einsum('hd,hnd->hn',q[row].float(),kr)*.125
        output.append(torch.einsum('hn,hnd->hd',scores.softmax(-1),vr))
    return torch.stack(output)

def independent_unsplit(rows,lengths,indices,indptr):
    # 0.6.16 CUDA-core ignores disable_split_kv. A large duplicate batch makes
    # the independent planner choose unsplit; verify its actual plan flag.
    assert flashinfer.__version__=='0.6.16.post3','review plan-info schema for another version'
    repeats=(4096+rows-1)//rows
    sizes=indptr[1:]-indptr[:-1]
    big_indptr=torch.cat([torch.zeros(1,dtype=torch.int32),sizes.repeat(repeats).cumsum(0).to(torch.int32)])
    big_indices=indices.repeat(repeats)
    last=torch.tensor([n%16 or 16 for n in lengths],dtype=torch.int32).repeat(repeats)
    work=torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda')
    wrapper=flashinfer.BatchDecodeWithPagedKVCacheWrapper(work,kv_layout='HND',use_tensor_cores=False)
    wrapper.plan(big_indptr,big_indices,last,9,3,64,16,pos_encoding_mode='NONE',
                 q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16)
    assert len(wrapper._plan_info)==10 and wrapper._plan_info[9]==0,'oracle planner still uses split KV'
    answer=wrapper.run(q[:rows].repeat(repeats,1,1),(k,v))[:rows].clone()
    torch.cuda.synchronize()
    return answer

records=[]
for rows in [1,4,16,32]:
    lengths=[1,16,398,4096]*(rows//4) if rows>1 else [1]
    host,indices,indptr=configure(rows,lengths)
    execute();torch.cuda.synchronize();baseline=out.clone()
    assert status.item()==0
    assert bool(baseline[:rows].isfinite().all())
    assert bool((baseline[rows:]==77).all()),'inactive output modified'
    ref=references(rows,lengths,host)
    if rows==1:assert torch.equal(baseline[:rows],ref.bfloat16())
    expected=None if args.native_only else independent_unsplit(rows,lengths,indices,indptr)
    # Same FlashInfer unsplit arithmetic, independent Python planning/metadata.
    if expected is not None:assert torch.equal(baseline[:rows],expected),'native differs from FlashInfer unsplit wrapper'
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):execute()
    out.fill_(77);graph.replay();torch.cuda.synchronize()
    assert torch.equal(baseline,out),'captured graph differs from eager'
    q.mul_(.5);status.zero_();execute();torch.cuda.synchronize();changed=out.clone()
    graph.replay();torch.cuda.synchronize();assert torch.equal(changed,out)
    # Change active count, page IDs and lengths without capture or host planning.
    new_rows=max(1,rows//2);new_lengths=[1 if i%2==0 else 31 for i in range(new_rows)]
    new_host,new_indices,new_indptr=configure(new_rows,new_lengths);execute();torch.cuda.synchronize();dynamic=out.clone()
    out.fill_(77);graph.replay();torch.cuda.synchronize();assert torch.equal(dynamic,out)
    assert status.item()==0
    if not args.native_only:
        dynamic_reference=independent_unsplit(new_rows,new_lengths,new_indices,new_indptr)
        assert torch.equal(out[:new_rows],dynamic_reference),'updated metadata differs from independent plan'
    error=(baseline[:rows].float()-ref).abs()
    records.append({'rows':rows,'lengths':lengths,'exact_flashinfer_unsplit':None if args.native_only else True,'graph_exact_eager':True,
                    'updated_q_graph_exact':True,'updated_metadata_and_rows_graph_exact':True,
                    'fp32_max_abs_error':float(error.max()),'fp32_rmse':float(error.square().mean().sqrt()),
                    'numeric_profile_accepted':False})
    print(json.dumps(records[-1]),flush=True)

# Invalid metadata must prevent the affected row from touching invalid pages.
configure(4,[1,16,398,4096]);shape[0,32]=4096
execute();torch.cuda.synchronize();assert status.item() & 8;assert bool((out[0]==77).all())
shape[1,1]=4096;status.zero_();out.fill_(77)
execute();torch.cuda.synchronize();assert status.item() & 4;assert bool((out[1]==77).all())
active.fill_(33);status.zero_();out.fill_(77)
execute();torch.cuda.synchronize();assert status.item() & 2;assert bool((out==77).all())
assert lib.riley_flashinfer_decode_prepare(torch.cuda.current_stream().cuda_stream,shape.data_ptr(),active.data_ptr(),workspace.data_ptr(),workspace_bytes-1,4096,4096,status.data_ptr())!=0
print(json.dumps({'cases_passed':len(records),'invalid_page_position_active_and_extent_rejected':True,
                  'full_model_tested':False,'serving_tested':False}),flush=True)
