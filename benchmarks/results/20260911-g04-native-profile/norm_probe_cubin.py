import ctypes as C,json,torch
from pathlib import Path
from safetensors.torch import load_file
r=Path('/tmp/riley-g04-native-profile-260911')
w=load_file('/data/riley-vllm-interim.CfrT9T/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors',device='cuda')
def read(name,dtype=torch.bfloat16):return torch.frombuffer(bytearray((r/f'teacher-tail22-{name}.bin').read_bytes()),dtype=dtype).cuda()
a=read('down');old=read('old_res');op=read('o_proj');weight=w['model.layers.23.input_layernorm.weight'];res=torch.empty_like(a);out=torch.empty_like(a);dbg=torch.empty(256*3,device='cuda');b=torch.zeros(768,device='cuda');b[:576]=read('float_res',torch.float32)
ptx=(r/'7UYU2FBV4PKWPHXY6FSENSZU2SIVHH6SUEVG6QI3D2HIBBA5XM5Q.ptx').read_text()
ptx=ptx.replace('%rd<18>','%rd<20>')
ptx=ptx.replace('\tret;', '\tld.param.b64 %rd18, [triton_per_fused_fused_add_rms_norm_2_param_8];\n\tmad.wide.u32 %rd19, %r21, 12, %rd18;\n\tst.global.f32 [%rd19], %r71;\n\tst.global.f32 [%rd19+4], %r73;\n\tst.global.f32 [%rd19+8], %r75;\n\tret;',1)
cu=C.CDLL('libcuda.so.1');module=C.c_void_p();buf=C.create_string_buffer(next(Path("/home/psyche/.cache/vllm/torch_compile_cache/torch_aot_compile/7cf101d73a71350b244cdb09374d2f9cefac6786ad2f9dd08ae0e454a22fb60e/inductor_cache/triton/0/7UYU2FBV4PKWPHXY6FSENSZU2SIVHH6SUEVG6QI3D2HIBBA5XM5Q").glob("*.cubin")).read_bytes());assert cu.cuModuleLoadData(C.byref(module),buf)==0
fn=C.c_void_p();assert cu.cuModuleGetFunction(C.byref(fn),module,b'triton_per_fused_fused_add_rms_norm_2')==0
args=[C.c_void_p(t.data_ptr()) for t in [a,op,old,weight,res,out]]+[C.c_uint(1),C.c_uint(576),C.c_void_p(dbg.data_ptr()),C.c_void_p(0)]
params=(C.c_void_p*len(args))(*[C.addressof(x) for x in args]);assert cu.cuLaunchKernel(fn,1,1,1,256,1,1,32,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)==0
torch.cuda.synchronize();print('ptx vs teacher',int((out!=read('next_norm')).sum()),'debug',dbg[:3].tolist(),flush=True)
lib=C.CDLL(str(r/'norm_probe.so'));lib.run.argtypes=[C.c_void_p]*7
native=torch.empty_like(out);native_res=torch.empty_like(res)
lib.run(C.c_void_p(torch.cuda.current_stream().cuda_stream),a.data_ptr(),b.data_ptr(),weight.data_ptr(),native_res.data_ptr(),native.data_ptr(),dbg.data_ptr());torch.cuda.synchronize()
print('native vs ptx',int((native!=out).sum()),'debug',dbg[:3].tolist(),flush=True)
print('native vs teacher',int((native!=read('next_norm')).sum()),flush=True)
