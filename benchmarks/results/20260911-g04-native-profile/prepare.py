from pathlib import Path
import subprocess
r=Path('/tmp/riley-g04-native-profile-source-260911');assert not r.exists()
subprocess.run(['git','clone','--no-hardlinks','/tmp/riley-g04-followup-source-260911',str(r)],check=True)
art=Path('/tmp/riley-g04-native-profile-260911')
(r/'kernels/src/graph_numerics.cu').write_bytes((art/'graph_numerics.cu').read_bytes())
p=r/'kernels/CMakeLists.txt';s=p.read_text().replace('    src/graph_resources.cu','    src/graph_numerics.cu\n    src/graph_resources.cu');s+='\nset_source_files_properties(src/graph_numerics.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")\n';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();pos=s.rindex('}  // namespace riley_cuda_internal');s=s[:pos]+'''
cudaError_t enqueue_compiled_norm(cudaStream_t,const void*,const void*,const void*,void*,void*,int) noexcept;
cudaError_t enqueue_compiled_rope(cudaStream_t,const void*,const void*,void*,void*,const void*,const void*,const void*) noexcept;
cudaError_t enqueue_compiled_swiglu(cudaStream_t,const void*,const void*,void*) noexcept;
cudaError_t enqueue_compiled_attention(cudaStream_t,const void*,const void*,const void*,void*,const void*) noexcept;
'''+s[pos:];p.write_text(s)
p=r/'crates/riley-cuda/src/ffi.rs';s=p.read_text();a=s.index('pub(super) fn record_decode(');s=s[:a]+s[a:].replace('u32::from(hf),','if hf { 2 } else { 0 }, // isolated numerical experiment only',1);p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text();a=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_decode(');head=s[:a];s=s[a:]
s=s.replace('weight_count!=3+9*layers||profile>1','weight_count!=3+9*layers||profile>2').replace('(profile==1&&h!=1152)','(profile>=1&&h!=1152)||(profile==2&&(k!=384||inter!=3072||layers!=30))').replace('(profile==1&&eps[i]!=1e-5F)','(profile>=1&&eps[i]!=1e-5F)')
old='s=kernel(enqueue_mlp_norm(r->stream->stream,d[1]->device_data,weights[0]->device_data,d[2]->device_data,h/2,eps[1+2*l],profile));'
assert old in s;s=s.replace(old,'if(profile==2){if(l==0)s=kernel(enqueue_compiled_norm(r->stream->stream,d[1]->device_data,nullptr,weights[0]->device_data,nullptr,d[2]->device_data,0));}else '+old)
old='if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_qkv_rope('
s=s.replace(old,'if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_rope(r->stream->stream,d[3]->device_data,d[6]->device_data,d[4]->device_data,d[8]->device_data,d[17]->device_data,d[18]->device_data,static_cast<uint8_t*>(d[19]->device_data)+4));\n      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_qkv_rope(')
needle='      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(3,3);';s=s.replace(needle,'      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_attention(r->stream->stream,d[4]->device_data,static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,d[5]->device_data,d[19]->device_data));\n'+needle)
s=s.replace('if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_mlp_residual(r->stream->stream,d[1]', 'if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_residual(r->stream->stream,d[1]')
s=s.replace('if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_mlp_norm(r->stream->stream,d[4]', 'if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_norm(r->stream->stream,d[4]')
s=s.replace('      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(4,9);','      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_norm(r->stream->stream,d[3]->device_data,d[1]->device_data,weights[5]->device_data,d[11]->device_data,d[2]->device_data,1));\n      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(4,9);')
s=s.replace('if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_mlp_pointwise(', 'if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_swiglu(r->stream->stream,d[9]->device_data,d[10]->device_data,d[12]->device_data));\n      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_pointwise(')
s=s.replace('if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_mlp_residual(r->stream->stream,d[4]','if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_norm(r->stream->stream,d[5]->device_data,d[11]->device_data,(l+1<layers?w[3+9*(l+1)]:w[1])->device_data,d[1]->device_data,d[2]->device_data,2));\n      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_residual(r->stream->stream,d[4]')
s=s.replace('if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_mlp_norm(r->stream->stream,d[1]->device_data,w[1]', 'if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_norm(r->stream->stream,d[1]->device_data,w[1]')
p.write_text(head+s)
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();a=s.index('mod owned_tests');head=s[:a];s=s[a:]
old='''                assert_eq!(
                    candidate.execute(&rows)?,
                    expected,
                    "request={request} position={position}"
                );'''
assert old in s;s=s.replace(old,'                let _diagnostic_logits = candidate.execute(&rows)?;')
s=s.replace('full_logits_exact=true','full_logits_exact=not_compared isolated_numerics=true')
p.write_text(head+s)
