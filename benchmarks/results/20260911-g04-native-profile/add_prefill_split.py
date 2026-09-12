from pathlib import Path
r=Path('/tmp/riley-g04-native-profile-source-260911');art=Path('/tmp/riley-g04-native-profile-260911')
s=(art/'gemv_split.cu').read_text();s=s[:s.index('extern "C" void run')].replace('void gemv(', 'void gemv_prefill(').replace('int n,int k,int interval){','int n,int k,int interval,const uint32_t* position){\n if(*position>=128)return;')
s+='''
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_prefill_gemm(cudaStream_t s,const void* x,const void* w,void* y,int n,int k,int interval,const void* pos) noexcept {
 gemv_prefill<<<(n+31)/32,128,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,n,k,interval,(const uint32_t*)pos);return cudaGetLastError();
}
}
'''
p=r/'kernels/src/graph_numerics_precise.cu';p.write_text(p.read_text()+'\n'+s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();pos=s.rindex('}  // namespace riley_cuda_internal');s=s[:pos]+'cudaError_t enqueue_compiled_prefill_gemm(cudaStream_t,const void*,const void*,void*,int,int,int,const void*) noexcept;\n'+s[pos:];p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text();old='auto gemm=[&](size_t j,size_t out){return enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[out],states[l*7+j],error,"decode GEMM");};';assert old in s
s=s.replace(old,'''auto gemm=[&](size_t j,size_t out){
        auto status=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[out],states[l*7+j],error,"decode GEMM");
        if(status==RILEY_CUDA_STATUS_SUCCESS&&profile==2){
          const int ns[7]={576,192,192,576,1536,1536,576};const int ks[7]={576,576,576,576,576,576,1536};const int chunks[7]={192,192,192,128,0,0,320};
          const size_t inputs[7]={2,2,2,5,2,2,12};const size_t weight_ids[7]={1,2,3,4,6,7,8};
          status=kernel(enqueue_compiled_prefill_gemm(r->stream->stream,d[inputs[j]]->device_data,weights[weight_ids[j]]->device_data,d[out]->device_data,ns[j],ks[j],chunks[j],static_cast<uint8_t*>(d[19]->device_data)+4));
        }
        return status;
      };''').replace('native-trace-own-final.bin','native-trace-split-prefill.bin');p.write_text(s)
