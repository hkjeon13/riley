from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();s+='''
__global__ void v3_prefill_result_header(const uint32_t* m,uint32_t* result,const uint32_t* publish,const uint32_t* argmax){
 if(threadIdx.x)return;
 result[1]=*publish;result[2]=argmax[0];result[3]=argmax[1];
 for(int i=0;i<6;++i)result[4+i]=m[10+i]; // generation, replay, iteration
 for(int i=0;i<4;++i)result[10+i]=m[44+i]; // request and reservation cookie
 result[14]=m[37];result[15]=m[40];result[16]=m[38];result[17]=m[34];
 result[18]=m[41];result[19]=m[4];for(int i=0;i<8;++i)result[20+i]=m[16+i];
 result[28]=m[33];result[29]=m[42];result[30]=m[8];result[31]=0x33524d52;
}
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v3_result_header(cudaStream_t stream,const void* metadata,void* status,const void* publish,const void* argmax) noexcept {
 v3_prefill_result_header<<<1,32,0,stream>>>(static_cast<const uint32_t*>(metadata),static_cast<uint32_t*>(status),static_cast<const uint32_t*>(publish),static_cast<const uint32_t*>(argmax));return cudaGetLastError();
}
}
''';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();s+='\nnamespace riley_cuda_internal { cudaError_t enqueue_compiled_v3_result_header(cudaStream_t,const void*,void*,const void*,const void*) noexcept; }\n';p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('d[18]->byte_len!=4||d[19]->byte_len!=4','d[18]->byte_len!=128||d[19]->byte_len!=4');start=s.index('  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[18]->device_data,4,');end=s.index('  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer+128',start);s=s[:start]+'''  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_result_header(r->stream->stream,d[16]->device_data,d[18]->device_data,d[19]->device_data,d[21]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 completion header");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[18]->device_data,128,cudaMemcpyDeviceToHost);
'''+s[end:];p.write_text(s)
for name in ['v3_recorder_gpu.rs','v3_recorder_checkpoint_gpu.rs']:
 p=r/'crates/riley-cuda/tests'/name;s=p.read_text().replace('17536,1152,4,4,98304,8','17536,1152,128,4,98304,8')
 if name=='v3_recorder_gpu.rs':s=s.replace('assert!(out[8..].iter().all(|&x|x==0));','assert!(out[8..16].iter().all(|&x|x==0));assert!(out[128..].iter().all(|&x|x==0));assert_eq!(&out[24..32],&((i as u64)+1).to_le_bytes());assert_eq!(&out[40..48],&1u64.to_le_bytes());assert_eq!(&out[124..128],&0x33524d52u32.to_le_bytes());')
 else:
  s=s.replace('assert!(out[16..128].iter().all(|&x|x==0));','assert_eq!(&out[24..32],&replay_count.to_le_bytes());assert_eq!(&out[56..60],&((start+n)as u32).to_le_bytes());assert_eq!(&out[60..64],&(tokens.len()as u32).to_le_bytes());assert_eq!(&out[124..128],&0x33524d52u32.to_le_bytes());')
  s=s.replace('assert_eq!(out,*expected,"GPU logits/argmax/output mismatch");','assert_eq!(&out[..16],&expected[..16],"status/argmax mismatch");assert_eq!(&out[128..],&expected[128..],"GPU logits mismatch");')
 p.write_text(s)
