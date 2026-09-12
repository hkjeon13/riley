#include "ffi_internal.hpp"
#include <cuda_bf16.h>
namespace riley_cuda_internal {
namespace {
__global__ void multi_embedding(const __nv_bfloat16* weights, const uint32_t* packet,
                               __nv_bfloat16* hidden) {
  const unsigned row=blockIdx.x, active=packet[6];
  const unsigned token=row<active?packet[32+row*32]:0;
  for(unsigned i=threadIdx.x;i<576;i+=blockDim.x)
    hidden[row*576+i]=(row<active&&token<49152)?weights[token*576+i]:__ushort_as_bfloat16(0);
}
// Explicit wire encoding. All rows and reserved bytes are initialized on every
// replay, including transitions from a larger active count to a smaller one.
__global__ void multi_completion(const uint32_t* packet, const uint32_t* argmax,
                                const uint16_t* logits, uint32_t* output,
                                unsigned bucket, unsigned full) {
  const unsigned bytes=1152+(full?bucket*98304:0);
  if(blockIdx.x != 0) {
    const unsigned word=(blockIdx.x-1)*blockDim.x+threadIdx.x;
    if(word < bucket*24576)
      output[288+word]=word < packet[6]*24576 ? reinterpret_cast<const uint32_t*>(logits)[word] : 0;
    return;
  }
  for(unsigned i=threadIdx.x;i<288;i+=blockDim.x)output[i]=0;
  __syncthreads();
  if(threadIdx.x==0){
    for(unsigned i=0;i<32;++i)output[i]=packet[i];
    output[0]=0x314f4d52; output[2]=bytes; output[27]=0; output[28]=full?bucket:0;
    for(unsigned r=0;r<packet[6];++r){
      const uint32_t* in=packet+32+r*32; uint32_t* out=output+32+r*32;
      out[0]=in[20];out[1]=in[21];out[2]=in[22];out[3]=in[23];
      out[4]=packet[14];out[5]=packet[15];out[6]=packet[16];out[7]=packet[17];
      out[8]=r;out[9]=in[25];out[10]=in[1];out[11]=in[27];
      out[12]=argmax[2*r];out[13]=argmax[2*r+1];out[14]=32;
      out[22]=argmax[2*r+1]==0?1:2;
      if(argmax[2*r+1]!=0)output[27]=1;
      if(full){out[24]=1152+r*98304;out[26]=98304;}
    }
  }

}
}
cudaError_t enqueue_multi_embedding(cudaStream_t stream,const void* weights,
    const void* packet,void* hidden,uint32_t bucket) noexcept {
  multi_embedding<<<bucket,256,0,stream>>>(static_cast<const __nv_bfloat16*>(weights),
      static_cast<const uint32_t*>(packet),static_cast<__nv_bfloat16*>(hidden));
  return cudaGetLastError();
}
cudaError_t enqueue_multi_completion(cudaStream_t stream,const void* packet,
    const void* argmax,const void* logits,void* output,uint32_t bucket,uint32_t full) noexcept {
  multi_completion<<<1+(full?bucket*96:0),256,0,stream>>>(static_cast<const uint32_t*>(packet),
      static_cast<const uint32_t*>(argmax),static_cast<const uint16_t*>(logits),
      static_cast<uint32_t*>(output),bucket,full);
  return cudaGetLastError();
}
}
