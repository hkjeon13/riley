#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace riley_shared16_result {
constexpr uint32_t record_bytes=98432,batch_bytes=16*record_bytes;
// Each record follows descriptor row order; its output slot is explicit.
template<uint32_t Magic>
__global__ void finish(const uint32_t* m,const __nv_bfloat16* logits,const uint32_t* status,uint8_t* bytes){
 uint32_t row=blockIdx.x,active=m[5];if(active<1||active>16||row>=active)return;
 auto* shape=m+32+row*416;auto* result=reinterpret_cast<uint32_t*>(bytes+row*record_bytes);
 auto* out=reinterpret_cast<__nv_bfloat16*>(bytes+row*record_bytes+128);
 uint32_t tid=threadIdx.x;bool published=shape[11]!=0xffffffffu;
 __shared__ float maxima[256];__shared__ uint32_t tokens[256],errors[256];
 float maximum=-CUDART_INF_F;uint32_t token=0,failed=0;
 for(uint32_t i=tid;i<49152;i+=256){auto v=published?logits[row*49152+i]:__float2bfloat16_rn(0.F);out[i]=v;float f=__bfloat162float(v);failed|=!isfinite(f);if(f>maximum||(f==maximum&&i<token)){maximum=f;token=i;}}
 maxima[tid]=maximum;tokens[tid]=token;errors[tid]=failed;__syncthreads();
 for(uint32_t step=128;step;step>>=1){if(tid<step){float other=maxima[tid+step];uint32_t id=tokens[tid+step];if(other>maxima[tid]||(other==maxima[tid]&&id<tokens[tid])){maxima[tid]=other;tokens[tid]=id;}errors[tid]|=errors[tid+step];}__syncthreads();}
 if(tid==0){
 result[0]=*status;result[1]=published;result[2]=published?tokens[0]:0;result[3]=errors[0];
 for(int i=0;i<6;++i)result[4+i]=m[10+i];
 for(int i=0;i<4;++i)result[10+i]=shape[12+i];
 result[14]=shape[5];result[15]=shape[8];result[16]=shape[6];result[17]=shape[2];result[18]=shape[9];result[19]=m[4];
 for(int i=0;i<8;++i)result[20+i]=m[16+i];
 result[28]=shape[1];result[29]=shape[10];result[30]=m[8];result[31]=Magic;
 }
}
template<uint32_t Magic=0x33524d52>
inline cudaError_t enqueue(cudaStream_t stream,const void* metadata,const __nv_bfloat16* logits,const uint32_t* status,void* result){
 if(!metadata||!logits||!status||!result)return cudaErrorInvalidValue;
 auto e=cudaMemsetAsync(result,0,batch_bytes,stream);if(e!=cudaSuccess)return e;
 finish<Magic><<<16,256,0,stream>>>(static_cast<const uint32_t*>(metadata),logits,status,static_cast<uint8_t*>(result));return cudaGetLastError();
}
}
