#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include "riley_cuda.h"
namespace riley_compact_result {
constexpr uint32_t kThreads=256,kWarpSize=32,kFullWarpMask=0xffffffffU;
__device__ __forceinline__ void select_argmax_candidate(
    float candidate_value, uint32_t candidate_token, float* selected_value,
    uint32_t* selected_token) {
  if (candidate_token != RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID &&
      (*selected_token == RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID ||
       candidate_value > *selected_value ||
       (candidate_value == *selected_value &&
        candidate_token < *selected_token))) {
    *selected_value = candidate_value;
    *selected_token = candidate_token;
  }
}

__device__ __forceinline__ void reduce_argmax_warp(
    float* selected_value, uint32_t* selected_token,
    uint32_t* non_finite) {
  const uint32_t lane = threadIdx.x % kWarpSize;
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    const float candidate_value =
        __shfl_down_sync(kFullWarpMask, *selected_value, offset);
    const uint32_t candidate_token =
        __shfl_down_sync(kFullWarpMask, *selected_token, offset);
    const uint32_t candidate_non_finite =
        __shfl_down_sync(kFullWarpMask, *non_finite, offset);
    if (lane + offset < kWarpSize) {
      *non_finite |= candidate_non_finite;
      select_argmax_candidate(candidate_value, candidate_token,
                              selected_value, selected_token);
    }
  }
}

struct GreedyPartial {float value;uint32_t token,invalid,reserved;};
__device__ void finish_partial(float value,uint32_t token,uint32_t bad,GreedyPartial* output,float* values,uint32_t* tokens,uint32_t* invalid){
 const int lane=threadIdx.x%32,warp=threadIdx.x/32;
 reduce_argmax_warp(&value,&token,&bad);
 if(lane==0){values[warp]=value;tokens[warp]=token;invalid[warp]=bad;}
 __syncthreads();
 if(warp==0){
  value=lane<8?values[lane]:-CUDART_INF_F;token=lane<8?tokens[lane]:UINT32_MAX;bad=lane<8?invalid[lane]:0;
  reduce_argmax_warp(&value,&token,&bad);
  if(lane==0)*output={value,token,bad,0};
 }
 __syncthreads();
}

template<uint32_t Rows>
__global__ void parts(const uint32_t* metadata,const __nv_bfloat16* logits,GreedyPartial* partial){
 const uint32_t row=blockIdx.x/24,part=blockIdx.x%24,active=metadata[5];
 if(active<1||active>Rows||row>=active)return;
 const uint32_t* shape=metadata+32+row*416;
 if(shape[11]==0xffffffffU)return;
 __shared__ float values[8];__shared__ uint32_t tokens[8],invalid[8];
 float best=-CUDART_INF_F;uint32_t id=UINT32_MAX,bad=0;
 for(uint32_t column=part*2048+threadIdx.x;column<(part+1)*2048;column+=256){
  float x=__bfloat162float(logits[row*49152+column]);
  if(!isfinite(x))bad=1;else select_argmax_candidate(x,column,&best,&id);
 }
 finish_partial(best,id,bad,partial+row*24+part,values,tokens,invalid);
}
template<uint32_t Rows>
__global__ void finish(const uint32_t* m,const GreedyPartial* partial,const uint32_t* status,uint32_t* output){
 const uint32_t row=blockIdx.x,tid=threadIdx.x,active=m[5];uint32_t* result=output+row*32;
 if(active<1||active>Rows||row>=active){if(tid<32)result[tid]=0;return;}
 const uint32_t* shape=m+32+row*416;const bool published=shape[11]!=0xffffffffU;
 __shared__ float values[8];__shared__ uint32_t tokens[8],invalid[8];__shared__ GreedyPartial combined;
 float best=-CUDART_INF_F;uint32_t id=UINT32_MAX,bad=0;
 if(published&&tid<24){auto x=partial[row*24+tid];bad=x.invalid;best=x.value;id=x.token;}
 finish_partial(best,id,bad,&combined,values,tokens,invalid);
 if(tid==0){
  result[0]=*status;result[1]=published;result[2]=published?combined.token:0;
  result[3]=published?(combined.invalid||combined.token==UINT32_MAX):0;
  for(int i=0;i<6;++i)result[4+i]=m[10+i];
  for(int i=0;i<4;++i)result[10+i]=shape[12+i];
  result[14]=shape[5];result[15]=shape[8];result[16]=shape[6];result[17]=shape[2];result[18]=shape[9];result[19]=m[4];
  for(int i=0;i<8;++i)result[20+i]=m[16+i];
  result[28]=shape[1];result[29]=shape[10];result[30]=m[8];
  result[31]=(Rows==8?0x33524d52U:0x34524d52U)^0x80000000U;
 }
}
template<uint32_t Rows>
inline cudaError_t enqueue(cudaStream_t stream,const void* metadata,const void* logits,const uint32_t* status,void* partial,void* output){
 static_assert(Rows==8||Rows==16,"wire capacity");
 if(!metadata||!logits||!status||!partial||!output)return cudaErrorInvalidValue;
 parts<Rows><<<Rows*24,256,0,stream>>>(static_cast<const uint32_t*>(metadata),static_cast<const __nv_bfloat16*>(logits),static_cast<GreedyPartial*>(partial));
 auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
 finish<Rows><<<Rows,256,0,stream>>>(static_cast<const uint32_t*>(metadata),static_cast<const GreedyPartial*>(partial),status,static_cast<uint32_t*>(output));
 return cudaGetLastError();
}
}
