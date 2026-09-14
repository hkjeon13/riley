#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace riley_speculative_select {
constexpr uint64_t bytes=32ULL*49152*2;
// Existing V7 validation remains mandatory; this is an additional narrow gate.
__host__ __device__ inline bool eligible(const uint32_t* meta,uint32_t capacity) {
 if((meta[4]!=0 && meta[4]!=3) || meta[5]<1 || meta[5]>4 || meta[9]<1 || meta[9]>capacity)return false;
 unsigned offset=0;
 for(unsigned owner=0;owner<meta[5];++owner){const auto* row=meta+32+owner*416;
  if(row[18]!=meta[4] || row[2]<1 || row[2]>8 || row[16]!=offset)return false;
  offset+=row[2];
 }
 return offset==meta[9] && offset<=32;
}
// Preserve each owner's usual last-query slot. Additional query positions fill
// slots after the owners, allowing the existing 32-row head GEMM to verify all.
__global__ void gather(const __nv_bfloat16* hidden,__nv_bfloat16* selected,
                       const uint32_t* meta,unsigned capacity,const uint32_t* status) {
 if(!eligible(meta,capacity))return;
 unsigned slot=blockIdx.x,active=meta[5],source=0;bool valid=false;
 if(slot<active){const auto* row=meta+32+slot*416;source=row[16]+row[2]-1;valid=true;}
 else {unsigned begin=active;for(unsigned owner=0;owner<active;++owner){const auto* row=meta+32+owner*416;
  if(slot>=begin && slot<begin+row[2]-1){source=row[16]+slot-begin;valid=true;break;}
  begin+=row[2]-1;
 }}
 for(unsigned i=threadIdx.x;i<576;i+=blockDim.x)
  selected[slot*576+i]=valid&&*status==0?hidden[source*576+i]:__float2bfloat16_rn(0.F);
}
}

#include <math_constants.h>
namespace riley_speculative_select {
struct GreedyRecord {uint32_t token,error,slot,valid;};
constexpr uint64_t greedy_bytes=32*sizeof(GreedyRecord);
static_assert(greedy_bytes==512,"Rust verification record contract");
// Same greedy tie rule as the ordinary target: lowest vocabulary ID wins.
__global__ void greedy(const __nv_bfloat16* logits,GreedyRecord* output,
                        const uint32_t* meta,unsigned capacity,const uint32_t* status) {
 unsigned slot=blockIdx.x,tid=threadIdx.x;
 bool valid=eligible(meta,capacity) && slot<meta[9];
 if(!valid){if(tid==0)output[slot]={0,0,slot,0};return;}
 __shared__ float maxima[256];__shared__ unsigned tokens[256],errors[256];
 float maximum=-CUDART_INF_F;unsigned token=0xffffffffu,error=*status;
 for(unsigned i=tid;i<49152;i+=256){float value=__bfloat162float(logits[slot*49152+i]);
  error|=!isfinite(value);if(value>maximum || (value==maximum && i<token)){maximum=value;token=i;}
 }
 maxima[tid]=maximum;tokens[tid]=token;errors[tid]=error;__syncthreads();
 for(unsigned step=128;step;step>>=1){if(tid<step){float other=maxima[tid+step];unsigned id=tokens[tid+step];
  if(other>maxima[tid] || (other==maxima[tid] && id<tokens[tid])){maxima[tid]=other;tokens[tid]=id;}
  errors[tid]|=errors[tid+step];}__syncthreads();}
 if(tid==0)output[slot]={tokens[0],errors[0],slot,1};
}
}
