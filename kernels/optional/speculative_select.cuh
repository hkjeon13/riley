#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace riley_speculative_select {
constexpr uint64_t bytes=32ULL*49152*2;
// Existing V7 validation remains mandatory; this is an additional narrow gate.
__host__ __device__ inline bool eligible(const uint32_t* meta,uint32_t capacity) {
 if(meta[4]!=0 || meta[5]<1 || meta[5]>4 || meta[9]<1 || meta[9]>capacity)return false;
 unsigned offset=0;
 for(unsigned owner=0;owner<meta[5];++owner){const auto* row=meta+32+owner*416;
  if(row[18]!=0 || row[2]<1 || row[2]>8 || row[16]!=offset)return false;
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
