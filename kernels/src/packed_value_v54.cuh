#pragma once
#include <cuda_bf16.h>
#include <stdint.h>
// V7 owned sessions retain this page-local V layout across mixed and decode graphs.
// Same pool extent as token-major BF16; K remains token-major.
namespace riley_packed_value_v54 {
__device__ __forceinline__ int packed_index(int pos,int head,int dim,const uint32_t* pages){
 int page=pages?pages[pos/16]:pos/16,t=pos%16;
 return (page*3+head)*1024+(dim/8)*128+(t/8)*64+(dim%8)*8+t%8;
}
__device__ __forceinline__ uint32_t masked_pair(const __nv_bfloat16* p,int pos,int end){
 if(pos>=end)return 0;uint32_t bits=*reinterpret_cast<const uint32_t*>(p);return pos+1<end?bits:bits&0xffffU;
}
}
