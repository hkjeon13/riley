#pragma once
#include "compact_mixed_attention.cuh"
// Native candidate: pair adjacent 8-query tasks while preserving the recorded
// tile-map ABI. Each surviving CTA reuses K/V across sixteen causal queries.
namespace riley_query_reuse {
template<bool Compact,bool Audit=false>
__global__ void mapped(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta,unsigned* coverage=nullptr){
 const unsigned active=meta[5],total=meta[9],tiles=meta[24],tile=blockIdx.x;
 if(!active||active>32||!total||total>capacity||!tiles||tiles>total||tile>=tiles)return;
 const unsigned entry=meta[32+32*416+1024+tile],owner=entry>>16,local=entry&0xffffU;
 if(owner>=active)return;
 const unsigned* shape=meta+32+owner*416;
 const unsigned count=shape[2],offset=shape[16],nt=count<32?count:(count+7)/8;
 if(!count||offset>total||count>total-offset||local>=nt)return;
 if(count>=32&&(local&1))return;
 if constexpr(Audit)if(threadIdx.x==0){
  atomicAdd(coverage+tile*9+blockIdx.y,1);
  // Metadata is caller-validated: an owner's local task list is contiguous.
  if(count>=32&&local+1<nt)atomicAdd(coverage+(tile+1)*9+blockIdx.y,1);
 }
 const unsigned query_block=count<32?local:local/2;
 if constexpr(Compact){
  __shared__ float scores[16][128];
  riley_compact_mixed::attention_body<16>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,query_block,blockIdx.y,scores);
 }else{
  riley_mixed_attention::attention_body<16>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,query_block);
 }
}
}
