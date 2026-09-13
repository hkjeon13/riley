#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Diagnostic precursor to two-plan scheduling. The retained owner must supply
// canonical identities and retain previous results until this stream completes.
// Packet validation, KV reservation, EOS/cancel settlement remain caller duties.
namespace riley_future_token {
constexpr unsigned Rows=32, PacketWords=15392, ResultWords=32;
constexpr unsigned Failure=64, HostToken=UINT32_MAX;
struct Reference { unsigned source_row; unsigned expected[ResultWords]; };
static_assert(sizeof(Reference)==132);
__host__ __device__ inline uint64_t pair(const unsigned* p){return uint64_t(p[0])|(uint64_t(p[1])<<32);}
__host__ __device__ inline bool next(uint64_t a,uint64_t b){return a!=UINT64_MAX && b==a+1;}

// Only word2 (the not-yet-known token) is excluded from canonical comparison.
__device__ bool valid(const unsigned* m,unsigned row,const Reference* refs,const unsigned* previous){
 const auto& ref=refs[row]; const auto* s=m+32+row*416;
 if(ref.source_row==HostToken)return true;
 if(ref.source_row>=Rows || s[18]!=1 || s[2]!=1 || s[11]!=0 || s[0]!=0)return false;
 const auto* r=previous+ref.source_row*ResultWords;
 if(r[0] || r[1]!=1 || r[3] || r[2]>=49152 || r[31]!=0xb7524d52U || r[30]!=0)return false;
 for(unsigned i=0;i<ResultWords;++i)if(i!=2 && r[i]!=ref.expected[i])return false;
 if(!pair(r+4) || pair(r+4)!=pair(m+10) || !next(pair(r+6),pair(m+12)) || !next(pair(r+8),pair(m+14)))return false;
 for(unsigned i=0;i<8;++i)if(r[20+i]!=m[16+i])return false;
 if(!pair(r+10)||!pair(r+12)||pair(r+10)!=pair(s+12)||pair(r+12)!=pair(s+14))return false;
 if(r[14]!=s[4] || r[15]!=s[8] || r[18]!=s[9] || !next(r[16],s[6]) || !next(r[14],s[5]))return false;
 if(s[5]==0 || s[1]!=s[5]-1 || r[28]+1!=r[14])return false;
 if(m[4]!=1 && (s[16]>=m[9] || m[13344+s[16]]!=0))return false;
 for(unsigned other=0;other<row;++other)
  if(refs[other].source_row==ref.source_row)return false;
 return true;
}
__global__ void resolve(unsigned* m,const Reference* refs,const unsigned* previous,unsigned* status){
 __shared__ unsigned bad;
 if(threadIdx.x==0)bad=*status;
 __syncthreads();
 unsigned row=threadIdx.x;
 bool header=m[0]==0x37444d52U && m[1]==7 && m[2]==PacketWords*4 && m[3]==1664 &&
             m[4]<=2 && m[5]>0 && m[5]<=Rows && m[8]==0 && (m[4]==1 || (m[9]>0 && m[9]<=1024));
 if(!header || (row<m[5] && !valid(m,row,refs,previous)))atomicOr(&bad,Failure);
 __syncthreads();
 // Reject the whole batch before publishing even one replacement token.
 if(bad){if(row==0)atomicOr(status,bad);return;}
 if(row<m[5] && refs[row].source_row!=HostToken){
  auto* s=m+32+row*416;unsigned token=previous[refs[row].source_row*ResultWords+2];
  s[0]=token;if(m[4]!=1)m[13344+s[16]]=token;
 }
}
inline cudaError_t enqueue(cudaStream_t stream,unsigned* packet,const Reference* refs,const unsigned* previous,unsigned* status){
 if(!packet||!refs||!previous||!status)return cudaErrorInvalidValue;
 resolve<<<1,32,0,stream>>>(packet,refs,previous,status);return cudaGetLastError();
}
}
