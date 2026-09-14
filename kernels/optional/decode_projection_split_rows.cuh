#pragma once
#include "decode_adaptive_rows.cuh"
// Native experiment only: split the two M16 tiles across independent warps.
// Keep each K16 recurrence, BF16 partial boundary and 32-row scratch stride.
namespace riley_projection_split {
template<int N,int K,int Interval,bool Tiled>
__device__ __forceinline__ void dispatch(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,const uint32_t* live,int col,int chunk,unsigned base=(threadIdx.x/32)*16) {
 const unsigned rows=*live;
 if(rows<1||rows>32||base>=rows)return;
 const unsigned count=min(rows-base,16u);
 riley_adaptive_decode::projection_compute<N,K,Interval,Tiled,false,1>(x+base*K,w,parts+base*N,nullptr,&count,col,chunk);
}
template<int N,int K,int Interval,bool Tiled>
__global__ void projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,const uint32_t* live) {
 dispatch<N,K,Interval,Tiled>(x,w,parts,live,blockIdx.x,blockIdx.y);
}
__global__ void qkv_parts(const __nv_bfloat16* x,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* parts,const uint32_t* live) {
 int col=blockIdx.x;
 if(col<72)dispatch<576,576,192,false>(x,q,parts,live,col,blockIdx.y);
 else dispatch<192,576,192,false>(x,col<96?k:v,parts+3*32*576+(col<96?0:3*32*192),live,col<96?col-72:col-96,blockIdx.y);
}
}

namespace riley_projection_ctas {
template<int N,int K,int Interval,bool Tiled>
__global__ void projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,const uint32_t* live) {
 riley_projection_split::dispatch<N,K,Interval,Tiled>(x,w,parts,live,blockIdx.x,blockIdx.y,blockIdx.z*16);
}
__global__ void qkv_parts(const __nv_bfloat16* x,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* parts,const uint32_t* live) {
 int col=blockIdx.x;
 if(col<72)riley_projection_split::dispatch<576,576,192,false>(x,q,parts,live,col,blockIdx.y,blockIdx.z*16);
 else riley_projection_split::dispatch<192,576,192,false>(x,col<96?k:v,parts+3*32*576+(col<96?0:3*32*192),live,col<96?col-72:col-96,blockIdx.y,blockIdx.z*16);
}
}
