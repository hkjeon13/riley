#pragma once
#include "decode_tiled.cuh"
// One MMA shares weights across up to sixteen independent decode rows.
// Live rows are supplied by validated metadata; inactive rows never load/store.
template<int N,int K,int Interval,bool Tiled,bool Compact=false>
__device__ __forceinline__ void shared16_projection_compute(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out,const uint32_t* live_rows,int column,int chunk_index){
 const uint32_t rows=*live_rows;if(rows<1||rows>16)return;
 const int lane=threadIdx.x%32,g=lane/4,t=lane%4,base=column*8;
 constexpr int Chunk=Interval>0?Interval:K;
 int begin=chunk_index*Chunk,end=min(begin+Chunk,K);float d[4]={};
 // Hoist row/tile bases and issue four independent load groups before the
 // ordered MMA recurrence. Bound code size instead of fully unrolling K.
 const auto* xp=x+g*K+begin+2*t;
 const auto* wp=Tiled?w+column*(K/16)*128+(begin/16)*128+lane*2:w+(base+g)*K+begin+2*t;
 #pragma unroll 1
 for(int depth=0;depth<Chunk;depth+=64){
  uint32_t a[4],aa[4],a1[4],a3[4],b[4],bb[4];
  #pragma unroll
  for(int step=0;step<4;++step){
   int offset=depth+step*16;bool valid=begin+offset<end;
   a[step]=valid&&g<rows?*reinterpret_cast<const uint32_t*>(xp+offset):0;
   aa[step]=valid&&g<rows?*reinterpret_cast<const uint32_t*>(xp+offset+8):0;
   a1[step]=valid&&g+8<rows?*reinterpret_cast<const uint32_t*>(xp+8*K+offset):0;
   a3[step]=valid&&g+8<rows?*reinterpret_cast<const uint32_t*>(xp+8*K+offset+8):0;
   if constexpr(Tiled){
    b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128):0;
    bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128+64):0;
   }else{
    b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset):0;
    bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset+8):0;
   }
  }
  #pragma unroll
  for(int step=0;step<4;++step)if(begin+depth+step*16<end)riley_prefill_shape::mma(d,a[step],a1[step],aa[step],a3[step],b[step],bb[step]);
 }

 for(int half=0;half<2;++half){int row=g+half*8;if(row>=rows)continue;
  for(int j=0;j<2;++j){auto v=__float2bfloat16_rn(d[half*2+j]);
   if constexpr(Interval>0)parts[(chunk_index*16+row)*N+base+2*t+j]=__bfloat162float(v);
   else if constexpr(Compact)out[row*8+2*t+j]=v;
   else out[row*N+base+2*t+j]=v;
  }
 }

}

template<int N,int K,int Interval,bool Tiled>
__global__ void shared16_projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out,const uint32_t* live_rows){
 shared16_projection_compute<N,K,Interval,Tiled>(x,w,parts,out,live_rows,blockIdx.x,blockIdx.y);
}
// Independent projections share one dispatch while retaining each MMA order.
__global__ void shared16_gate_up(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* g,__nv_bfloat16* u,const uint32_t* live){
 shared16_projection_compute<1536,576,0,true>(x,blockIdx.y?up:gate,nullptr,blockIdx.y?u:g,live,blockIdx.x,0);
}
__global__ void shared16_qkv_parts(const __nv_bfloat16* x,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* parts,const uint32_t* live){
 int col=blockIdx.x;
 if(col<72)shared16_projection_compute<576,576,192,false>(x,q,parts,nullptr,live,col,blockIdx.y);
 else shared16_projection_compute<192,576,192,false>(x,col<96?k:v,parts+3*16*576+(col<96?0:3*16*192),nullptr,live,col<96?col-72:col-96,blockIdx.y);
}
__global__ void shared16_qkv_merge(const float* parts,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,const uint32_t* live){
 uint32_t rows=*live;if(rows<1||rows>16)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*960)return;
 int row=i/960,col=i%960,n=col<576?576:192;
 const float* src=col<576?parts:parts+3*16*576+(col<768?0:3*16*192);
 __nv_bfloat16* out=col<576?q:(col<768?k:v);col=col<576?col:(col<768?col-576:col-768);
 float value=0.;for(int c=0;c<3;++c)value+=src[c*16*n+row*n+col];out[row*n+col]=__float2bfloat16_rn(value);
}
inline void enqueue_shared16_qkv(cudaStream_t s,const __nv_bfloat16* x,const __nv_bfloat16* qw,const __nv_bfloat16* kw,const __nv_bfloat16* vw,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,float* parts,const uint32_t* active){
 shared16_qkv_parts<<<dim3(120,3),32,0,s>>>(x,qw,kw,vw,parts,active);
 shared16_qkv_merge<<<60,256,0,s>>>(parts,q,k,v,active);
}

template<int N,int K,int Interval>
__global__ void shared16_projection_merge(const float* parts,__nv_bfloat16* out,const uint32_t* live_rows){
 uint32_t rows=*live_rows;if(rows<1||rows>16)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*N)return;float value=0.;
 #pragma unroll
 for(int chunk=0;chunk<(K+Interval-1)/Interval;++chunk)value+=parts[chunk*16*N+i];
 out[i]=__float2bfloat16_rn(value);
}
template<int N,int K,int Interval,bool Tiled>
inline void enqueue_shared16_projection(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,float* parts,const uint32_t* live_rows){
 constexpr int chunk=Interval>0?Interval:K;
 shared16_projection_parts<N,K,Interval,Tiled><<<dim3(N/8,(K+chunk-1)/chunk),32,0,stream>>>(x,w,parts,out,live_rows);
 if constexpr(Interval>0)shared16_projection_merge<N,K,Interval><<<(16*N+255)/256,256,0,stream>>>(parts,out,live_rows);
}

// Two warps compute independent gate/up tiles, then share rounded results.
template<bool Tiled>
__global__ void shared16_gate_up_swiglu(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,const uint32_t* live){
 uint32_t rows=*live;if(rows<1||rows>16)return;
 int warp=threadIdx.x/32;__shared__ __nv_bfloat16 partials[2][128];
 shared16_projection_compute<1536,576,0,Tiled,true>(x,warp?up:gate,nullptr,partials[warp],live,blockIdx.x,0);
 __syncthreads();
 for(int i=threadIdx.x;i<128;i+=blockDim.x){int row=i/8,col=i%8;if(row<rows){float g=__bfloat162float(partials[0][i]);out[row*1536+blockIdx.x*8+col]=__float2bfloat16_rn((g/(1.F+expf(-g)))*__bfloat162float(partials[1][i]));}}
}
