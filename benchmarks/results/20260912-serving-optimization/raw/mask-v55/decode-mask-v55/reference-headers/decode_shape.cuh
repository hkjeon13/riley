#pragma once
#include "prefill_shape_attention.cuh"
// One-row projection: independent original BF16 rounding chunks run in parallel.
// Each warp writes eight output columns. The merge retains the original order.
template<int N,int K,int Interval>
__global__ void decode_projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out){
 const int lane=threadIdx.x,g=lane/4,t=lane%4,base=blockIdx.x*8;
 constexpr int Chunk=Interval>0?Interval:K;
 int begin=blockIdx.y*Chunk,end=min(begin+Chunk,K);float d[4]={};
 #pragma unroll
 for(int depth=0;depth<Chunk;depth+=16){
  if(begin+depth>=end)break;
  int k=begin+depth;
  uint32_t a=*reinterpret_cast<const uint32_t*>(x+k+2*t),aa=*reinterpret_cast<const uint32_t*>(x+k+2*t+8);
  uint32_t b=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t),bb=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t+8);
  riley_prefill_shape::mma(d,a,a,aa,aa,b,bb);
 }
 if(g==0)for(int j=0;j<2;++j){
  auto v=__float2bfloat16_rn(d[j]);
  if constexpr(Interval>0)parts[blockIdx.y*N+base+2*t+j]=__bfloat162float(v);
  else out[base+2*t+j]=v;
 }
}
template<int N,int K,int Interval>
__global__ void decode_projection_merge(const float* parts,__nv_bfloat16* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;float value=0.;
 #pragma unroll
 for(int chunk=0;chunk<(K+Interval-1)/Interval;++chunk)value+=parts[chunk*N+i];
 out[i]=__float2bfloat16_rn(value);
}
template<int N,int K,int Interval>
inline void enqueue_decode_projection(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,float* parts){
 constexpr int chunk=Interval>0?Interval:K;
 constexpr int chunks=(K+chunk-1)/chunk;
 decode_projection_parts<N,K,Interval><<<dim3(N/8,chunks),32,0,stream>>>(x,w,parts,out);
 if constexpr(Interval>0)decode_projection_merge<N,K,Interval><<<(N+255)/256,256,0,stream>>>(parts,out);
}
namespace riley_decode_shape {
// Parallel QK score tiles, preserving the exact four K16 MMA recurrence.
__global__ void scores(const __nv_bfloat16* q,const __nv_bfloat16* k,float* result,const uint32_t* shape,const uint32_t* pages){
 int count=shape[1]+1,token=blockIdx.x*8,head=blockIdx.y;
 if(token>=count || count>4096)return;
 int lane=threadIdx.x,g=lane/4,t=lane%4;float d[4]={};
 #pragma unroll
 for(int depth=0;depth<64;depth+=16){
  auto* qp=q+head*64+depth;
  uint32_t a=riley_prefill_shape::pair(qp[2*t],qp[2*t+1]);
  uint32_t aa=riley_prefill_shape::pair(qp[2*t+8],qp[2*t+9]);
  int kb=token+g<count?riley_prefill_shape::cache_index(token+g,head/3,depth,pages):0;
  uint32_t b=token+g<count?riley_prefill_shape::pair(k[kb+2*t],k[kb+2*t+1]):0;
  uint32_t bb=token+g<count?riley_prefill_shape::pair(k[kb+2*t+8],k[kb+2*t+9]):0;
  riley_prefill_shape::mma(d,a,a,aa,aa,b,bb);
 }
 if(g==0)for(int j=0;j<2;++j)if(token+2*t+j<count)result[head*4096+token+2*t+j]=d[j]*.125F;
}
// Independent output eight-column groups share the precomputed score buffer.
// Tile order, lane-local denominator and BF16 probability rounding are unchanged.
__global__ void values(const float* scores,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages){
 int count=shape[1]+1,head=blockIdx.y,block=blockIdx.x,lane=threadIdx.x,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 __shared__ __nv_bfloat16 probs[128];float maximum=-CUDART_INF_F,den=0.,accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[head*4096+begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  for(int i=lane;i<128;i+=32)probs[i]=__float2bfloat16_rn(i<end-begin?riley_prefill_shape::exponential(scores[head*4096+begin+i],mx):0.);
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)local_den+=riley_prefill_shape::exponential(scores[head*4096+begin+i],mx);}
  den=local_den;maximum=mx;__syncwarp();
  for(int j=0;j<4;++j)accum[j]*=alpha;
  for(int token=begin;token<end;token+=16){
   int pi=token-begin;uint32_t a=riley_prefill_shape::pair(probs[pi+2*t],probs[pi+2*t+1]);
   uint32_t aa=riley_prefill_shape::pair(probs[pi+2*t+8],probs[pi+2*t+9]);
   int dim=block*8+g;auto val=[&](int pos){return pos<end?v[riley_prefill_shape::cache_index(pos,head/3,dim,pages)]:zero;};
   uint32_t b=riley_prefill_shape::pair(val(token+2*t),val(token+2*t+1));
   uint32_t bb=riley_prefill_shape::pair(val(token+2*t+8),val(token+2*t+9));
   riley_prefill_shape::mma(accum,a,a,aa,aa,b,bb);
  }
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);float inverse=1.0F/den;
 if(g==0)for(int j=0;j<2;++j)out[head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}
inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scratch,const uint32_t* shape,const uint32_t* pages,uint32_t physical){
 scores<<<dim3((min(physical*16,4096u)+7)/8,9),32,0,stream>>>(q,k,scratch,shape,pages);
 values<<<dim3(8,9),32,0,stream>>>(scratch,v,out,shape,pages);
}
}
