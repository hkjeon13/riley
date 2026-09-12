#include "ffi_internal.hpp"
// Diagnostic SM89 BF16 tensor-core attention, fixed head dimension 64.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef MODE
#define MODE 6
#endif
__device__ float exponential(float score,float maximum){
#if MODE == 1 || MODE >= 4
 return exp2f(fmaf(score,1.4426950408889634F,-maximum*1.4426950408889634F));
#elif MODE == 2
 float x=(score-maximum)*1.4426950408889634F,y;
 asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
#else
 return exp2f((score-maximum)*1.4426950408889634F);
#endif
}
__device__ uint32_t pair(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a)) | (uint32_t(__bfloat16_as_ushort(b))<<16);}
__device__ void mma(float* d,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ int cache_index(int token,int head,int dim,const uint32_t* blocks){
 return blocks ? ((blocks[token/16]*3+head)*16+token%16)*64+dim : (token*3+head)*64+dim;
}
__global__ void attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const int* dynamic_n,const uint32_t* blocks){
 if(dynamic_n)n=*dynamic_n+1;
 int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;
 int row=blockIdx.x,kvh=blockIdx.y,qh=kvh*3+warp,count=n-rows+row+1;
 int qb=(row*9+qh)*64;
 if(n<rows || n>160){out[qb+lane]=__float2bfloat16_rn(CUDART_NAN_F);out[qb+lane+32]=__float2bfloat16_rn(CUDART_NAN_F);return;}
 __shared__ float scores[3][128];
 __shared__ __nv_bfloat16 probs[3][128];
 float maximum=-CUDART_INF_F,den=0.;float accum[8][4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  for(int token=begin;token<end;token+=8){
   float d[4]={};
   for(int depth=0;depth<64;depth+=16){
    uint32_t a=pair(q[qb+depth+2*t],q[qb+depth+2*t+1]);
    uint32_t aa=pair(q[qb+depth+2*t+8],q[qb+depth+2*t+9]);
    int kb=token+group<end?cache_index(token+group,kvh,depth,blocks):0;
    uint32_t b=token+group<end?pair(k[kb+2*t],k[kb+2*t+1]):0;
    uint32_t bb=token+group<end?pair(k[kb+2*t+8],k[kb+2*t+9]):0;
    mma(d,a,a,aa,aa,b,bb);
   }
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[warp][token-begin+2*t+j]=d[j]*.125F;}
  }
  __syncwarp();
  float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[warp][i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  #if MODE >= 4
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
#else
  float alpha=exponential(maximum,mx);
#endif
  float local_den=0.;
  for(int i=lane;i<128;i+=32){
   float p=i<end-begin?exponential(scores[warp][i],mx):0.;
   local_den+=p;probs[warp][i]=__float2bfloat16_rn(p);
  }
  for(int offset=16;offset>0;offset>>=1)local_den+=__shfl_xor_sync(0xffffffff,local_den,offset);
#if MODE == 3 || MODE == 5 || MODE == 6
#if MODE == 6
  local_den=den*alpha;
#else
  local_den=0.;
#endif
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
    if(i<end-begin)local_den+=exponential(scores[warp][i],mx);}
#if MODE != 6
  local_den+=__shfl_xor_sync(0xffffffff,local_den,2);
  local_den+=__shfl_xor_sync(0xffffffff,local_den,1);
#endif
#endif
#if MODE == 6
  den=local_den;
#else
  den=den*alpha+local_den;
#endif
  maximum=mx;
  __syncwarp();
  for(int block=0;block<8;++block){
   for(int j=0;j<4;++j)accum[block][j]*=alpha;
   for(int token=begin;token<end;token+=16){
    int pi=token-begin;
    uint32_t a=pair(probs[warp][pi+2*t],probs[warp][pi+2*t+1]);
    uint32_t aa=pair(probs[warp][pi+2*t+8],probs[warp][pi+2*t+9]);
    int dim=block*8+group;
    auto val=[&](int pos){return pos<end?v[cache_index(pos,kvh,dim,blocks)]:zero;};
    uint32_t b=pair(val(token+2*t),val(token+2*t+1));
    uint32_t bb=pair(val(token+2*t+8),val(token+2*t+9));
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
  __syncwarp();
 }
#if MODE == 6
 den+=__shfl_xor_sync(0xffffffff,den,2);
 den+=__shfl_xor_sync(0xffffffff,den,1);
#endif
 float inverse=1.0F/den;
 if(group==0)for(int block=0;block<8;++block)for(int j=0;j<2;++j)
#if MODE >= 4
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
#else
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]/den);
#endif
}

namespace riley_cuda_internal {
cudaError_t enqueue_compiled_attention(cudaStream_t s,const void* q,const void* k,const void* v,void* out,const void* metadata) noexcept {
 auto* m=(const uint8_t*)metadata;attention<<<dim3(1,3),96,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,1,160,(const int*)(m+4),(const uint32_t*)(m+16));return cudaGetLastError();}
}

namespace riley_cuda_internal {
// The attention kernel already treats blockIdx.x as a row and uses
// count=(last_position+1)-rows+row+1 for the causal prefix. Only its launch
// geometry changes; the reduction and tensor-core accumulation stay intact.
cudaError_t enqueue_compiled_attention_rows(cudaStream_t s,const void* q,const void* k,const void* v,void* out,const void* metadata,uint32_t rows) noexcept {
 if(rows==0||rows>128)return cudaErrorInvalidValue;
 if(rows==1)return enqueue_compiled_attention(s,q,k,v,out,metadata);
 auto* m=(const uint8_t*)metadata;attention<<<dim3(rows,3),96,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,static_cast<int>(rows),160,(const int*)(m+4),(const uint32_t*)(m+16));return cudaGetLastError();
}
} // namespace riley_cuda_internal

// Packed M1 decode only. One warp owns one head and half of its output
// dimensions. Each output keeps the original MODE6 MMA and denominator order.
// The unchanged attention kernel above remains the M1 oracle and P128 path.
#if MODE == 6
__global__ void attention_packed_decode(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* metadata){
 const uint32_t position=metadata[1];
 const int lane=threadIdx.x,group=lane/4,t=lane%4;
 const int qh=blockIdx.x,kvh=qh/3,first_block=blockIdx.y*4,qb=qh*64;
 if(position<128||position>=160){out[qb+first_block*8+lane]=__float2bfloat16_rn(CUDART_NAN_F);return;}
 const int count=static_cast<int>(position)+1;
 const uint32_t* blocks=metadata+4;
 __shared__ float scores[128];
 __shared__ __nv_bfloat16 probs[128];
 float maximum=-CUDART_INF_F,den=0.;float accum[4][4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  for(int token=begin;token<end;token+=8){
   float d[4]={};
   for(int depth=0;depth<64;depth+=16){
    uint32_t a=pair(q[qb+depth+2*t],q[qb+depth+2*t+1]);
    uint32_t aa=pair(q[qb+depth+2*t+8],q[qb+depth+2*t+9]);
    int kb=token+group<end?cache_index(token+group,kvh,depth,blocks):0;
    uint32_t b=token+group<end?pair(k[kb+2*t],k[kb+2*t+1]):0;
    uint32_t bb=token+group<end?pair(k[kb+2*t+8],k[kb+2*t+9]):0;
    mma(d,a,a,aa,aa,b,bb);
   }
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[token-begin+2*t+j]=d[j]*.125F;}
  }
  __syncwarp();
  float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  for(int i=lane;i<128;i+=32){
   float p=i<end-begin?exponential(scores[i],mx):0.;
   probs[i]=__float2bfloat16_rn(p);
   // Scores are no longer needed after mx. Keep the unrounded float, since
   // summing the BF16 probabilities would change the reference denominator.
   scores[i]=p;
  }
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
    if(i<end-begin)local_den+=scores[i];}
  den=local_den;
  maximum=mx;
  __syncwarp();
  for(int block=0;block<4;++block){
   for(int j=0;j<4;++j)accum[block][j]*=alpha;
   for(int token=begin;token<end;token+=16){
    int pi=token-begin;
    uint32_t a=pair(probs[pi+2*t],probs[pi+2*t+1]);
    uint32_t aa=pair(probs[pi+2*t+8],probs[pi+2*t+9]);
    int dim=(first_block+block)*8+group;
    auto val=[&](int pos){return pos<end?v[cache_index(pos,kvh,dim,blocks)]:zero;};
    uint32_t b=pair(val(token+2*t),val(token+2*t+1));
    uint32_t bb=pair(val(token+2*t+8),val(token+2*t+9));
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);
 den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.0F/den;
 if(group==0)for(int block=0;block<4;++block)for(int j=0;j<2;++j)
  out[qb+(first_block+block)*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}
#endif

namespace riley_cuda_internal {
cudaError_t enqueue_compiled_packed_decode_attention(cudaStream_t s,const void* q,const void* k,const void* v,void* out,const void* metadata) noexcept {
#if MODE == 6
 attention_packed_decode<<<dim3(9,2),32,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,(const uint32_t*)metadata);return cudaGetLastError();
#else
 // This entry is qualified only for the exact MODE6 packed-decode contract.
 return cudaErrorInvalidValue;
#endif
}
} // namespace riley_cuda_internal

// Packed M1 decode only. The two warps partition independent QK token groups
// and PV output blocks; neither an individual MMA chain nor the MODE6 delayed
// denominator reduction is split. Both preceding kernels remain unchanged.
#if MODE == 6
__global__ void attention_packed_decode_two_warp(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* metadata){
 const uint32_t position=metadata[1];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;
 const int qh=blockIdx.x,kvh=qh/3,first_block=blockIdx.y*4+warp*2,qb=qh*64;
 if(position<128||position>=160){
  if(warp==0)out[qb+blockIdx.y*32+lane]=__float2bfloat16_rn(CUDART_NAN_F);
  return;
 }
 const int count=static_cast<int>(position)+1;
 const uint32_t* blocks=metadata+4;
 __shared__ float scores[128];
 __shared__ __nv_bfloat16 probs[128];
 __shared__ float alpha_by_lane[32],inverse_by_lane[32];
 float maximum=-CUDART_INF_F,den=0.;float accum[2][4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  for(int token=begin+8*warp;token<end;token+=16){
   float d[4]={};
   for(int depth=0;depth<64;depth+=16){
    uint32_t a=pair(q[qb+depth+2*t],q[qb+depth+2*t+1]);
    uint32_t aa=pair(q[qb+depth+2*t+8],q[qb+depth+2*t+9]);
    int kb=token+group<end?cache_index(token+group,kvh,depth,blocks):0;
    uint32_t b=token+group<end?pair(k[kb+2*t],k[kb+2*t+1]):0;
    uint32_t bb=token+group<end?pair(k[kb+2*t+8],k[kb+2*t+9]):0;
    mma(d,a,a,aa,aa,b,bb);
   }
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[token-begin+2*t+j]=d[j]*.125F;}
  }
  // Includes the idle second warp when the tail has at most eight tokens.
  __syncthreads();
  if(warp==0){
   float mx=maximum;
   for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[i]);
   for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
   float alpha=exp2f((maximum-mx)*1.4426950408889634F);
   alpha_by_lane[lane]=alpha;
   for(int i=lane;i<128;i+=32){
    float p=i<end-begin?exponential(scores[i],mx):0.;
    probs[i]=__float2bfloat16_rn(p);
    scores[i]=p;
   }
   __syncwarp();
   // Warp zero retains the accepted per-lane j/z addition order and keeps
   // the original FP32 probabilities, not their BF16 PV representation.
   float local_den=den*alpha;
   for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
     if(i<end-begin)local_den+=scores[i];}
   den=local_den;
   maximum=mx;
  }
  __syncthreads(); // Publish probabilities and the corresponding lane's alpha.
  const float alpha=alpha_by_lane[lane];
  for(int block=0;block<2;++block){
   for(int j=0;j<4;++j)accum[block][j]*=alpha;
   for(int token=begin;token<end;token+=16){
    int pi=token-begin;
    uint32_t a=pair(probs[pi+2*t],probs[pi+2*t+1]);
    uint32_t aa=pair(probs[pi+2*t+8],probs[pi+2*t+9]);
    int dim=(first_block+block)*8+group;
    auto val=[&](int pos){return pos<end?v[cache_index(pos,kvh,dim,blocks)]:zero;};
    uint32_t b=pair(val(token+2*t),val(token+2*t+1));
    uint32_t bb=pair(val(token+2*t+8),val(token+2*t+9));
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
  __syncthreads(); // Finish both PV readers before reusing scores/probabilities.
 }
 if(warp==0){
  den+=__shfl_xor_sync(0xffffffff,den,2);
  den+=__shfl_xor_sync(0xffffffff,den,1);
  inverse_by_lane[lane]=1.0F/den;
 }
 __syncthreads();
 const float inverse=inverse_by_lane[lane];
 if(group==0)for(int block=0;block<2;++block)for(int j=0;j<2;++j)
  out[qb+(first_block+block)*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}
#endif

namespace riley_cuda_internal {
cudaError_t enqueue_compiled_packed_decode_attention_two_warp(cudaStream_t s,const void* q,const void* k,const void* v,void* out,const void* metadata) noexcept {
#if MODE == 6
 attention_packed_decode_two_warp<<<dim3(9,2),64,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,(const uint32_t*)metadata);return cudaGetLastError();
#else
 return cudaErrorInvalidValue;
#endif
}
} // namespace riley_cuda_internal

// Packed M1 decode only. Fold precise RoPE and current K/V publication into
// the accepted two-warp attention mapping. The complete preceding kernels
// remain available as independent numerical oracles and the P128 path.
#if MODE == 6
// These explicit non-FTZ operations preserve the precise RoPE SASS order
// under this translation unit's --use_fast_math flags, including signed zero.
// Packed table conversion and the mixed FMA pair passed the isolated v2 probe.
__device__ __forceinline__ float fused_rope_table(float value){
 uint32_t packed;
 asm volatile("cvt.rn.bf16x2.f32 %0, 0f00000000, %1;" : "=r"(packed) : "f"(value));
 return __uint_as_float(packed<<16);
}
__device__ __forceinline__ __nv_bfloat16 fused_rope_round(float value){
 uint16_t bits;
 asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
 return __ushort_as_bfloat16(bits);
}
__device__ __forceinline__ float fused_rope_mul(float a,float b){
 float value;
 asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a),"f"(b));
 return value;
}
__device__ __forceinline__ float fused_rope_fma(float a,float b,float c){
 float value;
 asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(value) : "f"(a),"f"(b),"f"(c));
 return value;
}
__device__ __forceinline__ float fused_rope_negate(float value){
 return __uint_as_float(__float_as_uint(value)^0x80000000u);
}
// All callers below prove four-byte alignment and preserve adjacent BF16 bits.
__device__ __forceinline__ uint32_t fused_history_shared_pair(const __nv_bfloat16* p){
 const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(p));
 uint32_t bits;asm volatile("ld.shared.b32 %0, [%1];" : "=r"(bits) : "r"(address) : "memory");return bits;
}
__device__ __forceinline__ uint32_t fused_history_global_pair(const __nv_bfloat16* p){
 const uint64_t address=static_cast<uint64_t>(__cvta_generic_to_global(p));
 uint32_t bits;asm volatile("ld.global.b32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");return bits;
}
// Exactly two tiles for positions128..159: tail first, then full history.
// History specialization has no current-token test, raw-V bypass or tail mask.
template<bool History>
__device__ __forceinline__ void fused_history_tile(
 const __nv_bfloat16* local_q,const __nv_bfloat16* local_k,const __nv_bfloat16* raw_v,
 const __nv_bfloat16* k,const __nv_bfloat16* v,const uint32_t* blocks,uint32_t position,
 float* scores,__nv_bfloat16* probs,float* alpha_by_lane,
 int lane,int warp,int group,int t,int kvh,int first_block,
 float& maximum,float& den,float (&accum)[2][4]){
 const int begin=History?0:128,end=History?128:static_cast<int>(position)+1;
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
  for(int token=begin+8*warp;token<end;token+=16){
   float d[4]={};
   const int key_token=token+group;
   int key_base=0;bool current=false;
   if constexpr(History){
    key_base=((blocks[token/16]*3+kvh)*16+token%16+group)*64;
   }else{
    current=key_token==static_cast<int>(position);
    if(key_token<end&&!current)key_base=cache_index(key_token,kvh,0,blocks);
   }
   for(int depth=0;depth<64;depth+=16){
    uint32_t a=fused_history_shared_pair(local_q+depth+2*t);
    uint32_t aa=fused_history_shared_pair(local_q+depth+2*t+8);
    uint32_t b=0,bb=0;
    if constexpr(History){
     b=fused_history_global_pair(k+key_base+depth+2*t);
     bb=fused_history_global_pair(k+key_base+depth+2*t+8);
    }else if(key_token<end){
     if(current){
      b=fused_history_shared_pair(local_k+depth+2*t);
      bb=fused_history_shared_pair(local_k+depth+2*t+8);
     }else{
      b=fused_history_global_pair(k+key_base+depth+2*t);
      bb=fused_history_global_pair(k+key_base+depth+2*t+8);
     }
    }
    mma(d,a,a,aa,aa,b,bb);
   }
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[token-begin+2*t+j]=d[j]*.125F;}
  }
  // Includes the idle second warp when the tail has at most eight tokens.
  __syncthreads();
  if(warp==0){
   float mx=maximum;
   for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[i]);
   for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
   float alpha=exp2f((maximum-mx)*1.4426950408889634F);
   alpha_by_lane[lane]=alpha;
   for(int i=lane;i<128;i+=32){
    float p=i<end-begin?exponential(scores[i],mx):0.;
    probs[i]=__float2bfloat16_rn(p);
    scores[i]=p;
   }
   __syncwarp();
   // Warp zero retains the accepted per-lane j/z addition order and keeps
   // the original FP32 probabilities, not their BF16 PV representation.
   float local_den=den*alpha;
   for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
     if(i<end-begin)local_den+=scores[i];}
   den=local_den;
   maximum=mx;
  }
  __syncthreads(); // Publish probabilities and the corresponding lane's alpha.
  const float alpha=alpha_by_lane[lane];
  for(int block=0;block<2;++block){
   for(int j=0;j<4;++j)accum[block][j]*=alpha;
   const int dim=(first_block+block)*8+group;
   for(int token=begin;token<end;token+=16){
    int pi=token-begin;
    uint32_t a=fused_history_shared_pair(probs+pi+2*t);
    uint32_t aa=fused_history_shared_pair(probs+pi+2*t+8);
    uint32_t b,bb;
    if constexpr(History){
     // Every 16-token historical iteration stays within one physical page.
     const int page=((blocks[token/16]*3+kvh)*16)*64+dim;
     b=pair(v[page+(2*t)*64],v[page+(2*t+1)*64]);
     bb=pair(v[page+(2*t+8)*64],v[page+(2*t+9)*64]);
    }else{
     auto val=[&](int pos){
      if(pos>=end)return zero;
      return pos==static_cast<int>(position)?raw_v[kvh*64+dim]:v[cache_index(pos,kvh,dim,blocks)];
     };
     b=pair(val(token+2*t),val(token+2*t+1));
     bb=pair(val(token+2*t+8),val(token+2*t+9));
    }
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
  __syncthreads(); // Finish both PV readers before reusing scores/probabilities.
}
__global__ void attention_packed_multisequence_rope(
 const __nv_bfloat16* raw_q,const __nv_bfloat16* raw_k,const __nv_bfloat16* raw_v,
 __nv_bfloat16* qo,__nv_bfloat16* k,__nv_bfloat16* v,__nv_bfloat16* out,
 const float* cos,const float* sin,const uint32_t* metadata){
 const uint32_t row=blockIdx.z;
 if(row>=metadata[6]){
  if(threadIdx.x<32){const uint32_t at=row*576+blockIdx.x*64+blockIdx.y*32+threadIdx.x;qo[at]=__ushort_as_bfloat16(0);out[at]=__ushort_as_bfloat16(0);}
  return;
 }
 raw_q+=row*960;raw_k+=row*960;raw_v+=row*960;qo+=row*576;out+=row*576;
 metadata+=32+row*32;
 const uint32_t position=metadata[1];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;
 const int qh=blockIdx.x,kvh=qh/3,first_block=blockIdx.y*4+warp*2,qb=qh*64;
 if(position<128||position>=160){
  if(warp==0)out[qb+blockIdx.y*32+lane]=__float2bfloat16_rn(CUDART_NAN_F);
  return;
 }
 const uint32_t* blocks=metadata+4;
 __shared__ float scores[128];
 __shared__ __align__(4) __nv_bfloat16 probs[128];
 __shared__ float alpha_by_lane[32],inverse_by_lane[32];
 __shared__ __align__(4) __nv_bfloat16 local_q[64],local_k[64];
 // Each CTA has its own rounded current Q/K. No CTA reads the cache slot
 // another CTA is publishing during this launch, so no grid barrier is needed.
 const float c=fused_rope_table(cos[position*32+lane]);
 const float s=fused_rope_table(sin[position*32+lane]);
 const int source_base=warp==0?qb:kvh*64;
 const __nv_bfloat16* source=warp==0?raw_q:raw_k;
 const float ra=__uint_as_float(uint32_t(__bfloat16_as_ushort(source[source_base+lane]))<<16);
 const float rb=__uint_as_float(uint32_t(__bfloat16_as_ushort(source[source_base+lane+32]))<<16);
 const float first=fused_rope_fma(ra,c,fused_rope_negate(fused_rope_mul(rb,s)));
 const float second=fused_rope_fma(ra,s,fused_rope_mul(rb,c));
 const __nv_bfloat16 x=fused_rope_round(first),y=fused_rope_round(second);
 __nv_bfloat16* local=warp==0?local_q:local_k;
 local[lane]=x;local[lane+32]=y;
 // Preserve the original rotary-Q scratch and publish K/V exactly once per
 // KV head. Every current-token attention load below uses local/raw data.
 if(warp==0&&blockIdx.y==0){qo[qb+lane]=x;qo[qb+lane+32]=y;}
 if(warp==1&&qh%3==0&&blockIdx.y==0){
  const int destination=cache_index(static_cast<int>(position),kvh,lane,blocks);
  k[destination]=x;k[destination+32]=y;
  v[destination]=raw_v[kvh*64+lane];v[destination+32]=raw_v[kvh*64+lane+32];
 }
 __syncthreads(); // Publish both local heads, including warp1's rounded K.
 float maximum=-CUDART_INF_F,den=0.;float accum[2][4]={};
 // The validated range guarantees tail128..position followed by history0..127.
 fused_history_tile<false>(local_q,local_k,raw_v,k,v,blocks,position,scores,probs,alpha_by_lane,
     lane,warp,group,t,kvh,first_block,maximum,den,accum);
 fused_history_tile<true>(local_q,local_k,raw_v,k,v,blocks,position,scores,probs,alpha_by_lane,
     lane,warp,group,t,kvh,first_block,maximum,den,accum);

 if(warp==0){
  den+=__shfl_xor_sync(0xffffffff,den,2);
  den+=__shfl_xor_sync(0xffffffff,den,1);
  inverse_by_lane[lane]=1.0F/den;
 }
 __syncthreads();
 const float inverse=inverse_by_lane[lane];
 if(group==0)for(int block=0;block<2;++block)for(int j=0;j<2;++j)
  out[qb+(first_block+block)*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}

__global__ void attention_packed_decode_rope(
 const __nv_bfloat16* raw_q,const __nv_bfloat16* raw_k,const __nv_bfloat16* raw_v,
 __nv_bfloat16* qo,__nv_bfloat16* k,__nv_bfloat16* v,__nv_bfloat16* out,
 const float* cos,const float* sin,const uint32_t* metadata){
 const uint32_t position=metadata[1];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;
 const int qh=blockIdx.x,kvh=qh/3,first_block=blockIdx.y*4+warp*2,qb=qh*64;
 if(position<128||position>=160){
  if(warp==0)out[qb+blockIdx.y*32+lane]=__float2bfloat16_rn(CUDART_NAN_F);
  return;
 }
 const uint32_t* blocks=metadata+4;
 __shared__ float scores[128];
 __shared__ __align__(4) __nv_bfloat16 probs[128];
 __shared__ float alpha_by_lane[32],inverse_by_lane[32];
 __shared__ __align__(4) __nv_bfloat16 local_q[64],local_k[64];
 // Each CTA has its own rounded current Q/K. No CTA reads the cache slot
 // another CTA is publishing during this launch, so no grid barrier is needed.
 const float c=fused_rope_table(cos[position*32+lane]);
 const float s=fused_rope_table(sin[position*32+lane]);
 const int source_base=warp==0?qb:kvh*64;
 const __nv_bfloat16* source=warp==0?raw_q:raw_k;
 const float ra=__uint_as_float(uint32_t(__bfloat16_as_ushort(source[source_base+lane]))<<16);
 const float rb=__uint_as_float(uint32_t(__bfloat16_as_ushort(source[source_base+lane+32]))<<16);
 const float first=fused_rope_fma(ra,c,fused_rope_negate(fused_rope_mul(rb,s)));
 const float second=fused_rope_fma(ra,s,fused_rope_mul(rb,c));
 const __nv_bfloat16 x=fused_rope_round(first),y=fused_rope_round(second);
 __nv_bfloat16* local=warp==0?local_q:local_k;
 local[lane]=x;local[lane+32]=y;
 // Preserve the original rotary-Q scratch and publish K/V exactly once per
 // KV head. Every current-token attention load below uses local/raw data.
 if(warp==0&&blockIdx.y==0){qo[qb+lane]=x;qo[qb+lane+32]=y;}
 if(warp==1&&qh%3==0&&blockIdx.y==0){
  const int destination=cache_index(static_cast<int>(position),kvh,lane,blocks);
  k[destination]=x;k[destination+32]=y;
  v[destination]=raw_v[kvh*64+lane];v[destination+32]=raw_v[kvh*64+lane+32];
 }
 __syncthreads(); // Publish both local heads, including warp1's rounded K.
 float maximum=-CUDART_INF_F,den=0.;float accum[2][4]={};
 // The validated range guarantees tail128..position followed by history0..127.
 fused_history_tile<false>(local_q,local_k,raw_v,k,v,blocks,position,scores,probs,alpha_by_lane,
     lane,warp,group,t,kvh,first_block,maximum,den,accum);
 fused_history_tile<true>(local_q,local_k,raw_v,k,v,blocks,position,scores,probs,alpha_by_lane,
     lane,warp,group,t,kvh,first_block,maximum,den,accum);

 if(warp==0){
  den+=__shfl_xor_sync(0xffffffff,den,2);
  den+=__shfl_xor_sync(0xffffffff,den,1);
  inverse_by_lane[lane]=1.0F/den;
 }
 __syncthreads();
 const float inverse=inverse_by_lane[lane];
 if(group==0)for(int block=0;block<2;++block)for(int j=0;j<2;++j)
  out[qb+(first_block+block)*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}
#endif

namespace riley_cuda_internal {
cudaError_t enqueue_compiled_packed_multisequence_rope_attention(cudaStream_t stream,
 const void* packed,void* qo,void* key,void* value,void* out,const void* cos,const void* sin,const void* packet,uint32_t bucket) noexcept {
#if MODE == 6
 const auto* q=static_cast<const __nv_bfloat16*>(packed);
 attention_packed_multisequence_rope<<<dim3(9,2,bucket),64,0,stream>>>(q,q+576,q+768,
  static_cast<__nv_bfloat16*>(qo),static_cast<__nv_bfloat16*>(key),static_cast<__nv_bfloat16*>(value),static_cast<__nv_bfloat16*>(out),
  static_cast<const float*>(cos),static_cast<const float*>(sin),static_cast<const uint32_t*>(packet));
 return cudaGetLastError();
#else
 return cudaErrorNotSupported;
#endif
}
cudaError_t enqueue_compiled_packed_decode_rope_attention(cudaStream_t s,
 const void* q,const void* k,const void* v,void* qo,void* keys,void* values,void* out,
 const void* cos,const void* sin,const void* metadata) noexcept {
#if MODE == 6
 attention_packed_decode_rope<<<dim3(9,2),64,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,
  (__nv_bfloat16*)qo,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(__nv_bfloat16*)out,
  (const float*)cos,(const float*)sin,(const uint32_t*)metadata);return cudaGetLastError();
#else
 return cudaErrorInvalidValue;
#endif
}
} // namespace riley_cuda_internal
