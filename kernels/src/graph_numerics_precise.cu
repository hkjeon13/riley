#include "ffi_internal.hpp"
#include <cuda_bf16.h>
__global__ void compiled_norm(const __nv_bfloat16* a,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode){
 __shared__ float sums[8];int tid=threadIdx.x,lane=tid%32;float x[4]={};
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576){x[j]=__bfloat162float(a[i]);
  if(mode==1)x[j]+=__bfloat162float(((const __nv_bfloat16*)b)[i]);if(mode==2)x[j]+=((const float*)b)[i];
  if(mode==1)((float*)residual)[i]=x[j];if(mode==2)((__nv_bfloat16*)residual)[i]=__float2bfloat16_rn(x[j]);}}
 float sum=x[1]*x[1];sum=fmaf(x[0],x[0],sum);sum=fmaf(x[2],x[2],sum);sum=fmaf(x[3],x[3],sum);
 for(int step=16;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);
 if(lane==0)sums[tid/32]=sum;__syncthreads();
 if(tid<32){sum=lane<8?sums[lane]:0.;for(int step=4;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);if(lane==0)sums[0]=sum;}__syncthreads();
 float mean=fmaf(sums[0],1.F/576.F,1e-5F),inv;
 asm("rsqrt.approx.ftz.f32 %0, %1;":"=f"(inv):"f"(mean));
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576)out[i]=__float2bfloat16_rn((x[j]*inv)*__bfloat162float(w[i]));}
}

__global__ void compiled_rope(const __nv_bfloat16* q,const __nv_bfloat16* k,__nv_bfloat16* qo,__nv_bfloat16* ko,const float* cos,const float* sin,const uint32_t* pos){
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[*pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[*pos*32+dim]));
 const __nv_bfloat16* src=head<9?q:k;__nv_bfloat16* dst=head<9?qo:ko;int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 dst[base+dim]=__float2bfloat16_rn(a*c-b*s);dst[base+dim+32]=__float2bfloat16_rn(b*c+a*s);
}
__global__ void compiled_swiglu(const __nv_bfloat16* g,const __nv_bfloat16* u,__nv_bfloat16* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536){float x=__bfloat162float(g[i]);out[i]=__float2bfloat16_rn((x/(1.F+expf(-x)))*__bfloat162float(u[i]));}
}
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_norm(cudaStream_t s,const void* a,const void* b,const void* w,void* residual,void* out,int mode) noexcept {
 compiled_norm<<<1,256,0,s>>>((const __nv_bfloat16*)a,b,(const __nv_bfloat16*)w,residual,(__nv_bfloat16*)out,mode);return cudaGetLastError();}
cudaError_t enqueue_compiled_rope(cudaStream_t s,const void* q,const void* k,void* qo,void* ko,const void* cos,const void* sin,const void* pos) noexcept {
 compiled_rope<<<2,256,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(__nv_bfloat16*)qo,(__nv_bfloat16*)ko,(const float*)cos,(const float*)sin,(const uint32_t*)pos);return cudaGetLastError();}
cudaError_t enqueue_compiled_swiglu(cudaStream_t s,const void* g,const void* u,void* out) noexcept {
 compiled_swiglu<<<6,256,0,s>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,(__nv_bfloat16*)out);return cudaGetLastError();}
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ uint32_t pack(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a))|(uint32_t(__bfloat16_as_ushort(b))<<16);}
__global__ void gemv_prefill(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,int n,int k,int interval,const uint32_t* position){
 if(*position>=128)return;
 int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4,base=(blockIdx.x*4+warp)*8;
 if(base>=n)return;float d[4]={},total[4]={};
 for(int depth=0;depth<k;depth+=16){
  uint32_t a=pack(x[depth+2*t],x[depth+2*t+1]),aa=pack(x[depth+2*t+8],x[depth+2*t+9]);
  uint32_t b=pack(w[(base+g)*k+depth+2*t],w[(base+g)*k+depth+2*t+1]),bb=pack(w[(base+g)*k+depth+2*t+8],w[(base+g)*k+depth+2*t+9]);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a),"r"(a),"r"(aa),"r"(aa),"r"(b),"r"(bb));
  if(interval>0&&((depth+16)%interval==0||depth+16==k)){for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.;}}
 }
 if(interval>0)for(int j=0;j<4;++j)d[j]=total[j];
 if(g==0){y[base+2*t]=__float2bfloat16_rn(d[0]);y[base+2*t+1]=__float2bfloat16_rn(d[1]);}
}

namespace riley_cuda_internal {
cudaError_t enqueue_compiled_prefill_gemm(cudaStream_t s,const void* x,const void* w,void* y,int n,int k,int interval,const void* pos) noexcept {
 gemv_prefill<<<(n+31)/32,128,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,n,k,interval,(const uint32_t*)pos);return cudaGetLastError();
}
}

__global__ void compiled_kv_write(const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* keys,__nv_bfloat16* values,const uint32_t* metadata){
 int i=threadIdx.x;if(i>=192)return;uint32_t pos=metadata[1],physical=metadata[4+pos/16];
 int destination=((physical*3+i/64)*16+pos%16)*64+i%64;keys[destination]=k[i];values[destination]=v[i];
}
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_kv_write(cudaStream_t s,const void* k,const void* v,void* keys,void* values,const void* metadata) noexcept {
 compiled_kv_write<<<1,256,0,s>>>((const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(const uint32_t*)metadata);return cudaGetLastError();
}
}

// Row-parallel prefill operators. The original M=1 kernels and launches above
// remain untouched; rows==1 delegates to those entrypoints. Each new CTA/warp
// performs exactly one original row reduction, with only pointer offsets added.
// The graph owner validates end_position+1 >= rows and all parent extents.
__global__ void compiled_norm_rows(const __nv_bfloat16* a,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode){
 const uint32_t row=blockIdx.x;
 a+=row*576;out+=row*576;
 if(mode==1){b=static_cast<const __nv_bfloat16*>(b)+row*576;residual=static_cast<float*>(residual)+row*576;}
 if(mode==2){b=static_cast<const float*>(b)+row*576;residual=static_cast<__nv_bfloat16*>(residual)+row*576;}

 __shared__ float sums[8];int tid=threadIdx.x,lane=tid%32;float x[4]={};
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576){x[j]=__bfloat162float(a[i]);
  if(mode==1)x[j]+=__bfloat162float(((const __nv_bfloat16*)b)[i]);if(mode==2)x[j]+=((const float*)b)[i];
  if(mode==1)((float*)residual)[i]=x[j];if(mode==2)((__nv_bfloat16*)residual)[i]=__float2bfloat16_rn(x[j]);}}
 float sum=x[1]*x[1];sum=fmaf(x[0],x[0],sum);sum=fmaf(x[2],x[2],sum);sum=fmaf(x[3],x[3],sum);
 for(int step=16;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);
 if(lane==0)sums[tid/32]=sum;__syncthreads();
 if(tid<32){sum=lane<8?sums[lane]:0.;for(int step=4;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);if(lane==0)sums[0]=sum;}__syncthreads();
 float mean=fmaf(sums[0],1.F/576.F,1e-5F),inv;
 asm("rsqrt.approx.ftz.f32 %0, %1;":"=f"(inv):"f"(mean));
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576)out[i]=__float2bfloat16_rn((x[j]*inv)*__bfloat162float(w[i]));}
}
__global__ void compiled_rope_rows(const __nv_bfloat16* q,const __nv_bfloat16* k,__nv_bfloat16* qo,__nv_bfloat16* ko,const float* cos,const float* sin,const uint32_t* end_position,uint32_t rows){
 const uint32_t row=blockIdx.y,end=*end_position;
 if(end>=160||end+1<rows)return;
 const uint32_t position=end+1-rows+row;
 const uint32_t* pos=&position;
 q+=row*576;qo+=row*576;k+=row*192;ko+=row*192;

 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[*pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[*pos*32+dim]));
 const __nv_bfloat16* src=head<9?q:k;__nv_bfloat16* dst=head<9?qo:ko;int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 dst[base+dim]=__float2bfloat16_rn(a*c-b*s);dst[base+dim+32]=__float2bfloat16_rn(b*c+a*s);
}
__global__ void compiled_swiglu_rows(const __nv_bfloat16* g,const __nv_bfloat16* u,__nv_bfloat16* out){
 const uint32_t row=blockIdx.y;g+=row*1536;u+=row*1536;out+=row*1536;

 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536){float x=__bfloat162float(g[i]);out[i]=__float2bfloat16_rn((x/(1.F+expf(-x)))*__bfloat162float(u[i]));}
}
__global__ void gemv_prefill_rows(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,int n,int k,int interval,const uint32_t* position){
 const uint32_t row=blockIdx.y;x+=row*k;y+=row*n;

 if(*position>=128)return;
 int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4,base=(blockIdx.x*4+warp)*8;
 if(base>=n)return;float d[4]={},total[4]={};
 for(int depth=0;depth<k;depth+=16){
  uint32_t a=pack(x[depth+2*t],x[depth+2*t+1]),aa=pack(x[depth+2*t+8],x[depth+2*t+9]);
  uint32_t b=pack(w[(base+g)*k+depth+2*t],w[(base+g)*k+depth+2*t+1]),bb=pack(w[(base+g)*k+depth+2*t+8],w[(base+g)*k+depth+2*t+9]);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a),"r"(a),"r"(aa),"r"(aa),"r"(b),"r"(bb));
  if(interval>0&&((depth+16)%interval==0||depth+16==k)){for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.;}}
 }
 if(interval>0)for(int j=0;j<4;++j)d[j]=total[j];
 if(g==0){y[base+2*t]=__float2bfloat16_rn(d[0]);y[base+2*t+1]=__float2bfloat16_rn(d[1]);}
}
__global__ void compiled_kv_write_rows(const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* keys,__nv_bfloat16* values,const uint32_t* metadata,uint32_t rows){
 const uint32_t row=blockIdx.x,end=metadata[1];
 if(end>=160||end+1<rows)return;
 k+=row*192;v+=row*192;

 int i=threadIdx.x;if(i>=192)return;uint32_t pos=end+1-rows+row,physical=metadata[4+pos/16];
 int destination=((physical*3+i/64)*16+pos%16)*64+i%64;keys[destination]=k[i];values[destination]=v[i];
}

// Packed M1 decode only. Preserve compiled_rope's BF16 table rounding and
// expression order, then publish the rounded K directly to its paged cache.
// Each K thread also copies the matching two raw V values without conversion.
// The aggregate owner validates the fresh position, physical map and extents.
__global__ void compiled_packed_decode_rope_kv(
    const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
    __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,
    const float* cos,const float* sin,const uint32_t* metadata){
 const uint32_t* pos=metadata+1;
 if(*pos<128||*pos>=160)return;
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[*pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[*pos*32+dim]));
 const __nv_bfloat16* src=head<9?q:k;int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 const __nv_bfloat16 first=__float2bfloat16_rn(a*c-b*s),second=__float2bfloat16_rn(b*c+a*s);
 if(head<9){qo[base+dim]=first;qo[base+dim+32]=second;}
 else{
  const uint32_t physical=metadata[4+*pos/16];
  const uint32_t destination=((physical*3+head-9)*16+*pos%16)*64+dim;
  keys[destination]=first;keys[destination+32]=second;
  values[destination]=v[base+dim];values[destination+32]=v[base+dim+32];
 }
}

namespace riley_cuda_internal {
cudaError_t enqueue_compiled_norm_rows(cudaStream_t s,const void* a,const void* b,const void* w,void* residual,void* out,int mode,uint32_t rows) noexcept {
 if(rows==0||rows>128)return cudaErrorInvalidValue;
 if(rows==1)return enqueue_compiled_norm(s,a,b,w,residual,out,mode);
 compiled_norm_rows<<<rows,256,0,s>>>((const __nv_bfloat16*)a,b,(const __nv_bfloat16*)w,residual,(__nv_bfloat16*)out,mode);return cudaGetLastError();
}
cudaError_t enqueue_compiled_rope_rows(cudaStream_t s,const void* q,const void* k,void* qo,void* ko,const void* cos,const void* sin,const void* end_position,uint32_t rows) noexcept {
 if(rows==0||rows>128)return cudaErrorInvalidValue;
 if(rows==1)return enqueue_compiled_rope(s,q,k,qo,ko,cos,sin,end_position);
 compiled_rope_rows<<<dim3(2,rows),256,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(__nv_bfloat16*)qo,(__nv_bfloat16*)ko,(const float*)cos,(const float*)sin,(const uint32_t*)end_position,rows);return cudaGetLastError();
}
cudaError_t enqueue_compiled_swiglu_rows(cudaStream_t s,const void* g,const void* u,void* out,uint32_t rows) noexcept {
 if(rows==0||rows>128)return cudaErrorInvalidValue;
 if(rows==1)return enqueue_compiled_swiglu(s,g,u,out);
 compiled_swiglu_rows<<<dim3(6,rows),256,0,s>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,(__nv_bfloat16*)out);return cudaGetLastError();
}
cudaError_t enqueue_compiled_prefill_gemm_rows(cudaStream_t s,const void* x,const void* w,void* y,int n,int k,int interval,const void* end_position,uint32_t rows) noexcept {
 if(rows==0||rows>128)return cudaErrorInvalidValue;
 if(rows==1)return enqueue_compiled_prefill_gemm(s,x,w,y,n,k,interval,end_position);
 gemv_prefill_rows<<<dim3((n+31)/32,rows),128,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,n,k,interval,(const uint32_t*)end_position);return cudaGetLastError();
}
cudaError_t enqueue_compiled_kv_write_rows(cudaStream_t s,const void* k,const void* v,void* keys,void* values,const void* metadata,uint32_t rows) noexcept {
 if(rows==0||rows>128)return cudaErrorInvalidValue;
 if(rows==1)return enqueue_compiled_kv_write(s,k,v,keys,values,metadata);
 compiled_kv_write_rows<<<rows,256,0,s>>>((const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(const uint32_t*)metadata,rows);return cudaGetLastError();
}
cudaError_t enqueue_compiled_packed_decode_rope_kv(
    cudaStream_t s,const void* q,const void* k,const void* v,void* qo,
    void* keys,void* values,const void* cos,const void* sin,const void* metadata) noexcept {
 compiled_packed_decode_rope_kv<<<2,256,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)qo,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(const float*)cos,(const float*)sin,(const uint32_t*)metadata);return cudaGetLastError();
}
} // namespace riley_cuda_internal

// Fixed P128 prefill: populate all 16 MMA rows instead of duplicating one row.
// Each output retains the original depth order and BF16 chunk-round sequence.
// The unchanged M1 and row-parallel kernels above remain available as oracles.
template<int N,int K,int Interval,int Warps>
__global__ void gemm_prefill_m16(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,const uint32_t* position){
 static_assert(N%(8*Warps)==0&&K%16==0,"fixed M16 projection geometry");
 if(*position!=127)return;
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int row=blockIdx.y*16+g,next_row=row+8,base=(blockIdx.x*Warps+warp)*8;
 float d[4]={},total[4]={};
 for(int depth=0;depth<K;depth+=16){
  uint32_t a0=pack(x[row*K+depth+2*t],x[row*K+depth+2*t+1]);
  uint32_t a1=pack(x[next_row*K+depth+2*t],x[next_row*K+depth+2*t+1]);
  uint32_t a2=pack(x[row*K+depth+2*t+8],x[row*K+depth+2*t+9]);
  uint32_t a3=pack(x[next_row*K+depth+2*t+8],x[next_row*K+depth+2*t+9]);
  uint32_t b=pack(w[(base+g)*K+depth+2*t],w[(base+g)*K+depth+2*t+1]),bb=pack(w[(base+g)*K+depth+2*t+8],w[(base+g)*K+depth+2*t+9]);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
  if constexpr(Interval>0){
   if((depth+16)%Interval==0||depth+16==K){for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.;}}
  }
 }
 if constexpr(Interval>0)for(int j=0;j<4;++j)d[j]=total[j];
 y[row*N+base+2*t]=__float2bfloat16_rn(d[0]);
 y[row*N+base+2*t+1]=__float2bfloat16_rn(d[1]);
 y[next_row*N+base+2*t]=__float2bfloat16_rn(d[2]);
 y[next_row*N+base+2*t+1]=__float2bfloat16_rn(d[3]);
}

// CUDA allocations and fixed even BF16 offsets guarantee four-byte alignment.
// Packed loads replace paired scalar loads; no floating-point operations change.
template<int N,int K,int Interval,int Warps>
__global__ void gemm_prefill_m16_vector(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,const uint32_t* position){
 static_assert(N%(8*Warps)==0&&K%16==0,"fixed M16 projection geometry");
 if(*position!=127)return;
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int row=blockIdx.y*16+g,next_row=row+8,base=(blockIdx.x*Warps+warp)*8;
 float d[4]={},total[4]={};
 // Keep the depth recurrence compact; MMA and BF16 chunk-round order are unchanged.
 #pragma unroll 1
 for(int depth=0;depth<K;depth+=16){
  uint32_t a0=*reinterpret_cast<const uint32_t*>(x+(row*K+depth+2*t));
  uint32_t a1=*reinterpret_cast<const uint32_t*>(x+(next_row*K+depth+2*t));
  uint32_t a2=*reinterpret_cast<const uint32_t*>(x+(row*K+depth+2*t+8));
  uint32_t a3=*reinterpret_cast<const uint32_t*>(x+(next_row*K+depth+2*t+8));
  uint32_t b=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t)),bb=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t+8));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
  if constexpr(Interval>0){
   if((depth+16)%Interval==0||depth+16==K){for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.;}}
  }
 }
 if constexpr(Interval>0)for(int j=0;j<4;++j)d[j]=total[j];
 y[row*N+base+2*t]=__float2bfloat16_rn(d[0]);
 y[row*N+base+2*t+1]=__float2bfloat16_rn(d[1]);
 y[next_row*N+base+2*t]=__float2bfloat16_rn(d[2]);
 y[next_row*N+base+2*t+1]=__float2bfloat16_rn(d[3]);
}

namespace riley_cuda_internal {
template<int N,int K,int Interval,int Warps>
cudaError_t launch_compiled_prefill_m16(cudaStream_t s,const void* x,const void* w,void* y,const void* position) noexcept {
 // Whole-model V5 trace regressed Q/O; keep packed loads for gate/up/down only.
 if constexpr(N==192||(N==576&&K==576)){
 gemm_prefill_m16<N,K,Interval,Warps><<<dim3(N/(8*Warps),8),32*Warps,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,(const uint32_t*)position);
 }else{
 gemm_prefill_m16_vector<N,K,Interval,Warps><<<dim3(N/(8*Warps),8),32*Warps,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,(const uint32_t*)position);
 }
 return cudaGetLastError();
}
cudaError_t enqueue_compiled_prefill_m16_gemm(cudaStream_t s,const void* x,const void* w,void* y,int n,int k,int interval,const void* position) noexcept {
 if(n==576&&k==576&&interval==192)return launch_compiled_prefill_m16<576,576,192,2>(s,x,w,y,position);
 if(n==192&&k==576&&interval==192)return launch_compiled_prefill_m16<192,576,192,1>(s,x,w,y,position);
 if(n==576&&k==576&&interval==128)return launch_compiled_prefill_m16<576,576,128,2>(s,x,w,y,position);
 if(n==1536&&k==576&&interval==0)return launch_compiled_prefill_m16<1536,576,0,4>(s,x,w,y,position);
 if(n==576&&k==1536&&interval==320)return launch_compiled_prefill_m16<576,1536,320,2>(s,x,w,y,position);
 return cudaErrorInvalidValue;
}
} // namespace riley_cuda_internal

// V11 shape primitives share the established graph owner's parent validation.
#include "prefill_shape_projection.cuh"
#include "prefill_shape_rope_kv.cuh"
namespace riley_cuda_internal {
cudaError_t enqueue_shape_prefill_gemm(cudaStream_t stream,const void* x,const void* w,void* y,int n,int k,int interval,const void* position,uint32_t rows) noexcept {
 // Keep the previously measured P128 Q/K/V/O load path; gate/up/down use the
 // tail-safe shape primitive. Other shapes use that primitive for every role.
 if(rows==128&&(n==192||(n==576&&k==576)))return enqueue_compiled_prefill_m16_gemm(stream,x,w,y,n,k,interval,position);
 auto* a=static_cast<const __nv_bfloat16*>(x);auto* b=static_cast<const __nv_bfloat16*>(w);auto* c=static_cast<__nv_bfloat16*>(y);
 if(n==576&&k==576&&interval==192)return launch_prefill_shape<576,576,192,2>(stream,a,b,c,rows);
 if(n==192&&k==576&&interval==192)return launch_prefill_shape<192,576,192,1>(stream,a,b,c,rows);
 if(n==576&&k==576&&interval==128)return launch_prefill_shape<576,576,128,2>(stream,a,b,c,rows);
 if(n==1536&&k==576&&interval==0)return launch_prefill_shape<1536,576,0,4>(stream,a,b,c,rows);
 if(n==576&&k==1536&&interval==320)return launch_prefill_shape<576,1536,320,2>(stream,a,b,c,rows);
 return cudaErrorInvalidValue;
}
cudaError_t enqueue_shape_prefill_rope_kv(cudaStream_t stream,const void* q,const void* k,const void* v,void* qo,void* keys,void* values,const void* cos,const void* sin,const void* metadata,uint32_t rows) noexcept {
 // Only the validated full P128 stage is connected so far. Do not infer new
 // request support from the wider private primitive.
 if(rows!=128)return cudaErrorInvalidValue;
 return launch_prefill_shape_rope_kv(stream,static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),static_cast<__nv_bfloat16*>(qo),static_cast<__nv_bfloat16*>(keys),static_cast<__nv_bfloat16*>(values),static_cast<const float*>(cos),static_cast<const float*>(sin),reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+16),0,128,160);
}
}

#include "prefill_shape_model.cuh"

namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v3_prefill_model(cudaStream_t s,void*const* d,const void*const* w,const void* m,void* k,void* v,const void* c,const void* sn,void* selected,uint32_t* status,uint32_t* publish,uint32_t rows,uint32_t physical,bool tiled) noexcept {
 return enqueue_v3_prefill_model(s,d,w,m,k,v,c,sn,selected,status,publish,rows,physical,tiled);
}
}

__global__ void v3_prefill_result_header(const uint32_t* m,uint32_t* result,const uint32_t* publish,const uint32_t* argmax){
 if(threadIdx.x)return;
 result[1]=*publish;result[2]=argmax[0];result[3]=argmax[1];
 for(int i=0;i<6;++i)result[4+i]=m[10+i]; // generation, replay, iteration
 for(int i=0;i<4;++i)result[10+i]=m[44+i]; // request and reservation cookie
 result[14]=m[37];result[15]=m[40];result[16]=m[38];result[17]=m[34];
 result[18]=m[41];result[19]=m[4];for(int i=0;i<8;++i)result[20+i]=m[16+i];
 result[28]=m[33];result[29]=m[42];result[30]=m[8];result[31]=0x33524d52;
}
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v3_result_header(cudaStream_t stream,const void* metadata,void* status,const void* publish,const void* argmax) noexcept {
 v3_prefill_result_header<<<1,32,0,stream>>>(static_cast<const uint32_t*>(metadata),static_cast<uint32_t*>(status),static_cast<const uint32_t*>(publish),static_cast<const uint32_t*>(argmax));return cudaGetLastError();
}
}
