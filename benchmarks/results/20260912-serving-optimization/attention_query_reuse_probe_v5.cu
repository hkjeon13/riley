
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

// Finite-input prototype only. K/V masking for nonfinite future tokens must be
// qualified before this kernel can replace the existing serving implementation.
template<int R,int S=4>
__global__ void attention_queries(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* blocks){
 int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 int head=warp%3;
 int qr=g%R,row=blockIdx.x*R+qr,kvh=blockIdx.y,qh=kvh*3+head,qb=(row*9+qh)*64;
 int count=row+1,end=(blockIdx.x+1)*R;
 __shared__ float scores[3][R][132];
 __shared__ __nv_bfloat16 probs[3][R][136];
 __shared__ float inverses[3][R];
 if(warp<3){
 float mx=-CUDART_INF_F;
 for(int token=0;token<end;token+=8){
  float d[4]={};
  for(int depth=0;depth<64;depth+=16){
   uint32_t a=pair(q[qb+depth+2*t],q[qb+depth+2*t+1]);
   uint32_t aa=pair(q[qb+depth+2*t+8],q[qb+depth+2*t+9]);
   int kb=token+g<end?cache_index(token+g,kvh,depth,blocks):0;
   uint32_t b=token+g<end?pair(k[kb+2*t],k[kb+2*t+1]):0;
   uint32_t bb=token+g<end?pair(k[kb+2*t+8],k[kb+2*t+9]):0;
   mma(d,a,a,aa,aa,b,bb);
  }
  for(int j=0;j<2;++j)if(token+2*t+j<count){float score=d[j]*.125F;mx=fmaxf(mx,score);if(g<R)scores[head][qr][token+2*t+j]=score;}
 }
 __syncwarp();
 mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
 float den=0.;
 #pragma unroll 1
 for(int j=0;j<16;++j){
  int i=2*t+j*8;
  float p0=i<count?exponential(scores[head][qr][i],mx):0.;
  float p1=i+1<count?exponential(scores[head][qr][i+1],mx):0.;
  if(i<count)den+=p0;if(i+1<count)den+=p1;
  if(g<R)*reinterpret_cast<uint32_t*>(&probs[head][qr][i])=pair(__float2bfloat16_rn(p0),__float2bfloat16_rn(p1));
 }
 __syncwarp();
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.F/den;
 if(g<R&&t==0)inverses[head][qr]=inverse;
 }
 __syncthreads();
 float inverse=inverses[head][qr];
 for(int block=(8/S)*(warp/3);block<(8/S)*(warp/3+1);++block){
  float d[4]={};
  for(int token=0;token<end;token+=16){
   uint32_t a=pair(probs[head][qr][token+2*t],probs[head][qr][token+2*t+1]);
   uint32_t aa=pair(probs[head][qr][token+2*t+8],probs[head][qr][token+2*t+9]);
   int dim=block*8+g;
   auto val=[&](int pos){return pos<end?v[cache_index(pos,kvh,dim,blocks)]:__float2bfloat16_rn(0.);};
   uint32_t b=pair(val(token+2*t),val(token+2*t+1)),bb=pair(val(token+2*t+8),val(token+2*t+9));
   mma(d,a,a,aa,aa,b,bb);
  }
  if(g<R)for(int j=0;j<2;++j)out[qb+block*8+2*t+j]=__float2bfloat16_rn(d[j]*inverse);
 }
}

#include <vector>
#include <cstdio>
#include <cstdlib>
void check(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));exit(2);}}
int main(){
 std::vector<__nv_bfloat16> q(128*576),k(20*16*192),v(k.size()),a(q.size()),b(q.size());
 std::vector<uint32_t> blocks={9,2,15,0,18,4,12,6};__nv_bfloat16 *dq,*dk,*dv,*da,*db;uint32_t* dt;
 check(cudaMalloc(&dq,q.size()*2));check(cudaMalloc(&dk,k.size()*2));check(cudaMalloc(&dv,v.size()*2));check(cudaMalloc(&da,a.size()*2));check(cudaMalloc(&db,b.size()*2));check(cudaMalloc(&dt,blocks.size()*4));check(cudaMemcpy(dt,blocks.data(),blocks.size()*4,cudaMemcpyHostToDevice));
 uint32_t state=271828;size_t mismatches=0;
 for(int seed=0;seed<24;++seed){
  for(auto* vec:{&q,&k,&v})for(auto& x:*vec){state=1664525*state+1013904223;float scale=seed%4==0?.05F:seed%4==1?.5F:seed%4==2?2.F:8.F;x=__float2bfloat16_rn((int(state%4097)-2048)/1024.F*scale);}
  if(seed==0){for(auto& x:q)x=__float2bfloat16_rn(0.);for(size_t i=0;i<v.size();++i)v[i]=__ushort_as_bfloat16(i%2?0x8000:0);}
  check(cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(dk,k.data(),k.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(dv,v.data(),v.size()*2,cudaMemcpyHostToDevice));
  attention<<<dim3(128,3),96>>>(dq,dk,dv,da,128,128,nullptr,dt);check(cudaGetLastError());check(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));
  for(int rows:{2,4,8}){
   check(cudaMemset(db,0xa5,b.size()*2));
   if(rows==2)attention_queries<2><<<dim3(64,3),384>>>(dq,dk,dv,db,dt);else if(rows==4)attention_queries<4><<<dim3(32,3),384>>>(dq,dk,dv,db,dt);else attention_queries<8><<<dim3(16,3),384>>>(dq,dk,dv,db,dt);
   check(cudaGetLastError());check(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));
   size_t diff=0,first=0;for(size_t i=0;i<a.size();++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){if(!diff)first=i;++diff;}
   printf("{\"seed\":%d,\"rows\":%d,\"mismatches\":%zu,\"first\":%zu,\"old_bits\":%u,\"new_bits\":%u}\n",seed,rows,diff,first,unsigned(__bfloat16_as_ushort(a[first])),unsigned(__bfloat16_as_ushort(b[first])));mismatches+=diff;
  }
 }
 if(mismatches)return 3;
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
 for(int variant:{0,2,4,8,8,4,2,0}){
  auto launch=[&]{if(variant==0)attention<<<dim3(128,3),96>>>(dq,dk,dv,da,128,128,nullptr,dt);else if(variant==2)attention_queries<2><<<dim3(64,3),384>>>(dq,dk,dv,db,dt);else if(variant==4)attention_queries<4><<<dim3(32,3),384>>>(dq,dk,dv,db,dt);else attention_queries<8><<<dim3(16,3),384>>>(dq,dk,dv,db,dt);};
  for(int i=0;i<20;++i)launch();check(cudaEventRecord(start));for(int i=0;i<100;++i)launch();check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));printf("{\"variant\":%d,\"us\":%.4f}\n",variant,ms*10);
 }
 check(cudaEventDestroy(start));check(cudaEventDestroy(end));for(void* p:{(void*)dq,(void*)dk,(void*)dv,(void*)da,(void*)db,(void*)dt})check(cudaFree(p));
}
