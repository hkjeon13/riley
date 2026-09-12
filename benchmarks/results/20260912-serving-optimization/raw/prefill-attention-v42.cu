#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
namespace oracle {
__device__ float exponential(float score,float maximum){
 return exp2f(fmaf(score,1.4426950408889634F,-maximum*1.4426950408889634F));
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
 if(n<rows || n>4096){out[qb+lane]=__float2bfloat16_rn(CUDART_NAN_F);out[qb+lane+32]=__float2bfloat16_rn(CUDART_NAN_F);return;}
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
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  float local_den=0.;
  for(int i=lane;i<128;i+=32){
   float p=i<end-begin?exponential(scores[warp][i],mx):0.;
   local_den+=p;probs[warp][i]=__float2bfloat16_rn(p);
  }
  for(int offset=16;offset>0;offset>>=1)local_den+=__shfl_xor_sync(0xffffffff,local_den,offset);
  local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
    if(i<end-begin)local_den+=exponential(scores[warp][i],mx);}
  den=local_den;
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
 den+=__shfl_xor_sync(0xffffffff,den,2);
 den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.0F/den;
 if(group==0)for(int block=0;block<8;++block)for(int j=0;j<2;++j)
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}
}

#include "prefill_shape_attention.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T* upload(const std::vector<T>& a){T* p;ck(cudaMalloc(&p,a.size()*sizeof(T)));ck(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
float bf(unsigned short x){return __bfloat162float(__ushort_as_bfloat16(x));}
void run(int rows,int start){
 const int context=4096,blocks=256,end=start+rows;std::vector<unsigned short>q(rows*576),k(context*192),v(k.size()),a(q.size()),b(q.size());std::vector<unsigned>pages(blocks);
 for(int i=0;i<blocks;++i)pages[i]=(i*5+17)%blocks;
 auto at=[&](int p,int h,int d){return ((pages[p/16]*3+h)*16+p%16)*64+d;};
 unsigned rng=start+rows+131;for(auto* xs:{&q,&k,&v})for(auto& x:*xs){rng=rng*1664525+1013904223;x=__bfloat16_as_ushort(__float2bfloat16_rn(float(int(rng%2049)-1024)/2048.f));}
 auto dq=upload(q),dk=upload(k),dv=upload(v),da=upload(a),db=upload(b);auto dp=upload(pages);
 ck(riley_prefill_shape::launch(0,(__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)da,dp,start,rows,context));ck(cudaDeviceSynchronize());ck(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));
 if(end<=4096){oracle::attention<<<dim3(rows,3),96>>>((__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)db,rows,end,nullptr,dp);ck(cudaDeviceSynchronize());ck(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));if(a!=b)exit(3);}
 // Independent FP64 softmax/dot diagnostic, not full-model numerical qualification.
 // Predeclared absolute bound0.005 for bounded inputs in[-0.5,0.5].
 double maxerr=0.;int sampled=0;
 for(int row:{0,rows/2,rows-1})for(int h=0;h<9;++h){int count=start+row+1;std::vector<double>prob(count);double mx=-1e300,den=0.;
  for(int p=0;p<count;++p){double sum=0;for(int d=0;d<64;++d)sum+=double(bf(q[(row*9+h)*64+d]))*bf(k[at(p,h/3,d)]);prob[p]=sum*.125;mx=std::max(mx,prob[p]);}
  for(double& x:prob){x=exp(x-mx);den+=x;}
  for(int d=0;d<64;++d){double y=0;for(int p=0;p<count;++p)y+=prob[p]*bf(v[at(p,h/3,d)]);double value=bf(a[(row*9+h)*64+d]);if(!std::isfinite(value))exit(4);maxerr=std::max(maxerr,std::abs(value-y/den));++sampled;}
 }
 if(maxerr>0.005){fprintf(stderr,"fp64 error %g\n",maxerr);exit(5);}
 // Poison the suffix for each sampled causal query and run that query alone.
 for(int row:{0,rows/2,rows-1}){auto pk=k,pv=v;int count=start+row+1;for(int p=count;p<context;++p)for(int h=0;h<3;++h)for(int d=0;d<64;++d)pk[at(p,h,d)]=pv[at(p,h,d)]=0x7fc1;
  ck(cudaMemcpy(dk,pk.data(),pk.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,pv.data(),pv.size()*2,cudaMemcpyHostToDevice));ck(riley_prefill_shape::launch(0,(__nv_bfloat16*)dq+row*576,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)db,dp,count-1,1,context));ck(cudaDeviceSynchronize());ck(cudaMemcpy(b.data(),db,576*2,cudaMemcpyDeviceToHost));for(int i=0;i<576;++i)if(a[row*576+i]!=b[i])exit(6);
 }
 for(void* p:{(void*)dq,(void*)dk,(void*)dv,(void*)da,(void*)db,(void*)dp})ck(cudaFree(p));
 printf("{\"rows\":%d,\"start\":%d,\"old_exact_checked\":%s,\"fp64_sampled_elements\":%d,\"max_abs_error\":%.9g,\"causal_suffix_nan_exact\":true}\n",rows,start,end<=4096?"true":"false",sampled,maxerr);
}
int main(){for(int start:{0,13,128,1024})for(int rows:{1,17,127,129,398})run(rows,start);run(1,4095);run(1024,3072);}
