from pathlib import Path
s=Path('/tmp/riley-multisequence-integration-20260912/kernels/src/graph_numerics.cu').read_text();s=s[:s.index('\nnamespace riley_cuda_internal {')].replace('#include "ffi_internal.hpp"','')
new=r'''
// Finite-input prototype only. K/V masking for nonfinite future tokens must be
// qualified before this kernel can replace the existing serving implementation.
template<int R>
__global__ void attention_queries(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* blocks){
 int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 int qr=g%R,row=blockIdx.x*R+qr,kvh=blockIdx.y,qh=kvh*3+warp,qb=(row*9+qh)*64;
 int count=row+1,end=(blockIdx.x+1)*R;
 __shared__ float scores[3][R][128];
 __shared__ __nv_bfloat16 probs[3][R][128];
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
  if(g<R)for(int j=0;j<2;++j)if(token+2*t+j<count)scores[warp][qr][token+2*t+j]=d[j]*.125F;
 }
 __syncwarp();
 float mx=-CUDART_INF_F;
 for(int i=t;i<count;i+=4)mx=fmaxf(mx,scores[warp][qr][i]);
 mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
 // All score maxima are consumed before reusing float storage for exponentials.
 __syncwarp();
 if(g<R)for(int i=t;i<128;i+=4){
  float p=i<count?exponential(scores[warp][qr][i],mx):0.;scores[warp][qr][i]=p;probs[warp][qr][i]=__float2bfloat16_rn(p);
 }
 __syncwarp();
 float den=0.;
 for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<count)den+=scores[warp][qr][i];}
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.F/den;
 for(int block=0;block<8;++block){
  float d[4]={};
  for(int token=0;token<end;token+=16){
   uint32_t a=pair(probs[warp][qr][token+2*t],probs[warp][qr][token+2*t+1]);
   uint32_t aa=pair(probs[warp][qr][token+2*t+8],probs[warp][qr][token+2*t+9]);
   int dim=block*8+g;
   auto val=[&](int pos){return pos<end?v[cache_index(pos,kvh,dim,blocks)]:__float2bfloat16_rn(0.);};
   uint32_t b=pair(val(token+2*t),val(token+2*t+1)),bb=pair(val(token+2*t+8),val(token+2*t+9));
   mma(d,a,a,aa,aa,b,bb);
  }
  if(g<R)for(int j=0;j<2;++j)out[qb+block*8+2*t+j]=__float2bfloat16_rn(d[j]*inverse);
 }
}
'''
tail=r'''
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
  for(int rows:{4,8}){
   check(cudaMemset(db,0xa5,b.size()*2));
   if(rows==4)attention_queries<4><<<dim3(32,3),96>>>(dq,dk,dv,db,dt);else attention_queries<8><<<dim3(16,3),96>>>(dq,dk,dv,db,dt);
   check(cudaGetLastError());check(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));
   size_t diff=0,first=0;for(size_t i=0;i<a.size();++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){if(!diff)first=i;++diff;}
   printf("{\"seed\":%d,\"rows\":%d,\"mismatches\":%zu,\"first\":%zu,\"old_bits\":%u,\"new_bits\":%u}\n",seed,rows,diff,first,unsigned(__bfloat16_as_ushort(a[first])),unsigned(__bfloat16_as_ushort(b[first])));mismatches+=diff;
  }
 }
 if(mismatches)return 3;
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
 for(int variant:{0,4,8,8,4,0}){
  auto launch=[&]{if(variant==0)attention<<<dim3(128,3),96>>>(dq,dk,dv,da,128,128,nullptr,dt);else if(variant==4)attention_queries<4><<<dim3(32,3),96>>>(dq,dk,dv,db,dt);else attention_queries<8><<<dim3(16,3),96>>>(dq,dk,dv,db,dt);};
  for(int i=0;i<20;++i)launch();check(cudaEventRecord(start));for(int i=0;i<100;++i)launch();check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));printf("{\"variant\":%d,\"us\":%.4f}\n",variant,ms*10);
 }
 check(cudaEventDestroy(start));check(cudaEventDestroy(end));for(void* p:{(void*)dq,(void*)dk,(void*)dv,(void*)da,(void*)db,(void*)dt})check(cudaFree(p));
}
'''
Path('/tmp/attention_query_reuse_probe.cu').write_text(s+new+tail)
